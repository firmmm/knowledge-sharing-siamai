"""
thai_reward_env.py

Dual-model NeMo-RL environment:
  rewards[:, 0] — Thai correctness  via Typhoon2.5-Qwen3-4B  (generative judge)
  rewards[:, 1] — Format quality    via Skywork-Reward-V2     (Bradley-Terry RM)

rewards shape: (batch_size, 2)  ← required by GDPO adv_estimator
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

try:
    import ray
    from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
    from nemo_rl.environments.utils import register_env
    _NEMO_AVAILABLE = True
except ImportError:
    _NEMO_AVAILABLE = False
    ray = None
    EnvironmentInterface = object

    class EnvironmentReturn:
        def __init__(self, observations, metadata, next_stop_strings,
                     rewards, terminateds, answers):
            self.observations = observations
            self.metadata = metadata
            self.next_stop_strings = next_stop_strings
            self.rewards = rewards
            self.terminateds = terminateds
            self.answers = answers

    def register_env(name, fqn):
        pass


# ── helpers ────────────────────────────────────────────────────────────────

_BOXED_RE   = re.compile(r"\\boxed\{[^}]+\}")
_THAI_RANGE = re.compile(r"[\u0E00-\u0E7F]")


def _extract_last_assistant(message_log: List[Dict[str, Any]]) -> str:
    for msg in reversed(message_log):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _extract_answer(response: str) -> str:
    boxed = _BOXED_RE.findall(response)
    if boxed:
        return re.sub(r"^\\boxed\{|\}$", "", boxed[-1])
    for line in reversed(response.splitlines()):
        if _THAI_RANGE.search(line) and line.strip():
            return line.strip()
    return response.strip()


# ── config ─────────────────────────────────────────────────────────────────

@dataclass
class DualRewardConfig:
    correctness_model_name: str = "typhoon-ai/typhoon2.5-qwen3-4b"
    format_model_name:      str = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    required_format:        str = "boxed"
    stop_strings:     List[str] = field(default_factory=lambda: ["</s>", "<|end|>"])
    max_response_length:    int = 2048
    correctness_prompt:     str = (
        "คุณเป็นผู้ตัดสินที่เชี่ยวชาญ กรุณาประเมินว่าคำตอบต่อไปนี้ถูกต้องหรือไม่\n\n"
        "คำถาม: {prompt}\n\n"
        "คำตอบ: {response}\n\n"
        "ตอบเพียง 'ใช่' ถ้าคำตอบถูกต้อง หรือ 'ไม่' ถ้าคำตอบผิด"
    )


# ── environment actor ──────────────────────────────────────────────────────

if _NEMO_AVAILABLE and ray is not None:

    @ray.remote
    class DualRewardEnvironmentActor(EnvironmentInterface):
        """
        step(message_log_batch, metadata) → EnvironmentReturn
          rewards shape: (B, 2)
            [:, 0] Thai correctness  — Typhoon2.5 generative judge  [0 or 1]
            [:, 1] Format quality    — Skywork Bradley-Terry RM      [0.0–1.0]
        """

        def __init__(self, config: Dict[str, Any]) -> None:
            cfg = DualRewardConfig(**{
                k: v for k, v in config.items()
                if k in DualRewardConfig.__dataclass_fields__
            }) if isinstance(config, dict) else config
            self.config: DualRewardConfig = cfg

            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            # ── FIX: Skywork = SequenceClassification (Bradley-Terry scalar) ──
            # Requires chat-formatted (prompt + response) pair as input.
            self.skywork_tokenizer = AutoTokenizer.from_pretrained(
                cfg.format_model_name
            )
            self.skywork_model = AutoModelForSequenceClassification.from_pretrained(
                cfg.format_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                num_labels=1,
            )
            self.skywork_model.eval()

            # ── FIX: Typhoon = CausalLM (generative), NOT a classifier ────────
            # We use next-token log-probs of "ใช่"/"ไม่" for a single forward pass.
            self.typhoon_tokenizer = AutoTokenizer.from_pretrained(
                cfg.correctness_model_name
            )
            self.typhoon_model = AutoModelForCausalLM.from_pretrained(
                cfg.correctness_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.typhoon_model.eval()

            # Pre-cache token ids for yes/no tokens
            self._yes_ids = self.typhoon_tokenizer.encode("ใช่", add_special_tokens=False)
            self._no_ids  = self.typhoon_tokenizer.encode("ไม่",  add_special_tokens=False)

        # ── NeMo RL interface ──────────────────────────────────────────────

        def step(
            self,
            message_log_batch: List[List[Dict[str, Any]]],  # FIX: correct signature
            metadata:          List[Dict[str, Any]],
        ) -> EnvironmentReturn:

            batch_size = len(message_log_batch)
            responses, prompts = [], []

            for message_log in message_log_batch:
                responses.append(_extract_last_assistant(message_log))
                user_turns = [
                    m.get("content", "") for m in message_log
                    if m.get("role") == "user"
                ]
                prompts.append(" ".join(user_turns))

            ground_truths = [
                (m.get("ground_truth") if isinstance(m, dict) else None)
                for m in metadata
            ]

            correctness = self._score_correctness_typhoon(prompts, responses, ground_truths)
            formatting  = self._score_format_skywork(prompts, responses)

            rewards = torch.stack(
                [
                    torch.tensor(correctness, dtype=torch.float32),
                    torch.tensor(formatting,  dtype=torch.float32),
                ],
                dim=1,
            )  # shape: (B, 2)

            return EnvironmentReturn(
                observations=[{"role": "assistant", "content": ""} for _ in range(batch_size)],
                metadata=metadata,
                next_stop_strings=[self.config.stop_strings] * batch_size,
                rewards=rewards,
                terminateds=torch.ones(batch_size, dtype=torch.bool),
                answers=[_extract_answer(r) for r in responses],
            )

        def batch_postprocess(self, batch):
            return batch, {}

        def global_post_process_and_metrics(self, all_rollout_data):
            """
            Required by EnvironmentInterface.
            Called once per training step after all rollouts are collected.
            Returns (processed_data, metrics_dict).
            We pass data through unchanged and return per-reward-dim mean scores
            for logging in W&B / tensorboard.
            """
            metrics = {}
            try:
                rewards = all_rollout_data.get("rewards")
                if rewards is not None and rewards.ndim == 2:
                    metrics["reward/thai_correctness"] = rewards[:, 0].mean().item()
                    metrics["reward/format_quality"]   = rewards[:, 1].mean().item()
                    metrics["reward/combined"]         = rewards.mean().item()
            except Exception:
                pass
            return all_rollout_data, metrics

        def shutdown(self):
            pass

        # ── Typhoon correctness scorer ─────────────────────────────────────

        def _score_correctness_typhoon(
            self,
            prompts:       List[str],
            responses:     List[str],
            ground_truths: List[Optional[str]],
        ) -> List[float]:
            """
            One forward pass per sample.
            Compares log-prob("ใช่") vs log-prob("ไม่") at the next token position.
            Returns 1.0 if yes wins, 0.0 if no wins, 0.5 if tokens not found.
            """
            scores: List[float] = []

            for prompt, response, gt in zip(prompts, responses, ground_truths):
                judge_text = self.config.correctness_prompt.format(
                    prompt=prompt, response=response
                )
                if gt:
                    judge_text += f"\n\nเฉลย: {gt}"

                inputs = self.typhoon_tokenizer(
                    self.typhoon_tokenizer.apply_chat_template(
                        [{"role": "user", "content": judge_text}],
                        tokenize=False,
                        add_generation_prompt=True,
                    ),
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_response_length,
                ).to(self.typhoon_model.device)

                with torch.no_grad():
                    logits   = self.typhoon_model(**inputs).logits[:, -1, :]
                    log_prob = torch.log_softmax(logits[0], dim=-1)

                yes_id = self._yes_ids[0] if self._yes_ids else None
                no_id  = self._no_ids[0]  if self._no_ids  else None

                if yes_id and no_id:
                    scores.append(
                        1.0 if log_prob[yes_id].item() > log_prob[no_id].item() else 0.0
                    )
                else:
                    scores.append(0.5)

            return scores

        # ── Skywork format scorer ──────────────────────────────────────────

        def _score_format_skywork(
            self,
            prompts:   List[str],
            responses: List[str],
        ) -> List[float]:
            """
            Skywork requires a chat-formatted (prompt, response) pair — NOT raw strings.
            Returns sigmoid(logit) so scores live in [0, 1].
            """
            raw: List[float] = []

            for prompt, response in zip(prompts, responses):
                # FIX: format as a conversation, not as a plain string
                input_text = self.skywork_tokenizer.apply_chat_template(
                    [
                        {"role": "user",      "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    tokenize=False,
                )
                inputs = self.skywork_tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_response_length,
                ).to(self.skywork_model.device)

                with torch.no_grad():
                    raw.append(self.skywork_model(**inputs).logits[0, 0].item())

            # sigmoid maps Bradley-Terry logits → [0, 1]
            return torch.sigmoid(torch.tensor(raw)).tolist()

else:
    class DualRewardEnvironmentActor(EnvironmentInterface):  # type: ignore[no-redef]
        def __init__(self, config):
            self.config = config


# ── FIX: register_env requires a string FQN, not the class object ─────────
register_env(
    "thai_reward_env",
    "thai_reward_env.DualRewardEnvironmentActor",
)
