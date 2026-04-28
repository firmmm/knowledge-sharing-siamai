"""
thai_data_processor.py

NeMo-RL data processor for Thai language GRPO training.

Registers a processor named "thai_processor" that converts raw dataset records
into DatumSpec objects consumed by the NeMo RL training loop.

Expected input record (JSONL) format
--------------------------------------
{
  "messages": [
    {"role": "user",    "content": "..."},   # one or more turns
    {"role": "assistant","content": "..."}   # optional prior turn(s)
  ],
  "answer":    "42",                         # ground-truth answer
  "task_name": "thai_kl"                     # optional, defaults to "thai_kl"
}

Usage in YAML config
--------------------
data:
  train_data_path: path/to/train.jsonl
  val_data_path:   path/to/val.jsonl
  processor: thai_processor               # matches register_processor() name below
  max_seq_length: 2048
"""

from __future__ import annotations

from typing import Any

from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.processors import register_processor


def thai_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """
    Convert one raw dataset record into a DatumSpec.

    Parameters
    ----------
    datum_dict:     A single record loaded from your JSONL dataset.
    task_data_spec: Task-level config (system_prompt, prompt template, etc.).
    tokenizer:      The HuggingFace tokenizer for the policy model.
    max_seq_length: Maximum token budget for the whole message log.
    idx:            Dataset index (required by DatumSpec).

    Returns
    -------
    DatumSpec with:
      message_log      – tokenised conversation ready for the policy.
      extra_env_info   – {"ground_truth": answer} passed to the environment.
      length           – total token count (used for batching / packing).
      loss_multiplier  – 1.0 normally; 0.0 when the sample is over-length.
      idx              – dataset index.
      task_name        – task identifier used to route to the right environment.
    """

    # ── 1. Extract fields ──────────────────────────────────────────────────
    messages: list[dict[str, str]] = datum_dict["messages"]
    answer: str = datum_dict["answer"]
    task_name: str = datum_dict.get("task_name", "thai_kl")

    # ── 2. Build message_log ───────────────────────────────────────────────
    message_log: list[dict[str, Any]] = []

    # Optional system prompt from task config
    if task_data_spec.system_prompt:
        sys_msg = {"role": "system", "content": task_data_spec.system_prompt}
        sys_text: str = tokenizer.apply_chat_template(
            [sys_msg],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_msg_out = {
            "role": "system",
            "content": sys_text,
            "token_ids": tokenizer(
                sys_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0],
        }
        message_log.append(sys_msg_out)

    # User / assistant turns
    for i, msg in enumerate(messages):
        role: str = msg["role"]
        content: str = msg["content"]

        # Apply optional prompt template to the *first* user message
        if role == "user" and i == 0 and task_data_spec.prompt:
            content = task_data_spec.prompt.format(content)

        # add_generation_prompt=True only after the final user turn
        is_last_user = role == "user" and (
            i == len(messages) - 1
            or all(m["role"] != "user" for m in messages[i + 1 :])
        )

        chat_text: str = tokenizer.apply_chat_template(
            [{"role": role, "content": content}],
            tokenize=False,
            add_generation_prompt=is_last_user,
            add_special_tokens=False,
        )

        message_log.append(
            {
                "role": role,
                "content": chat_text,
                "token_ids": tokenizer(
                    chat_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0],
            }
        )

    # ── 3. Length and truncation ───────────────────────────────────────────
    length: int = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier: float = 1.0

    if length > max_seq_length:
        # Hard truncation: keep at most max_seq_length // n_turns tokens per turn
        n = max(len(message_log), 1)
        per_turn_budget = max(4, max_seq_length // n)
        for msg in message_log:
            msg["token_ids"] = msg["token_ids"][:per_turn_budget]
        length = sum(len(m["token_ids"]) for m in message_log)
        loss_multiplier = 0.0  # mask this sample in the loss

    # ── 4. Return DatumSpec ────────────────────────────────────────────────
    return DatumSpec(
        message_log=message_log,
        length=length,
        extra_env_info={"ground_truth": answer},
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name=task_name,
    )


# Register so NeMo RL can find it by name from the YAML config
register_processor("thai_processor", thai_processor)
