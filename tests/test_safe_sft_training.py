from __future__ import annotations

import math

import pytest

from training.train_sft_atomicvision_safe import (
    assert_finite_number,
    parse_tool_call_text,
    render_chat_prompt_with_disabled_thinking,
    summarize_masked_examples,
    tokenize_with_assistant_mask,
    validate_sft_rows,
)


class TinyTokenizer:
    chat_template = "tiny"
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = "<pad>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        rendered = ""
        for message in messages:
            rendered += f"<{message['role']}>\n{message['content']}\n"
        if add_generation_prompt:
            rendered += "<assistant>\n"
        return rendered

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [(ord(char) % 251) + 1 for char in text]}


class ThinkingAwareTokenizer(TinyTokenizer):
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=None,
    ):
        self.calls.append(enable_thinking)
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


def _valid_row(sample_id: str = "medium-0-ask_prior"):
    return {
        "sample_id": sample_id,
        "sample_type": "ask_prior",
        "target_tool_name": "ask_prior",
        "messages": [
            {"role": "system", "content": "Use AtomicVision tools."},
            {"role": "user", "content": "Observation: synthetic case"},
            {
                "role": "assistant",
                "content": '<tool_call>{"name":"ask_prior","arguments":{}}</tool_call>',
            },
        ],
    }


def test_validate_sft_rows_accepts_atomicvision_tool_call():
    stats = validate_sft_rows([_valid_row()])

    assert stats.rows == 1
    assert stats.sample_counts == {"ask_prior": 1}
    assert stats.final_tool_counts == {"ask_prior": 1}


def test_validate_sft_rows_rejects_non_assistant_final_message():
    row = _valid_row()
    row["messages"][-1] = {"role": "user", "content": "not a target"}

    with pytest.raises(ValueError, match="final message must be assistant"):
        validate_sft_rows([row])


def test_validate_sft_rows_rejects_target_tool_mismatch():
    row = _valid_row()
    row["target_tool_name"] = "submit_defect_map"

    with pytest.raises(ValueError, match="does not match target_tool_name"):
        validate_sft_rows([row])


def test_parse_tool_call_text_rejects_bad_json():
    with pytest.raises(ValueError, match="invalid tool_call JSON"):
        parse_tool_call_text("<tool_call>{bad-json}</tool_call>", row_id="bad-row")


def test_tokenize_with_assistant_mask_has_trainable_labels():
    example = tokenize_with_assistant_mask(_valid_row(), TinyTokenizer(), max_length=256)

    assert example.valid_label_tokens > 0
    assert any(label != -100 for label in example.labels)
    assert all(label == -100 for label in example.labels[:5])


def test_tokenize_with_assistant_mask_preserves_labels_after_left_truncation():
    row = _valid_row()
    row["messages"][1]["content"] = "Observation: " + ("very long context " * 100)

    example = tokenize_with_assistant_mask(row, TinyTokenizer(), max_length=64)

    assert example.was_truncated is True
    assert example.valid_label_tokens > 0


def test_summarize_masked_examples_counts_truncation_and_labels():
    examples = [
        tokenize_with_assistant_mask(_valid_row("row-1"), TinyTokenizer(), max_length=256),
        tokenize_with_assistant_mask(_valid_row("row-2"), TinyTokenizer(), max_length=256),
    ]

    stats = summarize_masked_examples(examples, max_length=256)

    assert stats.examples == 2
    assert stats.min_label_tokens > 0
    assert stats.mean_label_tokens > 0
    assert stats.max_length == 256


def test_render_chat_prompt_disables_thinking_when_supported():
    tokenizer = ThinkingAwareTokenizer()

    prompt = render_chat_prompt_with_disabled_thinking(
        tokenizer,
        _valid_row()["messages"][:-1],
        add_generation_prompt=True,
    )

    assert "<assistant>" in prompt
    assert tokenizer.calls == [False]


def test_assert_finite_number_rejects_nan_and_inf():
    with pytest.raises(FloatingPointError, match="loss"):
        assert_finite_number(math.nan, "loss")

    with pytest.raises(FloatingPointError, match="grad_norm"):
        assert_finite_number(math.inf, "grad_norm")

    assert_finite_number(1.25, "loss")
