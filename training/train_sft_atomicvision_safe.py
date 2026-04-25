"""NaN-safe AtomicVision assistant-masked SFT trainer.

This script is intentionally conservative. It refuses to train when the JSONL
format is malformed, when assistant masking produces no trainable labels, or
when the first loss/any later loss is NaN or Inf. A checkpoint produced after a
NaN loss is worse than no checkpoint, so failures are hard stops.

The heavy ML imports are deferred until training time so validation helpers can
be unit-tested without requiring Torch, Transformers, PEFT, or bitsandbytes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_DATASET = "outputs/sft/atomicvision_cost_aware_masked_sft.jsonl"
DEFAULT_OUTPUT_DIR = "/kaggle/working/atomicvision-cost-aware-masked-sft-lora"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
VALID_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class SftRowStats:
    rows: int
    sample_counts: dict[str, int]
    final_tool_counts: dict[str, int]


@dataclass(frozen=True)
class MaskStats:
    examples: int
    min_label_tokens: int
    mean_label_tokens: float
    max_label_tokens: int
    truncated_examples: int
    max_length: int


@dataclass
class MaskedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    sample_id: str
    valid_label_tokens: int
    was_truncated: bool


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file and reject empty or malformed lines."""

    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"SFT JSONL not found: {jsonl_path}")
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            row.setdefault("_line_number", line_number)
            rows.append(row)
    if not rows:
        raise ValueError(f"SFT JSONL has no rows: {jsonl_path}")
    return rows


def validate_sft_rows(rows: Iterable[dict[str, Any]]) -> SftRowStats:
    """Validate AtomicVision SFT row structure before touching the GPU."""

    sample_counts: dict[str, int] = {}
    final_tool_counts: dict[str, int] = {}
    total = 0
    for index, row in enumerate(rows):
        total += 1
        row_id = str(row.get("sample_id") or f"row-{index}")
        line_number = row.get("_line_number", "?")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{row_id} line {line_number}: missing non-empty messages list")
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"{row_id}: message {message_index} is not an object")
            role = message.get("role")
            content = message.get("content")
            if role not in VALID_ROLES:
                raise ValueError(f"{row_id}: message {message_index} has invalid role {role!r}")
            if role == "assistant" and message.get("tool_calls") is not None:
                parse_tool_call_message(
                    message,
                    row_id=f"{row_id}: message {message_index}",
                )
                continue
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{row_id}: message {message_index} has empty content")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"{row_id}: final message must be assistant")
        call = parse_tool_call_message(messages[-1], row_id=row_id)
        tool_name = call.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(f"{row_id}: final tool call missing name")
        if not isinstance(call.get("arguments"), dict):
            raise ValueError(f"{row_id}: final tool call arguments must be an object")
        expected_tool = row.get("target_tool_name")
        if expected_tool is not None and tool_name != expected_tool:
            raise ValueError(
                f"{row_id}: final tool {tool_name!r} does not match target_tool_name "
                f"{expected_tool!r}",
            )
        sample_type = str(row.get("sample_type") or "unknown")
        sample_counts[sample_type] = sample_counts.get(sample_type, 0) + 1
        final_tool_counts[tool_name] = final_tool_counts.get(tool_name, 0) + 1
    if total <= 0:
        raise ValueError("No rows were provided for validation")
    return SftRowStats(
        rows=total,
        sample_counts=dict(sorted(sample_counts.items())),
        final_tool_counts=dict(sorted(final_tool_counts.items())),
    )


def parse_tool_call_text(text: str, row_id: str = "row") -> dict[str, Any]:
    """Parse the exact AtomicVision tool-call envelope."""

    start = "<tool_call>"
    end = "</tool_call>"
    if start not in text or end not in text:
        raise ValueError(f"{row_id}: final assistant content lacks <tool_call> envelope")
    payload = text.split(start, 1)[1].split(end, 1)[0].strip()
    if not payload:
        raise ValueError(f"{row_id}: empty tool_call payload")
    try:
        call = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{row_id}: invalid tool_call JSON: {exc}") from exc
    if not isinstance(call, dict):
        raise ValueError(f"{row_id}: tool_call payload is not an object")
    return call


def parse_tool_call_message(message: dict[str, Any], row_id: str = "row") -> dict[str, Any]:
    """Parse a final assistant tool call from literal or structured chat data."""

    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise ValueError(f"{row_id}: assistant tool_calls must contain exactly one entry")
        entry = tool_calls[0]
        if not isinstance(entry, dict):
            raise ValueError(f"{row_id}: assistant tool_calls entry is not an object")
        entry_type = entry.get("type")
        if entry_type not in (None, "function"):
            raise ValueError(f"{row_id}: unsupported assistant tool_calls type {entry_type!r}")
        function = entry.get("function", entry)
        if not isinstance(function, dict):
            raise ValueError(f"{row_id}: assistant tool_calls function payload is not an object")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{row_id}: assistant tool call missing name")
        if not isinstance(arguments, dict):
            raise ValueError(f"{row_id}: assistant tool call arguments must be an object")
        return {"name": name, "arguments": arguments}

    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"{row_id}: assistant message content must be a string")
    return parse_tool_call_text(content, row_id=row_id)


def build_prompt_and_target(row: dict[str, Any], tokenizer: Any) -> tuple[str, str]:
    """Render all context as prompt and the final assistant turn as target."""

    messages = row["messages"]
    prompt_messages = messages[:-1]
    target_message = messages[-1]
    structured_target = bool(target_message.get("tool_calls"))
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        if structured_target:
            rendered = render_structured_prompt_and_target(
                tokenizer,
                prompt_messages,
                target_message,
            )
            if rendered is not None:
                return rendered
        prompt = render_chat_prompt_with_disabled_thinking(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
    else:
        prompt = render_fallback_chat_prompt(prompt_messages)
    target = assistant_message_target_text(target_message, row=row)
    return prompt, target


def render_chat_prompt_with_disabled_thinking(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    """Apply the tokenizer chat template while disabling reasoning tokens when supported."""

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def render_structured_prompt_and_target(
    tokenizer: Any,
    prompt_messages: list[dict[str, Any]],
    target_message: dict[str, Any],
) -> tuple[str, str] | None:
    """Render a structured assistant tool call using the chat template itself."""

    full_messages = [*prompt_messages, target_message]
    try:
        prompt = render_chat_prompt_with_disabled_thinking(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
        full_text = render_chat_prompt_with_disabled_thinking(
            tokenizer,
            full_messages,
            add_generation_prompt=False,
        )
    except Exception:
        return None
    if not full_text.startswith(prompt):
        return None
    target = full_text[len(prompt) :]
    if not target:
        return None
    return prompt, target


def apply_training_chat_template_if_available(
    tokenizer: Any,
    *,
    get_training_chat_template_fn: Any | None = None,
) -> bool:
    """Patch known tokenizers to TRL's training-compatible chat template.

    TRL ships a Qwen3 training template that is prefix-preserving for tool use
    and stable for assistant-only loss masking. Using the same training-aware
    template in the SFT warmup keeps the prompt format closer to the later GRPO
    path.
    """

    if not getattr(tokenizer, "chat_template", None):
        return False
    if get_training_chat_template_fn is None:
        try:
            from trl.chat_template_utils import get_training_chat_template
        except Exception:
            return False
        get_training_chat_template_fn = get_training_chat_template
    training_template = get_training_chat_template_fn(tokenizer)
    if not training_template:
        return False
    tokenizer.chat_template = training_template
    return True


def render_fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Simple fallback used only if a tokenizer has no chat template."""

    rendered: list[str] = []
    for message in messages:
        rendered.append(
            f"<|{message['role']}|>\n{fallback_message_content(message).strip()}\n"
        )
    rendered.append("<|assistant|>\n")
    return "".join(rendered)


def assistant_message_target_text(message: dict[str, Any], row: dict[str, Any]) -> str:
    """Return the assistant target text, using metadata when structured."""

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    target = row.get("target_tool_call")
    if isinstance(target, str) and target.strip():
        return target
    return fallback_message_content(message)


def fallback_message_content(message: dict[str, Any]) -> str:
    """Render one message into a fallback plain-text representation."""

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if message.get("role") == "assistant" and message.get("tool_calls") is not None:
        return render_tool_call_text(parse_tool_call_message(message))
    raise ValueError(f"Unsupported fallback message content for role {message.get('role')!r}")


def render_tool_call_text(call: dict[str, Any]) -> str:
    """Render one AtomicVision tool call as the exact XML envelope."""

    return (
        "<tool_call>"
        f"{json.dumps(call, separators=(',', ':'), ensure_ascii=True)}"
        "</tool_call>"
    )


def tokenize_with_assistant_mask(
    row: dict[str, Any],
    tokenizer: Any,
    max_length: int,
) -> MaskedExample:
    """Tokenize one row while training only the final assistant tool call."""

    if max_length <= 8:
        raise ValueError("max_length must be greater than 8")
    sample_id = str(row.get("sample_id") or row.get("_line_number") or "row")
    prompt, target = build_prompt_and_target(row, tokenizer)
    eos = getattr(tokenizer, "eos_token", None) or ""
    full_text = prompt + target + eos
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if not full_ids:
        raise ValueError(f"{sample_id}: tokenized text is empty")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError(
            f"{sample_id}: assistant target produced no extra tokens "
            f"(prompt={len(prompt_ids)}, full={len(full_ids)})",
        )
    labels = [-100] * len(full_ids)
    for position in range(len(prompt_ids), len(full_ids)):
        labels[position] = full_ids[position]

    was_truncated = False
    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        full_ids = full_ids[overflow:]
        labels = labels[overflow:]
        was_truncated = True

    valid_label_tokens = count_valid_label_tokens(labels)
    if valid_label_tokens <= 0:
        raise ValueError(
            f"{sample_id}: assistant masking left zero valid label tokens after "
            f"truncation. Increase max_length or inspect this row.",
        )
    return MaskedExample(
        input_ids=list(full_ids),
        attention_mask=[1] * len(full_ids),
        labels=list(labels),
        sample_id=sample_id,
        valid_label_tokens=valid_label_tokens,
        was_truncated=was_truncated,
    )


def count_valid_label_tokens(labels: Iterable[int]) -> int:
    return sum(1 for label in labels if int(label) != -100)


def summarize_masked_examples(examples: list[MaskedExample], max_length: int) -> MaskStats:
    if not examples:
        raise ValueError("No tokenized examples available")
    label_counts = [example.valid_label_tokens for example in examples]
    return MaskStats(
        examples=len(examples),
        min_label_tokens=min(label_counts),
        mean_label_tokens=mean(label_counts),
        max_label_tokens=max(label_counts),
        truncated_examples=sum(1 for example in examples if example.was_truncated),
        max_length=max_length,
    )


def assert_finite_number(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise FloatingPointError(f"{name} is not finite: {value!r}")


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run NaN-safe QLoRA SFT training."""

    rows = load_jsonl(args.dataset_jsonl)
    row_stats = validate_sft_rows(rows)
    print("DATASET VALIDATION PASSED")
    print(json.dumps(row_stats.__dict__, indent=2, sort_keys=True))

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for QLoRA training")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if apply_training_chat_template_if_available(tokenizer):
        print("Applied TRL training chat template for tokenizer compatibility")

    if args.max_examples:
        rows = rows[: args.max_examples]
    tokenized = [
        tokenize_with_assistant_mask(row, tokenizer=tokenizer, max_length=args.max_length)
        for row in rows
    ]
    mask_stats = summarize_masked_examples(tokenized, max_length=args.max_length)
    print("ASSISTANT MASK VALIDATION PASSED")
    print(json.dumps(mask_stats.__dict__, indent=2, sort_keys=True))

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    init_adapter_dir = None
    if args.init_adapter_dir:
        init_adapter_dir = Path(args.init_adapter_dir).resolve()
        if not init_adapter_dir.exists():
            raise FileNotFoundError(f"Initial adapter directory not found: {init_adapter_dir}")
        model = PeftModel.from_pretrained(model, str(init_adapter_dir), is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(args.target_modules),
        )
        model = get_peft_model(model, peft_config)
    model.train()
    model.print_trainable_parameters()

    dataset = MaskedSftDataset(tokenized)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_masked_batch(batch, tokenizer.pad_token_id),
    )
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    total_micro_steps = args.max_updates * args.grad_accum
    checkpoint_steps = set(args.checkpoint_steps)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite_output_dir:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iterator = cycling_iterator(dataloader)
    optimizer.zero_grad(set_to_none=True)
    completed_updates = 0
    micro_step = 0
    last_losses: list[float] = []

    while completed_updates < args.max_updates:
        batch = next(iterator)
        batch = {key: value.to(model.device) for key, value in batch.items()}
        valid_labels = int((batch["labels"] != -100).sum().item())
        if valid_labels <= 0:
            raise ValueError(f"micro_step {micro_step}: batch has zero valid labels")
        outputs = model(**batch)
        raw_loss = outputs.loss
        loss_value = float(raw_loss.detach().cpu().item())
        assert_finite_number(loss_value, f"loss at micro_step {micro_step}")
        loss = raw_loss / args.grad_accum
        loss.backward()
        last_losses.append(loss_value)
        micro_step += 1

        if micro_step % args.grad_accum != 0:
            continue

        grad_norm = clip_grad_norm_(trainable_params, args.max_grad_norm)
        grad_norm_value = float(grad_norm.detach().cpu().item())
        assert_finite_number(grad_norm_value, f"grad_norm at update {completed_updates + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed_updates += 1
        window = last_losses[-args.grad_accum :]
        mean_loss = sum(window) / max(1, len(window))
        assert_finite_number(mean_loss, f"mean_loss at update {completed_updates}")

        if completed_updates == 1 or completed_updates % args.log_every == 0:
            print(
                f"update {completed_updates}/{args.max_updates} | "
                f"loss {mean_loss:.6f} | grad_norm {grad_norm_value:.6f}",
                flush=True,
            )
        if completed_updates in checkpoint_steps:
            checkpoint_dir = output_dir / f"checkpoint-{completed_updates}"
            save_adapter(model, tokenizer, checkpoint_dir)
            print(f"saved checkpoint: {checkpoint_dir}", flush=True)

    if completed_updates not in checkpoint_steps:
        final_checkpoint = output_dir / f"checkpoint-{completed_updates}"
        save_adapter(model, tokenizer, final_checkpoint)
        print(f"saved checkpoint: {final_checkpoint}", flush=True)
    save_adapter(model, tokenizer, output_dir)
    zip_path = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)

    report = {
        "status": "success",
        "model": args.model,
        "dataset_jsonl": str(args.dataset_jsonl),
        "output_dir": str(output_dir),
        "zip_path": zip_path,
        "max_updates": args.max_updates,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "init_adapter_dir": str(init_adapter_dir) if init_adapter_dir else None,
        "row_stats": row_stats.__dict__,
        "mask_stats": mask_stats.__dict__,
        "final_mean_loss": sum(last_losses[-args.grad_accum :]) / args.grad_accum,
    }
    report_path = output_dir / "safe_sft_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("TRAINING DONE")
    print(f"Adapter saved at: {output_dir}")
    print(f"Zip saved at: {zip_path}")
    print(f"Report saved at: {report_path}")
    return report


class MaskedSftDataset:
    def __init__(self, examples: list[MaskedExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "labels": example.labels,
        }


def collate_masked_batch(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Any]:
    import torch

    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def cycling_iterator(dataloader: Any) -> Iterable[Any]:
    while True:
        for batch in dataloader:
            yield batch


def save_adapter(model: Any, tokenizer: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NaN-safe AtomicVision QLoRA SFT trainer")
    parser.add_argument("--dataset-jsonl", default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-updates", type=int, default=80)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--init-adapter-dir",
        default=None,
        help="Optional existing PEFT adapter directory to continue training from.",
    )
    parser.add_argument("--target-modules", nargs="+", default=list(DEFAULT_TARGET_MODULES))
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, default=[40, 60, 80])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate JSONL structure without loading model/tokenizer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        rows = load_jsonl(args.dataset_jsonl)
        stats = validate_sft_rows(rows)
        print("DATASET VALIDATION PASSED")
        print(json.dumps(stats.__dict__, indent=2, sort_keys=True))
        return
    train(args)


if __name__ == "__main__":
    main()
