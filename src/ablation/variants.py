import sys
from pathlib import Path

import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.official_stacks.defensivetoken.core import (
    CHAT_TEMPLATES,
    SUPPORTED_MODELS,
    load_defensive_tokens,
    resolve_defended_model_path,
)

_BASE_MODEL = SUPPORTED_MODELS[0]
_STACK_ROOT = Path(__file__).resolve().parents[2] / "src" / "official_stacks" / "defensivetoken"

_DT_TOKENS = [f"[DefensiveToken{i}]" for i in range(5)]
_MSG_BLOCK = "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] | trim + '\\n\\n<|im_end|>\\n' }}\n"
_GEN_BLOCK = "{%- if add_generation_prompt %}\n{{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"


def _token_output(tokens: list[str]) -> str:
    return "{{- '" + "".join(tokens) + "' }}\n"


def _token_block(tokens: list[str]) -> str:
    return "{%- if add_defensive_tokens %}\n" + _token_output(tokens) + "{%- endif %}\n"


def _messages_block() -> str:
    return "{%- for message in messages %}\n" + _MSG_BLOCK + "{%- endfor %}\n"


def _sandwich_tail_template(tail_count: int) -> str:
    return (
        _token_block(_DT_TOKENS)
        + _messages_block()
        + _token_block(_DT_TOKENS[-tail_count:])
        + _GEN_BLOCK
    )


POSITION_TEMPLATES = {
    "prefix": CHAT_TEMPLATES[_BASE_MODEL],
    "sandwich_tail_1": _sandwich_tail_template(1),
    "sandwich_tail_2": _sandwich_tail_template(2),
    "sandwich_tail_3": _sandwich_tail_template(3),
    "sandwich_tail_5": _sandwich_tail_template(5),
}

POSITION_METADATA = {
    "prefix": {"learned_tokens": 5, "prefix_tokens": 5, "tail_tokens": 0, "insertion_sites": 1},
    "sandwich_tail_1": {"learned_tokens": 5, "prefix_tokens": 5, "tail_tokens": 1, "insertion_sites": 2},
    "sandwich_tail_2": {"learned_tokens": 5, "prefix_tokens": 5, "tail_tokens": 2, "insertion_sites": 2},
    "sandwich_tail_3": {"learned_tokens": 5, "prefix_tokens": 5, "tail_tokens": 3, "insertion_sites": 2},
    "sandwich_tail_5": {"learned_tokens": 5, "prefix_tokens": 5, "tail_tokens": 5, "insertion_sites": 2},
}

ALL_POSITIONS = list(POSITION_TEMPLATES)


def resolve_position_variant_path(model_name: str, position: str, output_root: Path | None = None) -> Path:
    root = output_root or _STACK_ROOT
    return root / f"{model_name}-5tokens-{position}"


def prepare_position_variant(
    model_name: str,
    position: str,
    output_root: Path | None = None,
) -> Path:
    if model_name != _BASE_MODEL:
        raise ValueError(f"Unsupported model: {model_name}")
    if position not in POSITION_TEMPLATES:
        raise ValueError(f"Unknown position '{position}'. Choose from: {ALL_POSITIONS}")

    # For prefix, reuse the paper's existing model if available.
    if position == "prefix":
        paper_model = resolve_defended_model_path(model_name, output_root)
        if paper_model.exists():
            return paper_model

    output_dir = resolve_position_variant_path(model_name, position, output_root)
    if output_dir.exists():
        return output_dir

    defensive_tokens = load_defensive_tokens()
    if model_name not in defensive_tokens:
        raise ValueError(f"No defensive token vectors for: {model_name}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    defensive_tensor = torch.tensor(defensive_tokens[model_name], device=model.device)
    n = len(defensive_tensor)
    additional_special_tokens = [f"[DefensiveToken{i}]" for i in range(n)]
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    model.resize_token_embeddings(len(tokenizer))
    for i in range(n):
        model.get_input_embeddings().weight.data[-n + i] = defensive_tensor[i : i + 1]

    tokenizer.chat_template = POSITION_TEMPLATES[position]
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir
