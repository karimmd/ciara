from __future__ import annotations

import os

MODELS = os.environ.get("CIARA_MODELS", "models")

BASES = {
    "q35-0.8b": ("Qwen__Qwen3.5-0.8B", "qwen35"),
    "q35-2b": ("Qwen__Qwen3.5-2B", "qwen35"),
    "q35-4b": ("Qwen__Qwen3.5-4B", "qwen35"),
    "q35-9b": ("Qwen__Qwen3.5-9B", "qwen35"),
    "q3-0.6b": ("Qwen__Qwen3-0.6B", "qwen3"),
    "q3-1.7b": ("Qwen__Qwen3-1.7B", "qwen3"),
    "q3-4b": ("Qwen__Qwen3-4B", "qwen3"),
    "q3-8b": ("Qwen__Qwen3-8B", "qwen3"),
    "m3-3b": ("mistralai__Ministral-3-3B-Instruct-2512-BF16", "ministral3"),
    "m3-8b": ("mistralai__Ministral-3-8B-Instruct-2512-BF16", "ministral3"),
    "g42-3b": ("ibm-granite__granite-4.2-3b", "granite"),
    "g42-8b": ("ibm-granite__granite-4.2-8b", "granite"),
}

CKPTS = {
    "q35-0.8b-fp16": ("Qwen__Qwen3.5-0.8B", "q35-0.8b", "fp16"),
    "q35-2b-fp16": ("Qwen__Qwen3.5-2B", "q35-2b", "fp16"),
    "q35-4b-fp16": ("Qwen__Qwen3.5-4B", "q35-4b", "fp16"),
    "q35-4b-fp8": ("RedHatAI__Qwen3.5-4B-FP8-dynamic", "q35-4b", "fp8"),
    "q35-4b-w4a16": ("RedHatAI__Qwen3.5-4B-quantized.w4a16", "q35-4b", "w4a16"),
    "q35-9b-w4a16": ("RedHatAI__Qwen3.5-9B-quantized.w4a16", "q35-9b", "w4a16"),
    "q35-9b-fp8": ("RedHatAI__Qwen3.5-9B-FP8-dynamic", "q35-9b", "fp8"),
    "q3-0.6b-fp16": ("Qwen__Qwen3-0.6B", "q3-0.6b", "fp16"),
    "q3-1.7b-fp16": ("Qwen__Qwen3-1.7B", "q3-1.7b", "fp16"),
    "q3-4b-fp16": ("Qwen__Qwen3-4B", "q3-4b", "fp16"),
    "q3-4b-fp8": ("Qwen__Qwen3-4B-FP8", "q3-4b", "fp8"),
    "q3-4b-awq": ("Qwen__Qwen3-4B-AWQ", "q3-4b", "awq"),
    "q3-8b-fp8": ("Qwen__Qwen3-8B-FP8", "q3-8b", "fp8"),
    "q3-8b-awq": ("Qwen__Qwen3-8B-AWQ", "q3-8b", "awq"),
    "m3-3b-fp16": ("mistralai__Ministral-3-3B-Instruct-2512-BF16", "m3-3b", "fp16"),
    "m3-3b-fp8": ("mistralai__Ministral-3-3B-Instruct-2512", "m3-3b", "fp8"),
    "m3-8b-awq": ("cyankiwi__Ministral-3-8B-Instruct-2512-AWQ-4bit", "m3-8b", "awq"),
    "g42-3b-fp16": ("ibm-granite__granite-4.2-3b", "g42-3b", "fp16"),
    "g42-3b-fp8": ("ibm-granite__granite-4.2-3b-fp8", "g42-3b", "fp8"),
    "g42-8b-fp8": ("ibm-granite__granite-4.2-8b-fp8", "g42-8b", "fp8"),
}
CKPTS.update({
    "q35-9b-fp16": ("Qwen__Qwen3.5-9B", "q35-9b", "fp16"),
    "q3-8b-fp16": ("Qwen__Qwen3-8B", "q3-8b", "fp16"),
    "m3-8b-fp16": ("mistralai__Ministral-3-8B-Instruct-2512-BF16", "m3-8b", "fp16"),
    "g42-8b-fp16": ("ibm-granite__granite-4.2-8b", "g42-8b", "fp16"),
})
for _b in ("q35-2b", "q35-4b", "q35-9b", "q3-4b", "q3-8b", "m3-3b", "m3-8b", "g42-3b", "g42-8b"):
    for _s in ("w8a8", "w4a16"):
        CKPTS[f"{_b}-ih-{_s}"] = (f"inhouse__{_b}-{_s}", _b, _s)

MULTIMODAL = {"qwen35", "ministral3"}
THINKING_SWITCH = {"qwen35", "qwen3"}


def family_of(base_key: str) -> str:
    return BASES[base_key][1]


def model_path(dirname: str) -> str:
    return os.path.join(MODELS, dirname)
