from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import time
import traceback

MODELS = os.environ.get("CIARA_MODELS", "models")
CALIB = os.environ.get("CIARA_CALIB", os.path.join("calib", "open-platypus"))
LOGS = os.environ.get("CIARA_LOGS", os.path.join("runs", "quantization"))
N_SAMPLES, MAX_LEN, SEED = 512, 2048, 42

BASES = {
    "q35-2b": ("Qwen__Qwen3.5-2B", "Qwen3_5ForConditionalGeneration", "Qwen3_5DecoderLayer"),
    "q35-4b": ("Qwen__Qwen3.5-4B", "Qwen3_5ForConditionalGeneration", "Qwen3_5DecoderLayer"),
    "q35-9b": ("Qwen__Qwen3.5-9B", "Qwen3_5ForConditionalGeneration", "Qwen3_5DecoderLayer"),
    "q3-4b": ("Qwen__Qwen3-4B", "AutoModelForCausalLM", "Qwen3DecoderLayer"),
    "q3-8b": ("Qwen__Qwen3-8B", "AutoModelForCausalLM", "Qwen3DecoderLayer"),
    "m3-3b": ("mistralai__Ministral-3-3B-Instruct-2512-BF16", "Mistral3ForConditionalGeneration", "Ministral3DecoderLayer"),
    "m3-8b": ("mistralai__Ministral-3-8B-Instruct-2512-BF16", "Mistral3ForConditionalGeneration", "Ministral3DecoderLayer"),
    "g42-3b": ("ibm-granite__granite-4.2-3b", "AutoModelForCausalLM", "GraniteDecoderLayer"),
    "g42-8b": ("ibm-granite__granite-4.2-8b", "AutoModelForCausalLM", "GraniteDecoderLayer"),
}
IGNORE = ["re:.*lm_head", "re:.*embed_tokens$", "re:.*visual.*", "re:.*vision_tower.*",
          "re:.*multi_modal_projector.*", "re:.*linear_attn.*", "re:.*mtp.*"]


def calibration(tok):
    from datasets import load_dataset
    files = sorted(glob.glob(os.path.join(CALIB, "**/*.parquet"), recursive=True))
    ds = load_dataset("parquet", data_files=files, split="train").shuffle(seed=SEED).select(range(N_SAMPLES))

    def text(ex):
        t = ex["instruction"] + ("\n" + ex["input"] if ex.get("input") else "")
        return {"text": t}

    ds = ds.map(text)
    return ds.map(lambda s: tok(s["text"], truncation=True, max_length=MAX_LEN, add_special_tokens=False),
                  remove_columns=ds.column_names)


def main(base: str, scheme: str) -> int:
    import torch
    import transformers as T
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    dirname, cls_name, layer_cls = BASES[base]
    src = os.path.join(MODELS, dirname)
    out = os.path.join(MODELS, f"inhouse__{base}-{scheme}")
    t0 = time.time()
    model = getattr(T, cls_name).from_pretrained(src, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tok = T.AutoTokenizer.from_pretrained(src)
    kw = dict(targets="Linear", scheme=scheme.upper(), ignore=IGNORE, dampening_frac=0.05)
    if scheme == "w4a16":
        kw["actorder"] = "static"
    recipe = GPTQModifier(**kw)
    oneshot(model=model, dataset=calibration(tok), recipe=recipe, max_seq_length=MAX_LEN,
            num_calibration_samples=N_SAMPLES, processor=tok, sequential_targets=[layer_cls] if layer_cls else None)
    tmp = out + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, save_compressed=True)
    for f in os.listdir(src):
        if f.endswith(".safetensors") or f in ("config.json", "model.safetensors.index.json", "SHA256SUMS"):
            continue
        p = os.path.join(src, f)
        if os.path.isfile(p) and not os.path.exists(os.path.join(tmp, f)):
            shutil.copy2(p, tmp)
    json.dump({"base": base, "source": dirname, "scheme": scheme, "recipe": repr(recipe), "ignore": IGNORE,
               "calibration": {"dataset": "garage-bAInd/Open-Platypus", "samples": N_SAMPLES, "max_len": MAX_LEN,
                               "seed": SEED},
               "seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(os.path.join(tmp, "BUILD.json"), "w"), indent=1)
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    print(f"built {out} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    b, s = sys.argv[1], sys.argv[2]
    try:
        sys.exit(main(b, s))
    except Exception:
        traceback.print_exc()
        os.makedirs(LOGS, exist_ok=True)
        json.dump({"base": b, "scheme": s, "error": traceback.format_exc()[-4000:]},
                  open(os.path.join(LOGS, f"{b}-{s}.failed.json"), "w"), indent=1)
        sys.exit(1)
