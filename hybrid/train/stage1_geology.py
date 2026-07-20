"""Stage 1 — geology adapter (Unsloth 4-bit LoRA SFT).

Teaches Qwen2.5-1.5B geology chain-of-thought on GeoGPT-CoT-QA, producing the
FROZEN geology adapter the main model loads. Runs as its own process: Unsloth
patches transformers/peft/trl at import, so `unsloth` MUST be imported before
them — hence a standalone script, not a function in the training pipeline.

Config is GEOLOGY_CFG (in hybrid.model.geology); its hash names the output dir,
so this script and the cached adapter stay in lockstep. Built once; the loader
`load_geology_adapter` just reads the cache afterwards.

Run:  python -m hybrid.train.stage1_geology

NOTE: the original was deleted with the archive; the dataset formatting below is
reconstructed from GEOLOGY_CFG + project notes. The cached adapter already
exists, so re-run only to rebuild — and adjust the FIELD MAPPING in
`format_example` (and the SFTConfig/SFTTrainer kwargs) if your GeoGPT-CoT-QA
columns or trl version differ.
"""
import unsloth  # noqa: F401  -- MUST be first (patches transformers/peft/trl)
from unsloth import FastLanguageModel

import json

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from hybrid.model.geology import GEOLOGY_CFG, adapter_dir

DATASET_NAME = "GeoGPT-Research-Project/GeoGPT-CoT-QA"
DATASET_SPLIT = "train"
SEED = 42
OUT = adapter_dir(GEOLOGY_CFG)


def format_example(ex):
    """GeoGPT-CoT-QA row -> chatml messages. Keeps <think>/<answer> as PLAIN
    text (no vocab additions). Wraps the CoT in the evidence/think/answer
    skeleton the later stages expect (evidence_placeholder).

    FIELD MAPPING is a reconstruction — adjust to the dataset's real columns.
    """
    q = ex.get("question") or ex.get("instruction") or ex.get("input") or ""
    cot = ex.get("cot") or ex.get("reasoning") or ex.get("think") or ""
    ans = ex.get("answer") or ex.get("output") or ex.get("response") or ""
    if not cot and ans:          # single answer field -> treat it as the CoT
        cot, ans = ans, ""
    assistant = f"<think>{cot.strip()}</think>\n<answer>{ans.strip()}</answer>"
    return {"messages": [
        {"role": "user", "content": q.strip()},
        {"role": "assistant", "content": assistant},
    ]}


def main():
    cfg = GEOLOGY_CFG
    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    if cfg["max_train_samples"]:
        ds = ds.select(range(min(cfg["max_train_samples"], len(ds))))
    ds = ds.map(format_example, remove_columns=ds.column_names)
    ds = ds.map(lambda ex: {"text": tok.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)},
        remove_columns=["messages"])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=cfg["max_seq_length"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            num_train_epochs=cfg["num_epochs"],
            learning_rate=cfg["learning_rate"],
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=10,
            seed=SEED,
            output_dir=str(OUT / "_trainer"),
            report_to="none",
        ),
    )
    trainer.train()

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT))      # LoRA adapter only (save_mode="adapter")
    tok.save_pretrained(str(OUT))
    (OUT / "stage1_config.json").write_text(json.dumps(cfg, sort_keys=True, indent=2))
    print(f"[stage1] geology adapter saved -> {OUT}", flush=True)
    print("STAGE1_GEOLOGY_DONE", flush=True)


if __name__ == "__main__":
    main()
