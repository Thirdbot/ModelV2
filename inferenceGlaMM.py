from pathlib import Path

import torch
from torch.utils.data import DataLoader

from trainGlaMM import GLaMMTrainer, build_dataset
from utils.data import EncoderDecoderCollate


CHECKPOINT_DIR = Path("GLaMM")
NUM_SAMPLES = 5
MAX_NEW_TOKENS = 180


def find_checkpoint():
    last = CHECKPOINT_DIR / "last.ckpt"
    if last.exists():
        return last

    checkpoints = sorted(CHECKPOINT_DIR.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")
    return checkpoints[-1]


def main():
    checkpoint = find_checkpoint()
    model = GLaMMTrainer.load_from_checkpoint(checkpoint.as_posix())
    model.eval()
    model.model.dual_encoder.is_train = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collator = EncoderDecoderCollate(model.tokenizer)
    _, _, test_dataset = build_dataset()
    sample_count = min(NUM_SAMPLES, len(test_dataset))
    loader = DataLoader(
        test_dataset.select(range(sample_count)),
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )

    print(f"Loaded checkpoint: {checkpoint}")

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            heights = [size[0] for size in batch["sizes"]]
            widths = [size[1] for size in batch["sizes"]]

            dual_outputs = model.model.dual_encoder(
                pixel_values=batch["pixel_values"],
                tiles=batch["tiles"],
                bbox=None,
                H=heights,
                W=widths,
            )

            input_ids = batch["prompt_input_ids"].to(device)
            attention_mask = batch["prompt_attention_mask"].to(device)

            visual_tokens = [
                model.model.lang_decoder._build_visual_tokens(output, device)
                for output in dual_outputs
            ]
            visual_tokens = visual_tokens[0].unsqueeze(0)
            text_embeds = model.model.lang_decoder.model.get_input_embeddings()(input_ids)
            visual_tokens = visual_tokens.to(dtype=text_embeds.dtype)
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

            visual_attention = torch.ones(
                (1, visual_tokens.shape[1]),
                device=device,
                dtype=attention_mask.dtype,
            )
            full_attention_mask = torch.cat([visual_attention, attention_mask], dim=1)

            generated = model.model.lang_decoder.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
            )

            generated_text = model.tokenizer.decode(
                generated[0],
                skip_special_tokens=False,
            )
            valid_target = batch["labels"][0] != -100
            target_text = model.tokenizer.decode(
                batch["labels"][0][valid_target],
                skip_special_tokens=False,
            )

            print("=" * 80)
            print(f"sample: {idx}")
            print("TARGET:")
            print(target_text.strip())
            print("GENERATED:")
            print(generated_text.strip())


if __name__ == "__main__":
    main()
