from pathlib import Path

import torch
from torch.utils.data import DataLoader

from trainGlaMM import GLaMMTrainer, build_dataset
from utils.data import BBOX_SLOT, OBJ_SLOT, REG_SLOT, EncoderDecoderCollate


CHECKPOINT_DIR = Path("GLaMM")
NUM_SAMPLES = 5
MAX_NEW_TOKENS = 180
CLASS_NAMES = {
    0: "no_object",
    1: "fault",
    2: "closure",
    3: "salt",
    4: "onlap",
    5: "lithology",
}


def find_checkpoint():
    last = CHECKPOINT_DIR / "last.ckpt"
    if last.exists():
        return last

    checkpoints = sorted(CHECKPOINT_DIR.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")
    return checkpoints[-1]


def format_bbox(box):
    return [int(round(value)) for value in box]


def render_slots(text, object_names, bboxes, numbers):
    object_index = 0
    while OBJ_SLOT in text and object_names:
        text = text.replace(OBJ_SLOT, object_names[min(object_index, len(object_names) - 1)], 1)
        object_index += 1

    bbox_index = 0
    while BBOX_SLOT in text and bboxes:
        text = text.replace(BBOX_SLOT, str(format_bbox(bboxes[min(bbox_index, len(bboxes) - 1)])), 1)
        bbox_index += 1

    for value in numbers:
        text = text.replace(REG_SLOT, f"{value:.4g}", 1)
    return text


def strip_prompt_echo(text):
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[-1]
    return text.replace("<|im_end|>", "").strip()


def best_proposal(output):
    proposal = output["proposal"]
    scores = proposal["objectness_logits"].sigmoid()
    best_idx = scores.argmax()
    class_id = int(proposal["class_logits"][best_idx].argmax().detach().cpu().item())
    bbox = output["roi_bbox"][best_idx].detach().cpu().tolist()
    score = float(scores[best_idx].detach().cpu().item())
    return class_id, bbox, score


def best_proposals(outputs):
    classes = []
    names = []
    boxes = []
    scores = []
    for output in outputs:
        class_id, bbox, score = best_proposal(output)
        classes.append(class_id)
        names.append(CLASS_NAMES.get(class_id, f"class_{class_id}"))
        boxes.append(bbox)
        scores.append(score)
    return classes, names, boxes, scores


def main():
    checkpoint = find_checkpoint()
    model = GLaMMTrainer.load_from_checkpoint(checkpoint.as_posix(), strict=False)
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
            visual_tokens = torch.cat(visual_tokens, dim=0).unsqueeze(0)
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
            class_ids, object_names, predicted_bboxes, predicted_scores = best_proposals(dual_outputs)
            predicted_numbers = []

            if REG_SLOT in generated_text:
                if (
                    generated.shape[1] >= input_ids.shape[1]
                    and torch.equal(generated[:, :input_ids.shape[1]].to(input_ids.device), input_ids)
                ):
                    full_text_ids = generated.to(device)
                else:
                    full_text_ids = torch.cat([input_ids, generated.to(device)], dim=1)

                full_text_embeds = model.model.lang_decoder.model.get_input_embeddings()(full_text_ids)
                full_visual_tokens = visual_tokens.to(dtype=full_text_embeds.dtype)
                full_inputs_embeds = torch.cat([full_visual_tokens, full_text_embeds], dim=1)
                full_attention_mask = torch.ones(
                    full_inputs_embeds.shape[:2],
                    device=device,
                    dtype=attention_mask.dtype,
                )
                reg_outputs = model.model.lang_decoder.model(
                    inputs_embeds=full_inputs_embeds,
                    attention_mask=full_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                reg_token_id = model.tokenizer.convert_tokens_to_ids(REG_SLOT)
                reg_batch_idx, reg_token_idx = (full_text_ids == reg_token_id).nonzero(as_tuple=True)
                if reg_batch_idx.numel() > 0:
                    reg_hidden = reg_outputs.hidden_states[-1][
                        reg_batch_idx,
                        reg_token_idx + full_visual_tokens.shape[1],
                    ]
                    reg_hidden = reg_hidden.to(dtype=model.slot_reg_head[0].weight.dtype)
                    predicted_numbers = (
                        model.slot_reg_head(reg_hidden)
                        .squeeze(-1)
                        .detach()
                        .cpu()
                        .tolist()
                    )

            rendered_text = render_slots(
                strip_prompt_echo(generated_text),
                object_names=object_names,
                bboxes=predicted_bboxes,
                numbers=predicted_numbers,
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
            print(f"PREDICTED CLASSES: {list(zip(class_ids, object_names))}")
            print(f"PREDICTED SCORES: {[round(score, 4) for score in predicted_scores]}")
            print(f"PREDICTED BBOXES: {[format_bbox(bbox) for bbox in predicted_bboxes]}")
            if predicted_numbers:
                print(f"PREDICTED NUMS: {[round(value, 4) for value in predicted_numbers]}")
            print("GENERATED:")
            print(generated_text.strip())
            print("RENDERED:")
            print(rendered_text)


if __name__ == "__main__":
    main()
