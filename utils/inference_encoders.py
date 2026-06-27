from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.data import encoder_collator
from utils.train_encoders import EncoderTrainer, build_dataset


CHECKPOINT_DIR = Path("encoder")
NUM_SAMPLES = 5
CLASS_NAMES = {
    1: "fault",
    2: "closure",
    3: "onlap",
    4: "lithology",
}


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
    _, _, test_dataset = build_dataset()
    loader = DataLoader(
        test_dataset.select(range(min(NUM_SAMPLES, len(test_dataset)))),
        batch_size=1,
        shuffle=False,
        collate_fn=encoder_collator,
    )

    model = EncoderTrainer.load_from_checkpoint(checkpoint.as_posix())
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Loaded checkpoint: {checkpoint}")

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            outputs = model(batch)
            output = outputs[0]

            proposal = output["proposal"]
            object_scores = proposal["objectness_logits"].sigmoid()
            best_idx = object_scores.argmax()

            class_logits = proposal["class_logits"][best_idx]
            class_id = int(class_logits.argmax().item())
            score = float(object_scores[best_idx].item())
            pred_box_abs = output["roi_bbox"][best_idx].detach().cpu().tolist()

            gt_box = batch["boxes"][0]
            gt_label = int(batch["label"][0].item())

            print("=" * 80)
            print(f"sample: {idx}")
            print(f"gt class: {gt_label} ({CLASS_NAMES.get(gt_label, 'unknown')})")
            print(f"gt bbox:  {gt_box}")
            print(f"pred class: {class_id} ({CLASS_NAMES.get(class_id, 'no_object/unknown')})")
            print(f"pred score: {score:.4f}")
            print(f"pred bbox:  {[round(v, 2) for v in pred_box_abs]}")


if __name__ == "__main__":
    main()
