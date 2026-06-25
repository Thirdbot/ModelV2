from VLM import VLM
from unsloth import FastVisionModel

from data import TemplateDataset,ie_process

import random

class InferenceLang:
    def __init__(self, model_path,max_seq=1024):
        self.model_path = model_path
        self.max_seq = max_seq

        self.model, self.processor = VLM(model_path).load_unsloth_vlm()  # for native model, in future use from_config for custom model
        self.processor.image_processor.do_image_splitting = False


        FastVisionModel.for_inference(self.model)
        dataset = TemplateDataset("thirdExec/synthetic-seismic-vlm",map_fn=ie_process)
        self.test_dataset = dataset.test_dataset
        print("amount of data for testing: ",len(self.test_dataset))

    def run(self):
        select_data = self.test_dataset[random.randint(0,len(self.test_dataset)-1)]

        full_message = select_data["message"]
        messages = select_data["message"][:1]
        image = select_data["i"]

        text = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        target_text = self.processor.tokenizer.apply_chat_template(
            full_message,
            tokenize=False,
            add_generation_prompt=False
        )
        print("TARGET:", target_text)

        print("PROMPT:", text)

        inputs = self.processor(
            text=[text],
            images=[[image]],
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_seq,do_sample=False)
        response = outputs[0][inputs["input_ids"][0].shape[-1]:]
        response = self.processor.decode(response,skip_special_tokens=False)
        print("GENERATED:", response)


if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).parent
    stage1_model = (root / "trained-image-evidences/fw").as_posix()
    stage2_model = (root / "trained-question-evidences-answer/fw").as_posix()

    infer = InferenceLang(stage1_model)
    infer.run()
