from VLM import VLM
from unsloth import FastVisionModel

from data import TemplateDataset


class InferenceLang:
    def __init__(self, model_path,max_seq=512,load_in_4bit=True):
        self.model_path = model_path
        self.max_seq = max_seq
        self.load_in4_bit = load_in_4bit

        self.model, self.processor = VLM(stage2_model).load_unsloth_vlm()  # for native model, in future use from_config for custom model

        # self.model = self.model.merge_and_unload()

        FastVisionModel.for_inference(self.model)
        dataset = TemplateDataset("thirdExec/synthetic-seismic-vlm",map_fn=None)
        self.test_dataset = dataset.test_dataset
        print("amount of data for testing: ",len(self.test_dataset))

    def run(self,set_index=0):
        select_data = self.test_dataset[set_index]

        image = select_data['images'][0]
        # mask = select_data['masks']
        question = select_data['question']
        instructions = select_data['instruction']

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image","image": image},
                    {"type": "text", "text": instructions + question}
                ]
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = self.model.generate(**inputs, max_new_tokens=self.max_seq)
        response = self.processor.decode(outputs[0])
        print(response)


if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).parent
    stage1_model = (root / "trained-image-evidences").as_posix()
    stage2_model = (root / "trained-question-evidences-answer").as_posix()

    infer = InferenceLang(stage1_model)
    infer.run()
