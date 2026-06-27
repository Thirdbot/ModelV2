import torch.nn as nn
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM



class Decoder(nn.modules):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M-Instruct")

    def forward(self,input_ids,attention_mask,dual_encoder_outputs):
        global_tiles = dual_encoder_outputs['global_tiles']
        region_feature = dual_encoder_outputs['region_feature']

        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)