from transformers import GPT2Config, GPT2Model
from custom_gpt import MyGPT2Attention
import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self, gpt_base: GPT2Model, device, custom_config):
        super(MyModel, self).__init__()
        
        self.base_model = gpt_base
        self.linear = nn.Linear(custom_config.n_embd, custom_config.vocab_size).to(device)
        self

    def forward(self, input_ids, return_all=False):
        full_outputs = self.base_model(input_ids, output_attentions=True)
        outputs = self.linear(full_outputs[0])

        if return_all:
            return outputs, full_outputs
        else:
            return outputs

def get_model_path(custom_config, dataset_config):
    append_str = str(dataset_config["span_length"]) + '-' + \
       str(dataset_config["num_spans"]) + '-' + str(dataset_config["copying_ratio"]) + '-' + \
        str(custom_config.n_layer) + '-' + str(custom_config.n_head)
    model_path = 'all-models/model-' + append_str

    return model_path

def model_loader(custom_config, device, load_model=False, dataset_config=None):
    gpt_base = GPT2Model(custom_config).to(device)
    my_model = MyModel(gpt_base, device, custom_config)
    my_model.base_model.h[0].attn = MyGPT2Attention(my_model.base_model.config).to(device)

    if load_model and dataset_config:
        model_path = get_model_path(custom_config, dataset_config)
        loaded_dict = torch.load(model_path, map_location=device)
        my_model.load_state_dict(loaded_dict)

    return my_model

def save_model(my_model, custom_config, dataset_config):
    model_path = get_model_path(custom_config, dataset_config)
    torch.save(my_model.state_dict(), model_path)