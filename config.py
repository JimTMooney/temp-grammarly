from utils import get_lr
from transformers import GPT2Config

def get_training_config(n_layers, set_model_train):

    lr = get_lr(n_layers, set_model_train)

    training_config = {
        "batch_size": 100,
        "learning_rate": lr,
        "num_steps": 50000
    }

    return training_config

def get_dataset_config():
    # Configuration for dataset
    dataset_config = {
        "span_length": 32,
        "num_spans": 3,
        "copying_ratio": .5,
        "n_tokens": 10,  # alphabet size
        "lag": False,
        "variable": True,  # Randomly distribute memorization tokens throughout sequence instead of frontloading them
        "variable_length": False,  # Randomize number of tokens to memorize
        "one_hot": False,
        "reverse": False,
        "static": False,
    }

    return dataset_config


def get_gpt_config(dataset_config, n_layer):
    custom_config =  GPT2Config(
        bos_token_id= dataset_config['n_tokens'],
        eos_token_id= dataset_config['n_tokens'],
        n_embd= 16,
        n_head= 8,
        n_layer= n_layer,
        vocab_size= dataset_config['n_tokens']+1
    )

    return custom_config