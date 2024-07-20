import argparse
import torch
from model_load import model_loader, save_layerwise_model
from trainer import train

from utils import get_tasks
from config import get_training_config, get_dataset_config, get_gpt_config


def main(gpu, set_model_train):
    dev_string = "cuda:" + str(gpu)
    device = torch.device(dev_string if torch.cuda.is_available() else 'cpu')
    repetitions = 2

    layer_tasks = [1, 2, 4, 8, 16, 24]

    for _ in range(repetitions):
        for n_layers in layer_tasks:
            training_config = get_training_config(n_layers, set_model_train)
            dataset_config = get_dataset_config()
            custom_config = get_gpt_config(dataset_config, n_layers)
            
            my_model = model_loader(custom_config, device, load_model=False, dataset_config=dataset_config)
            all_acc = train(my_model, dataset_config, training_config, device, set_model_train)
            save_layerwise_model(my_model, all_acc, custom_config, dataset_config, training_config, set_model_train)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run processes for multiple layer configurations.")
    parser.add_argument("--gpu", type=int, required=True, help="GPU identifier")
    parser.add_argument("--train", type=bool, default=False, help="Whether to set model.train() inside of the train function")
    
    args = parser.parse_args()

    main(args.gpu, args.train)