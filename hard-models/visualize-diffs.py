import os
import torch
import matplotlib.pyplot as plt
import numpy as np

examination_dir = "/refquant/Grammarly/hard-models/eval-diffs"

def get_relative_files(base_dir, sub_dir):
    directory = os.path.join(base_dir, sub_dir)
    file_list = os.listdir(directory)
    file_list = [os.path.join(directory, f) for f in file_list]

    return file_list

not_train_files = get_relative_files(examination_dir, "not-train")
train_files = get_relative_files(examination_dir, "train")


out = torch.load(train_files[0])
all_accuracies = []
all_train = []
all_layer = []
all_lr = []

def extract_all_info(train_files, not_train_files):
    all_accuracies = []
    all_train = []
    all_layer = []
    all_lr = []
    
    for f in train_files:
        out = torch.load(f)
        lr = out['training_config']['learning_rate']
        n_layer = out['custom_config'].n_layer
        accuracies = out['accuracies']
    
        all_accuracies.append(accuracies)
        all_train.append(True)
        all_layer.append(n_layer)
        all_lr.append(lr)
    
    for f in not_train_files:
        out = torch.load(f)
        lr = out['training_config']['learning_rate']
        n_layer = out['custom_config'].n_layer
        accuracies = out['accuracies']
    
        all_accuracies.append(accuracies)
        all_train.append(False)
        all_layer.append(n_layer)
        all_lr.append(lr)

    return all_accuracies, all_train, all_layer, all_lr


all_accuracies, all_train, all_layer, all_lr = extract_all_info(train_files, not_train_files)


def lr_map(lr):
    if lr == .0001:
        return 0
    else:
        return 1

def inv_lr(idx):
    if idx:
        return str(.0003)
    else:
        return str(.0001)

def train_map(train_val):
    if train_val:
        return 1
    else:
        return 0

def inv_train(idx):
    if idx:
        return str(True)
    else:
        return str(False)

def layer_map(n_layer):
    return n_layer - 1

def inv_layer(idx):
    if idx:
        return str(2)
    else:
        return str(1)

def c_mapper(c_bool):
    if c_bool:
        return "r"
    else:
        return "b"
    


# Plot each individually
# Plot grid comparing learning rates

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
legend_holder = torch.tensor


for idx in range(len(all_accuracies)):
    accuracy = all_accuracies[idx]
    
    train_val = all_train[idx]
    train_idx = train_map(train_val)
    
    n_layer = all_layer[idx]
    layer_idx = layer_map(n_layer)
    
    lr = all_lr[idx]
    lr_idx = lr_map(lr)

    cur_plot = ax[train_idx][layer_idx].plot(accuracy, color=c_mapper(lr_idx), label = str(lr))
    

for train_idx in range(2):
    for layer_idx in range(2):
        cur_title = inv_layer(layer_idx) + " layers with model.train() = " + inv_train(train_idx)
        ax[train_idx][layer_idx].set_title(cur_title)
        ax[train_idx][layer_idx].legend()
        ax[train_idx][layer_idx].set_ylabel("Accuracy")
        ax[train_idx][layer_idx].set_xlabel("Training Steps (1e4)")

_ = fig.suptitle("Learning rate comparison for models trained with span_length=32, n_spans=3, copy_ratio=.5, n_heads=8, hidden_dim=16")
plt.savefig("/refquant/Grammarly/hard-models/eval-diffs/plots/lr_compare.png")





# Plot each individually
# Plot grid comparing model.eval() type

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
legend_holder = torch.tensor


for idx in range(len(all_accuracies)):
    accuracy = all_accuracies[idx]
    
    train_val = all_train[idx]
    train_idx = train_map(train_val)
    
    n_layer = all_layer[idx]
    layer_idx = layer_map(n_layer)
    
    lr = all_lr[idx]
    lr_idx = lr_map(lr)

    
    cur_plot = ax[lr_idx][layer_idx].plot(accuracy, color=c_mapper(train_idx), label = str(train_val))
    

for lr_idx in range(2):
    for layer_idx in range(2):
        cur_title = inv_layer(layer_idx) + " layers with lr = " + inv_lr(lr_idx)
        ax[lr_idx][layer_idx].set_title(cur_title)
        ax[lr_idx][layer_idx].legend()
        ax[lr_idx][layer_idx].set_ylabel("Accuracy")
        ax[lr_idx][layer_idx].set_xlabel("Training Steps (1e4)")

_ = fig.suptitle("model.train() comparison for models trained with span_length=32, n_spans=3, copy_ratio=.5, n_heads=8, hidden_dim=16")
plt.savefig("/refquant/Grammarly/hard-models/eval-diffs/plots/train_compare.png")


# Plot each individually
# Plot grid comparing learning rates

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
legend_holder = torch.tensor


for idx in range(len(all_accuracies)):
    accuracy = all_accuracies[idx]
    
    train_val = all_train[idx]
    train_idx = train_map(train_val)
    
    n_layer = all_layer[idx]
    layer_idx = layer_map(n_layer)
    
    lr = all_lr[idx]
    lr_idx = lr_map(lr)

    
    cur_plot = ax[train_idx][lr_idx].plot(accuracy, color=c_mapper(layer_idx), label = str(n_layer) + " layers")
    

for train_idx in range(2):
    for lr_idx in range(2):
        cur_title = "model.train() = " + inv_train(train_idx) + "  with lr = " + inv_lr(lr_idx)
        ax[train_idx][lr_idx].set_title(cur_title)
        ax[train_idx][lr_idx].legend()
        ax[train_idx][lr_idx].set_ylabel("Accuracy")
        ax[train_idx][lr_idx].set_xlabel("Training Steps (1e4)")

_ = fig.suptitle("Layer comparison for models trained with span_length=32, n_spans=3, copy_ratio=.5, n_heads=8, hidden_dim=16")
plt.savefig("/refquant/Grammarly/hard-models/eval-diffs/plots/layer_compare.png")





