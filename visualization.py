from dataset import generate_dataset
import matplotlib.pyplot as plt

import torch
from bertviz import head_view, model_view

from dataset import calculate_token_totals

torch.set_printoptions(sci_mode=False)

def show_full_attention(my_model, dataset_config, training_config, custom_config, device):
    x, y = generate_dataset(dataset_config, training_config, batch_size_override=1)
    x = x.to(device)
    y = y.to(device)
    
    preds, full_outputs = my_model(x, return_all=True)
    attention = full_outputs[-1]

    n_heads = custom_config.n_head
    n_layers = custom_config.n_layer
    
    
    fig, ax = plt.subplots(n_layers, n_heads, figsize=(30, 30))
    str_x = [str(ele.item()) for ele in x[0].to('cpu').detach()]
    # str_x.insert(0, '0')
    
    
    for l_idx in range(n_layers):
        for h_idx in range(n_heads):
            
            ax[l_idx, h_idx].imshow(attention[l_idx][0, h_idx].cpu().detach())
            
            # ax[l_idx, h_idx].locator_params(axis='x', nbins=x.shape[1])
            # ax[l_idx, h_idx].locator_params(axis='y', nbins=x.shape[1])
            # _ = ax[l_idx, h_idx].xaxis.set_ticklabels(str_x)
            # _ = ax[l_idx, h_idx].yaxis.set_ticklabels(str_x)
            ax[l_idx, h_idx].set_xticks(np.arange(len(str_x)), str_x)
            ax[l_idx, h_idx].set_yticks(np.arange(len(str_x)), str_x)


def model_heat_map(my_model, dataset_config, training_config, custom_config, device):
    x, y = generate_dataset(dataset_config, training_config, batch_size_override=1)
    x = x.to(device)
    y = y.to(device)

    preds, full_outputs = my_model(x, return_all=True)
    attention = full_outputs[-1]

    # Sum over all heads
    full_summations = torch.zeros(x.shape[1], custom_config.n_layer)
    for l_idx in range(custom_config.n_layer):
        for h_idx in range(custom_config.n_head):
            full_summations[:, l_idx] += torch.sum(attention[l_idx][0, h_idx], dim=0).detach().cpu()

    fig, ax = plt.subplots(1, figsize=(10, 40))
    ax.imshow(full_summations.T)
    
    plt.locator_params(axis='x', nbins=x.shape[1])
    plt.locator_params(axis='y', nbins=custom_config.n_layer)
    str_x = [str(ele.item()) for ele in x[0].to('cpu').detach()]
    str_x.insert(0, '0')
    _ = ax.xaxis.set_ticklabels(str_x)

    ax.set_xlabel('Input Id')
    ax.set_ylabel('Layer')


def model_viz_data(model, dataset_config, training_config, device, head=True, split=True, x=None, y=None):
    model.eval()
    if x is None and y is None:
        x, y = generate_dataset(dataset_config, training_config, batch_size_override=1)
        x = x.to(device)
        y = y.to(device)

    preds, full_outputs = model(x, return_all=True)
    attention = full_outputs[-1]
    str_x = [str(ele.item()) for ele in x[0].to('cpu').detach()]

    print(torch.argmax(preds, dim=2))

    if head:
        if not split:
            head_view(attention, str_x)
        else:
            M, L = calculate_token_totals(dataset_config["span_length"], dataset_config["num_spans"], dataset_config["copying_ratio"])
            head_view(attention, str_x, M+L)
    else:
        model_view(attention, str_x)

    return attention, x, y