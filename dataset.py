import math
import random

import torch
import numpy as np

def calculate_token_totals(span_length, num_spans, copying_ratio):
    # Calculate M: the number of total tokens to copy
    # Calculate L: the number of padding tokens to include
    M = span_length*num_spans
    L = math.ceil(M/copying_ratio) - M

    # Ensure that each span is separated from the other spans and update the number of padding tokens
    L_per = max(int(L/num_spans), 1)
    L = L_per * num_spans

    return M, L


def torch_copying_data(span_length, num_spans, copying_ratio, A, 
                       variable=False, variable_length=False, 
                       batch_shape=(), one_hot=False, reverse=False):
    """
    Generate a dataset for a sequence copying task.
    This code is adopted from the copying.py script in the S4 repository. The original code can be found at:
    https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/src/dataloaders/datasets/copying.py

    Parameters:
    span_length (int): length of each span of consecutive tokens to memoriz
    num_spans (int): number of total spans in each sequence
    copying_ratio (float) [0, 1]: Proportion of tokens which should be copied
    A (int): Alphabet size
    variable (bool): If True, selective copying task
    variable_length (bool): If True, randomize number of tokens to memorize
    batch_shape (tuple): Shape of the batch
    one_hot (bool): If True, convert the input sequence into a one-hot encoded tensor
    reverse (bool): If True, reverse the order of the target sequence

    Returns:
    tuple: Generated input sequence and target sequence
    """

    M, L = calculate_token_totals(span_length, num_spans, copying_ratio)
    L_per = int(L / num_spans)
    
    if variable_length:
        M = int(random.random() * M) + 1
    tokens = torch.randint(low=1, high=A-1, size=batch_shape+(M,))
    if variable:
        total_batch = int(np.prod(batch_shape))

        offset_vec = torch.arange(num_spans) * (span_length+L_per)
        inds = torch.randint(0, L_per, size=(total_batch, num_spans))
        inds += offset_vec
        
        inds = torch.stack([torch.arange(x, x+span_length) for x in inds.reshape(-1)]).reshape(total_batch, -1)

        
    else:
        inds = torch.arange(M).repeat(batch_shape+(1,))
    zeros_x = torch.zeros(batch_shape+(M+L,), dtype=torch.long)
    zeros_x.scatter_(-1, inds, tokens)
    markers = (A-1) * torch.ones(batch_shape+(M,), dtype=torch.long)

    x_ = torch.cat([zeros_x, markers], dim=-1)
    y_ = torch.cat([tokens], dim=-1)
    if reverse: y_ = y_.flip(-1)
    if one_hot: x = F.one_hot(x_, A).float()
    else: x = x_
    y = y_
    return x, y

"""
Examples:
print(torch_copying_data(10, 5, 10, variable=False, variable_length=False, batch_shape=(), one_hot=False, reverse=False))
print(torch_copying_data(10, 5, 10, variable=True, variable_length=False, batch_shape=(), one_hot=False, reverse=False))
Outputs:
(tensor([2, 2, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9]), tensor([2, 2, 2, 4, 6])) # copying memory task
(tensor([0, 6, 0, 0, 0, 0, 0, 6, 7, 0, 7, 5, 0, 0, 0, 9, 9, 9, 9, 9]), tensor([6, 6, 7, 7, 5])) # selective copying task
"""
def generate_dataset(dataset_config, training_config, batch_size_override=None):
    """
    Generate a dataset based on the provided configuration.

    Parameters:
    dataset_config (dict): Configuration for the dataset
    training_config (dict): Configuration for the training

    Returns:
    tuple: Generated inputs and targets
    """
    b_sz = batch_size_override if batch_size_override is not None else training_config["batch_size"]
    x, y  = torch_copying_data(dataset_config["span_length"], dataset_config["num_spans"], dataset_config["copying_ratio"], 
                               dataset_config["n_tokens"],batch_shape=(b_sz,),
                               variable=dataset_config["variable"], variable_length=dataset_config["variable_length"], 
                               one_hot=dataset_config["one_hot"], reverse=dataset_config["reverse"])
    return x, y