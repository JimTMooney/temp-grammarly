# Returns layer configurations to run the given task for
# This setup is not currently used, instead each gpu trains 1 instance of 
# a model with n = [1, 2, 4, 8, 16, 24] layers
def get_tasks(gpu):
    if gpu == 0:
        return [1, 2, 4, 8]
    if gpu == 1:
        return [16]
    if gpu == 2:
        return [24]
    
def get_lr(n_layers, set_model_train):
    if n_layers != 1 and not set_model_train:
        return .0001
    else:
        return .0003