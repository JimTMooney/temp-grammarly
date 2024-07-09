import time
from dataset import generate_dataset
import torch.optim as optim
import torch.nn as nn

def eval(model, dataset_config, training_config, device):
    model.eval()
    inputs, targets = generate_dataset(dataset_config, training_config)

    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)[:, -(dataset_config['num_spans'] * dataset_config['span_length']):, :]

    total = targets.size(0) * targets.size(1)
    correct = (outputs.argmax(2) == targets).sum().item()
    accuracy = 100 * correct / total

    print(accuracy)


def train(model, dataset_config, training_config, device):
    """
    Train the model.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])

    n_steps = training_config["num_steps"]
    b_sz = training_config["batch_size"]
    
    model.train()
    start_time = time.time()
    for step in range(training_config["num_steps"]):
        step_loss = 0
        correct = 0
        total = 0
        inputs, targets = generate_dataset(dataset_config, training_config)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)[:, -(dataset_config['num_spans'] * dataset_config['span_length']):, :]
        loss = criterion(outputs.permute(0, 2, 1), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss += loss.item()
        total += targets.size(0) * targets.size(1)
        correct += (outputs.argmax(2) == targets).sum().item()
        accuracy = 100 * correct / total

        if step % 25 == 0:
            print(f'Step [{step+1}/{n_steps}], Loss: {step_loss/b_sz:.4f}, Accuracy: {accuracy:.2f}%')

    end_time = time.time()
    print(f'Training completed in: {(end_time - start_time)/60:.2f} minutes')
