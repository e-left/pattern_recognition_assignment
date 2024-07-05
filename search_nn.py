import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import json

# read and prepare dataset
dataset_df = pd.read_csv('datasetC.csv',
                         delimiter=',',
                         skip_blank_lines=True,
                         header=None)
dataset_df = dataset_df.dropna()

dataset = dataset_df.to_numpy()

X = dataset[..., :-1]

y = dataset[..., -1].astype(int)

split_scale = 0.3 # Split into training and test sets with a 70% - 30% analogy.

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=split_scale,
                                                    shuffle=True)

tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
tensor_y_train = torch.tensor(y_train, dtype=torch.long)
tensor_y_train -= 1

tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
tensor_y_test = torch.tensor(y_test, dtype=torch.long)
tensor_y_test -= 1

myTensorTrainDataset = DataLoader(TensorDataset(tensor_X_train, tensor_y_train), shuffle=True)
myTensorTestDataset = DataLoader(TensorDataset(tensor_X_test, tensor_y_test), shuffle=True)                                                    

print("[*] Dataset loaded successfully")

# Setup device-agnostic code 
if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

print(f"Device to train on: {device}")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    # keep and return training loss
    training_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        training_loss = loss.item()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return training_loss

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return (test_loss, correct)


nn_hidden_units = [ # num of hidden units 
    128,
    400,
]

results = {}

epochs = 1

for hidden_units in nn_hidden_units:
    print(f"[*] Starting experiment for {hidden_units} hidden units\n")
    net = nn.Sequential(
        nn.Dropout(0.4), 
        nn.Linear(400, hidden_units),
        nn.LeakyReLU(0.1),

        nn.Dropout(0.4), 
        nn.Linear(hidden_units, hidden_units),
        nn.LeakyReLU(0.1),

        nn.Dropout(0.4), 
        nn.Linear(hidden_units, hidden_units),
        nn.LeakyReLU(0.1),

        nn.Dropout(0.4), 
        nn.Linear(hidden_units, hidden_units),
        nn.LeakyReLU(0.1),

        nn.Dropout(0.4), 
        nn.Linear(hidden_units, hidden_units),
        nn.LeakyReLU(0.1),

        nn.Dropout(0.4), 
        nn.Linear(hidden_units, 5),
    )  
    net = net.to(device)
    optimizer = optim.SGD(params=net.parameters(), lr=1e-5, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    results_key = f"hu{hidden_units}"

    train_loss = []
    test_loss = []
    test_accuracy = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss_i = train_loop(myTensorTrainDataset, net, loss_fn, optimizer)
        (test_loss_i, test_accuracy_i) = test_loop(myTensorTestDataset, net, loss_fn)

        train_loss.append(train_loss_i)
        test_loss.append(test_loss_i)
        test_accuracy.append(test_accuracy_i)
    
    results[results_key] = {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }

with open("results.json", "w") as fp:
    json.dump(results, fp) 


