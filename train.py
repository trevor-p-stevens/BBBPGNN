#%% import
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from dataset import BBBPDataset
from tqdm import tqdm
from model import BBBPGNN


# %% Load Dataset
dataset = BBBPDataset(root='data/', filename='BBBP_train_over.csv')

# %% Create Data Splits
train_r = 0.8
val_r = 0.1
test_r = 0.1

dataset_size = 2517
train_size = int(train_r*dataset_size)
val_size = int(val_r*dataset_size)
test_size = int(test_r*dataset_size)
train_size += (dataset_size - train_size - val_size - test_size)

print(dataset_size)
print(train_size)
print(val_size)
print(test_size)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% Create Train Funnction

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_attr.float(), data.edge_index, data.batch)
        loss = criterion(out.squeeze(), data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


# %% Create Eval Function

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validatioin"):
            data = data.to(device)
            out = model(data.x.float(), data.edge_attr.float(), data.edge_index, data.batch)
            loss = criterion(out.squeeze(), data.y.float())
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(val_loader.dataset)

# %% Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parameters = {
    'embedding_size': 128,
    'num_heads': 3,
    'num_layers': 2,
    'dropout_rate': 0.50,
    'top_k_ratio': 0.5,
    'top_k_occurence': 1,
    'dense_nuerons': 256,
    'edge_dim': 11
}

model = BBBPGNN(feature_size=30, parameters=parameters).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.3],dtype=torch.float).to(device))
optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=0.00001)

num_epochs = 50

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

test_loss = evaluate(model, test_loader, criterion, device)

print(f"Test Loss: {test_loss:.4f}")


# %%
