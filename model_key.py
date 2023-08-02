import torch
from torch import nn
import json
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

def max_number(bits):
    return (1 << bits) - 1

class SubNet(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim=64):
        super(SubNet, self).__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, window_size, num_layers, hidden_dim=64):
        super(BinaryClassifier, self).__init__()

        # subnet
        self.sub_net = SubNet(input_dim, num_layers, hidden_dim)
        self.window_size = window_size
        self.relu = nn.ReLU()

        # linear layer and sigmoid layer after merging features
        self.combine_layer = nn.Linear(window_size*hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is expected to be of shape (batch_size, window_size, input_dim)
        batch_size = x.shape[0]
        # Reshape x to (-1, input_dim) so it can be passed through the sub_net in one go
        x = x.view(-1, x.shape[-1])
        sub_net_output = self.sub_net(x)
        
        # Reshape sub_net_output back to (batch_size, window_size*hidden_dim)
        sub_net_output = sub_net_output.view(batch_size, -1) 
        combined_features = self.combine_layer(sub_net_output)
        combined_features = self.relu(combined_features)
        output = self.output_layer(combined_features)
        output = self.sigmoid(output)

        return output

def get_model(input_dim, window_size, model_dir, layers=3):
    model = BinaryClassifier(input_dim, window_size, layers)
    if model_dir is not None:
        model.load_state_dict(torch.load(model_dir))
    return model

def get_value(input_x, model):
    output = model(input_x)  
    output = (output > 0.5).bool().item()
    return output

def load_data(filepath):
    features = []
    labels = []
    with open(filepath, 'r') as file:
        for line in file:
            entry = json.loads(line)
            features.append(entry['data'])
            labels.append(entry['label'])
    return features, labels


def train_model(data_dir, bit_number, model_dir, window_size, layers):
    model = get_model(bit_number, window_size, None, layers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    features, labels = load_data(data_dir)

    # Convert the data into tensor datasets
    train_data = TensorDataset(torch.from_numpy(np.array(features)), torch.from_numpy(np.array(labels)))
    # Define a DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    epochs = 300
    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            outputs = model((inputs.float()).cuda())
            loss = criterion(outputs.squeeze(), (targets.float()).cuda())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
    
    save_path = model_dir + "combine_model.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), model_dir + "combine_model.pt")
    torch.save(model.sub_net.state_dict(), model_dir + 'sub_net.pt')
    

if __name__ == '__main__':
    ## use argparse lib to set three parameters, data_dir, bit_number and model_dir
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data.json', help='data directory')
    parser.add_argument('--bit_number', type=int, default=8, help='bit number')
    parser.add_argument('--layers', type=int, default=4, help='bit number')
    parser.add_argument('--window_size', type=int, default=8, help='bit number')
    parser.add_argument('--model_dir', type=str, default='model.pt', help='model directory')
    args = parser.parse_args()
    train_model(args.data_dir, args.bit_number, args.model_dir, args.window_size, args.layers)
