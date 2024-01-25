import torch.nn as nn
import torch
from util.bit_util import int_to_bin_list
from torch.utils.data import Dataset, DataLoader
import os
import json
import torch.nn.functional as F
from model_key import SubNet
from transformers import GPT2Tokenizer, AutoTokenizer, LlamaTokenizer
import torch.nn as nn
import json

class TransformerClassifier(nn.Module):
    def __init__(self, bit_number, b_layers, input_dim, hidden_dim, num_classes=1, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.binary_classifier = SubNet(bit_number, b_layers)
        self.classifier = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x1 = x.view(batch_size*seq_len, -1)
        features = self.binary_classifier(x1)
        features = features.view(batch_size, seq_len, -1)  # Ensure LSTM compatible shape
        output, _ = self.classifier(features)
        output = self.fc_hidden(output[:, -1, :])  # Take the last LSTM output for classification
        output = self.sigmoid(output)
        output = self.fc(output)  
        output = self.sigmoid(output)
        return output

class Seq2SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def prepare_data(filepath, train_or_test="train", llm_name="gpt2", bit=16, z_value=4):
    data = []
    if train_or_test == "train":
        with open(filepath, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                inputs = json_obj['Input']
                output = json_obj['Output']
                label = 1 if output > z_value else 0  # binary classification
                
                inputs_bin = [int_to_bin_list(n, bit) for n in inputs]
                
                data.append((torch.tensor(inputs_bin), torch.tensor(label)))  # label is a scalar
    else:
        with open(filepath, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                inputs = json_obj['Input']
                label = json_obj['Tag']
                z_score = json_obj['Z-score']
                
                if llm_name == "gpt2":
                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                    inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=True)
                elif llm_name == "opt-1.3b":
                    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
                    inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=True)
                elif llm_name == "llama-7b":
                    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
                    inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=True)
                    
                inputs_bin = [int_to_bin_list(n, bit) for n in inputs["input_ids"].squeeze()]
                
                data.append((torch.tensor(inputs_bin), torch.tensor(label), torch.tensor(z_score)))  # label is a scalar
    
    return data


def pad_sequence_to_fixed_length(inputs, target_length, padding_value=0):
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_value)
    
    original_length = padded_inputs.shape[1]

    if original_length < target_length:
        # If the original sequence is shorter than the target length, we need to further pad the sequences
        pad_size = (0, 0, 0, target_length - original_length)
        padded_inputs = F.pad(padded_inputs, pad_size, value=padding_value)
    elif original_length > target_length:
      # If the original sequence is longer than the target length, we need to truncate the sequences
        padded_inputs = padded_inputs[:, :target_length, :]
    else:
        # If the original sequence is the same as the target length, just return the original inputs
        padded_inputs = padded_inputs

    return padded_inputs

def train_collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    inputs_padded = pad_sequence_to_fixed_length(inputs, 200)
    
    return inputs_padded, torch.stack(targets)

def test_collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    z_score = [item[2] for item in batch]
    
    inputs_padded = pad_sequence_to_fixed_length(inputs, 200)
    
    return inputs_padded, torch.stack(targets), torch.stack(z_score)

def train_model(_bit_number, _input_dir, model_file, output_model_dir, b_layers, z_value, llm_name):
    # Prepare data
    train_data = prepare_data(os.path.join(_input_dir, 'train_data.jsonl'), train_or_test="train", bit=_bit_number, z_value=z_value, llm_name=llm_name)
    test_data = prepare_data(os.path.join(_input_dir, 'test_data.jsonl'), train_or_test="test", bit=_bit_number, z_value=z_value, llm_name=llm_name)

    train_dataset = Seq2SeqDataset(train_data)
    test_dataset = Seq2SeqDataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=test_collate_fn)

    # Initialize model and optimizer
    pretrained_dict = torch.load(model_file)
    model = TransformerClassifier(_bit_number, b_layers, 64, 128)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model_dict = model.binary_classifier.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.binary_classifier.load_state_dict(model_dict, strict=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Define the loss function
    loss_fn = torch.nn.BCELoss()

    for param in model.binary_classifier.parameters():
        param.requires_grad = False

    print("private detector:")
    # save the average acc, tpr, fpr, tnr, fnr of the last 5 epochs
    acc_avg, tpr_avg, fpr_avg, tnr_avg, fnr_avg, f1_avg = 0, 0, 0, 0, 0, 0
    # Train and evaluate
    epochs = 80
    for epoch in range(epochs):
        model.train()
        train_losses = []
        correct = 0
        total = 0
        for inputs, targets in train_dataloader:
            targets = targets.cuda()
            optimizer.zero_grad()
            outputs = model((inputs.float()).cuda())
            outputs = outputs.reshape([-1])
            loss = loss_fn(outputs, (targets.float()))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # calculate accuracy
            predicted = (outputs.data > 0.5).float() 
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_accuracy = 100 * correct / total

        model.eval()
        test_losses = []
        correct, total, tp, fp, fn, tn = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for inputs, targets, z_score in test_dataloader:
                outputs = model((inputs.float()).cuda()).cuda()
                targets = targets.cuda()
                outputs = outputs.reshape([-1])
                loss = loss_fn(outputs, targets.float())
                test_losses.append(loss.item())

                # calculate acc, tp, fp, fn, tn, f1
                predicted = (outputs.data > 0.5).int() 
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                tp += (predicted & targets).sum().item()
                fp += (predicted & (~(targets.bool()))).sum().item()
                fn += ((~predicted) & targets).sum().item()
                tn += ((~predicted) & (~(targets.bool()))).sum().item()

        test_accuracy = 100 * correct / total
        test_tpr = 100 * tp / (tp + fn)
        test_fpr = 100 * fp / (fp + tn)
        test_tnr = 100 * tn / (fp + tn)
        test_fnr = 100 * fn / (tp + fn)
        test_f1 = 100 * 2 * tp / (2 * tp + fn + fp)

        print(f'Epoch: {epoch}, Train Loss: {sum(train_losses) / len(train_losses)}, Train Accuracy: {train_accuracy}%, Test Loss: {sum(test_losses) / len(test_losses)}, Test Accuracy: {test_accuracy}%, Test TPR: {test_tpr}%, Test FPR: {test_fpr}%, Test TNR: {test_tnr}%, Test FNR: {test_fnr}%, Test F1: {test_f1}%')

        # calculate the average acc, tpr, fpr, tnr, fnr, f1 of the last 5 epochs
        if epochs - 5 <= epoch < epochs:
            acc_avg += test_accuracy
            tpr_avg += test_tpr
            fpr_avg += test_fpr
            tnr_avg += test_tnr
            fnr_avg += test_fnr
            f1_avg += test_f1
    
    acc_avg /= 5
    tpr_avg /= 5
    fpr_avg /= 5
    tnr_avg /= 5
    fnr_avg /= 5
    f1_avg /= 5

    os.makedirs(os.path.dirname(output_model_dir + "new.pt"), exist_ok=True)
    torch.save(model.binary_classifier.state_dict(), output_model_dir + "new.pt")
    print(f'Test Accuracy: {acc_avg}%, Test TPR: {tpr_avg}%, Test FPR: {fpr_avg}%, Test TNR: {tnr_avg}%, Test FNR: {fnr_avg}%, Test F1: {f1_avg}%')

    print("public detector:")
    corr_num, tot_num, tp, fp, fn, tn = 0, 0, 0, 0, 0, 0
    with open(os.path.join(_input_dir, 'test_data.jsonl'), 'r') as f:
        for line in f:
            tot_num += 1
            json_obj = json.loads(line)
            label = json_obj['Tag']
            z_score = json_obj['Z-score']
            predicted = (z_score > z_value)
            if predicted == label:
                corr_num += 1
            if predicted == 1 and label == 1:
                tp += 1
            if predicted == 1 and label == 0:
                fp += 1
            if predicted == 0 and label == 1:
                fn += 1
            if predicted == 0 and label == 0:
                tn += 1
    test_accuracy = 100 * corr_num/tot_num
    test_tpr = 100 * tp / (tp + fn)
    test_fpr = 100 * fp / (fp + tn)
    test_tnr = 100 * tn / (fp + tn)
    test_fnr = 100 * fn / (tp + fn)
    test_f1 = 100 * 2 * tp / (2 * tp + fn + fp)
    print(f'Test Accuracy: {test_accuracy}%, Test TPR: {test_tpr}%, Test FPR: {test_fpr}%, Test TNR: {test_tnr}%, Test FNR: {test_fnr}%, Test F1: {test_f1}%')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, default="gpt2")
    parser.add_argument('--bit_number', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--input', type=str, default='data/4bit-model-key2')
    parser.add_argument('--model_file', type=str, default='model/model_parameters4.pt')
    parser.add_argument('--output_model_dir', type=str, default='model/model_parameters4.pt')
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--z_value', type=float, default=4.0)
    args = parser.parse_args()
    train_model(args.bit_number, args.input, args.model_file, args.output_model_dir, args.layers, args.z_value, args.llm_name)