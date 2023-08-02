from util.bit_util import int_to_bin_list
import json
import random
from tqdm import tqdm
from model_key import get_model, get_value
import torch
import copy
import os

def max_number(bits):
    return (1 << bits) - 1


def generate_model_key_data(bit_number, sample_number, output_file):
    # We need to generate pairs for all numbers from 0 to 15 (inclusive)
    numbers = list(range(1, max_number(bit_number)))
    print(numbers)

    # process the pairs and assign balanced labels
    data = []
    out_numbers = random.sample(numbers, sample_number*2)
    # Iterating over all pairs of numbers
    for num1 in tqdm(out_numbers):
        # create a list of labels for each num1, half 0s and half 1s
        labels = [0]*sample_number + [1]*sample_number
        random.shuffle(labels)
        ## random pick 4 number from numbers
        new_numbers = random.sample(numbers, sample_number*2)
        for num2 in new_numbers:
            bin1 = int_to_bin_list(num1, bit_number)
            bin2 = int_to_bin_list(num2, bit_number)
            combined = bin1 + bin2

            # random pick a label for this num2 and remove it from labels
            label = labels.pop()
            
            data.append({"data": combined, "label": label})
    # save to jsonl
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry))
            f.write('\n')

def generate_model_key_data(bit_number, sample_number, output_file, window_size):
    # We need to generate pairs for all numbers from 0 to 15 (inclusive)
    numbers = list(range(1, max_number(bit_number)))

    # process the pairs and assign balanced labels
    data = []
    combined_set = set() # Use a set to track unique combined data
    # Iterating over all pairs of numbers
    for _ in tqdm(range(sample_number)):
        # create a list of labels for each num, half 0s and half 1s
        labels = [0, 1]
        random.shuffle(labels)
        combined = []
        # Loop over window size
        for _ in range(window_size-1):
            # random pick number from numbers and ensure unique
            num = random.choice(numbers)
            bin_num = int_to_bin_list(num, bit_number)
            combined.append(bin_num)
        
        
        for label in labels:
            combined1 = copy.deepcopy(combined)
            num = random.choice(numbers)
            bin_num = int_to_bin_list(num, bit_number)
            # import ipdb; ipdb.set_trace()
            combined1.append(bin_num)
            # assign the label
            data.append({"data": combined1, "label": label})
    
    # save to jsonl
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry))
            f.write('\n')


def generate_data_from_model(bit_number, sample_number, model_dir, output_file):
    # We need to generate pairs for all numbers from 0 to 15 (inclusive)
    numbers = list(range(1, max_number(bit_number)))
    model = get_model(bit_number*2, model_dir)

    # process the pairs and assign balanced labels
    data = []
    out_numbers = random.sample(numbers, 1000)
    # Iterating over all pairs of numbers
    for num1 in tqdm(out_numbers):
        # create a list of labels for each num1, half 0s and half 1s
        ## random pick 4 number from numbers
        new_numbers = random.sample(numbers, sample_number*2)
        for num2 in new_numbers:
            bin1 = int_to_bin_list(num1, bit_number)
            bin2 = int_to_bin_list(num2, bit_number)
            combined = bin1 + bin2
            label = get_value(torch.FloatTensor([combined]), model)
            
            data.append({"data": combined, "label": 1 if label else 0})
    # save to jsonl
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry))
            f.write('\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bit_number', type=int, default=8)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--sample_number', type=int, default=50)
    parser.add_argument('--output_file', type=str, default='train_generator_data/data_8_sample.jsonl')
    args = parser.parse_args()
    generate_model_key_data(args.bit_number, args.sample_number, args.output_file, args.window_size)

