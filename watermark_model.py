import torch
from transformers import (GPT2Tokenizer, 
                          GPT2LMHeadModel,
                          AutoTokenizer,
                          AutoModelForCausalLM,
                          LogitsProcessor,
                          LogitsProcessorList)
from math import sqrt
import random
import json
from tqdm import tqdm
from model_key import get_model, get_value
from functools import partial

import os

def create_directory_for_file(file_path):
    directory = os.path.dirname(file_path)
    
    if not os.path.exists(directory):
        os.makedirs(directory)


def int_to_bin_list(n, number=4):
    bin_str = format(n, 'b').zfill(number)
    return [int(b) for b in bin_str]

class CustomLogitsProcessor(LogitsProcessor):

    def __init__(self, llm_name):
        super().__init__()
        self.llm_name = llm_name

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.llm_name == "gpt2":
            for b_idx in range(input_ids.shape[0]):
                scores[b_idx][50256] = -10000
        elif self.llm_name == "opt-1.3b":
            for b_idx in range(input_ids.shape[0]):
                scores[b_idx][2] = -10000
        elif self.llm_name == "opt-2.7b":
            for b_idx in range(input_ids.shape[0]):
                scores[b_idx][2] = -10000
        return scores
    
class WatermarkLogitsProcessor(LogitsProcessor):

    def __init__(self, vocab, delta, model, window_size, cache, bit_number, beam_size, llm_name):
        self.vocab = vocab
        self.delta = delta
        self.model = model
        self.window_size = window_size
        self.cache = cache
        self.bit_number = bit_number
        self.llm_name = llm_name
        if beam_size > 0:
            self.beam_size = beam_size
            self.mode = "beam"
        else:
            self.mode = "sample"
    

    def _get_greenlist_ids(self, input_ids, scores):
        greenlist_ids = []
        # Get the last 'window_size - 1' items from input_ids
        last_nums = input_ids[-(self.window_size-1):]
        if self.mode == "sample":
            _, candidate_tokens = torch.topk(input=scores, k=20, largest=True, sorted=False)
        else:
            # Get the score at index 'beam_size'
            threshold_score = torch.topk(input=scores, k=self.beam_size, largest=True, sorted=False)[0][-1]
            
            # Get all indices where score is greater than 'score - delta'
            candidate_tokens = (scores >= (threshold_score - self.delta)).nonzero(as_tuple=True)[0]

        for v in candidate_tokens:
            # Append the current number to the list
            pair = list(last_nums) + [v]
            merged_tuple = tuple(pair)
            bin_list = [int_to_bin_list(num, self.bit_number) for num in pair]

            # load & update cache
            if merged_tuple in self.cache:
                result = self.cache[merged_tuple]
            else:
                result = get_value(torch.FloatTensor(bin_list).unsqueeze(0), self.model)
                self.cache[merged_tuple] = result
            if result:
                greenlist_ids.append(int(v))

        return greenlist_ids

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # if the length of input_id < self.window_size - 1, there is no need to add bias
        if input_ids.shape[-1] < self.window_size - 1:
            if self.llm_name == "gpt2":
                for b_idx in range(input_ids.shape[0]):
                    scores[b_idx][50256] = -10000
            elif self.llm_name == "opt-1.3b":
                for b_idx in range(input_ids.shape[0]):
                    scores[b_idx][2] = -10000
            elif self.llm_name == "opt-2.7b":
                for b_idx in range(input_ids.shape[0]):
                    scores[b_idx][2] = -10000
            return scores
        
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx], scores=scores[b_idx])
            green_tokens_mask[b_idx][greenlist_ids] = 1 
        green_tokens_mask = green_tokens_mask.bool()


        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
        
        if self.llm_name == "gpt2":
            for b_idx in range(input_ids.shape[0]):
                scores[b_idx][50256] = -10000
        elif self.llm_name == "opt-2.7b":
            for b_idx in range(input_ids.shape[0]):
                scores[b_idx][2] = -10000
        elif self.llm_name == "opt-1.3b":
            for b_idx in range(input_ids.shape[0]):
                scores[b_idx][2] = -10000

        return scores
    

class Watermark:
    def __init__(
        self,
        bit_number: int = 8,
        window_size: int = 3,
        layers: int = 3,
        gamma: float = 0.5,
        delta: float = 2.0,
        model_dir: str = None,
        beam_size: int = 0,
    ):
        # watermarking parameters
        self.bit_number = bit_number
        self.vocab = list(range(1, 2 ** bit_number-1))
        self.vocab_size = len(self.vocab)
        self.gamma = gamma
        self.min_prefix_len = window_size-1
        self.window_size = window_size
        self.model = get_model(bit_number, window_size, model_dir, layers) # 划分器
        self.cache = {}
        self.delta = delta
        self.beam_size = beam_size


    def random_sample(self, input_ids, is_green):
        # Get the last 'window_size - 1' items from input_ids
        last_nums = input_ids[-(self.window_size-1):]
        while True:
            number = random.choice(self.vocab)
            # Append the new random number to the list
            pair = list(last_nums) + [number]
            merged_tuple = tuple(pair)
            bin_list = [int_to_bin_list(num, self.bit_number) for num in pair]

            if merged_tuple in self.cache:
                result = self.cache[merged_tuple]
            else:
                result = get_value(torch.FloatTensor(bin_list).unsqueeze(0), self.model)
                self.cache[merged_tuple] = result

            if is_green and result:
                return number

            elif not is_green and not result:
                return number
            
    
    def judge_green(self, input_ids, current_number):
        # Get the last 'window_size - 1' items from input_ids
        last_nums = input_ids[-(self.window_size-1):]
        # Append the current number to the list
        pair = list(last_nums) + [current_number]
        merged_tuple = tuple(pair)
        bin_list = [int_to_bin_list(num, self.bit_number) for num in pair]
        # merged_list = sum(bin_list, [])

        # load & update cache
        if merged_tuple in self.cache:
            result = self.cache[merged_tuple]
        else:
            result = get_value(torch.FloatTensor(bin_list).unsqueeze(0), self.model)
            self.cache[merged_tuple] = result

        return result
    
    def green_token_mask_and_stats(self, input_ids: torch.Tensor):
        mask_list = []
        green_token_count = 0
        for idx in range(self.min_prefix_len, len(input_ids)):
            curr_token = input_ids[idx]
            if self.judge_green(input_ids[:idx], curr_token):
                mask_list.append(True)
                green_token_count += 1
            else:
                mask_list.append(False)
        num_tokens_scored = len(input_ids) - self.min_prefix_len
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return mask_list, green_token_count, z_score

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z
    
    def generate_list_with_green_ratio(self, length: int, green_ratio: float):

        token_list = random.sample(self.vocab, self.window_size - 1)
        is_green = []

        while len(token_list) < length:
            green = 1 if random.random() < green_ratio else 0
            if green:
                token = self.random_sample(torch.LongTensor(token_list), True)
                token_list.append(token)
                is_green.append(1)
            else:
                token = self.random_sample(torch.LongTensor(token_list), False)
                token_list.append(token)
                is_green.append(0)
        
        # loop
        is_green_append = []
        for i in range(0, self.window_size - 1):
            tail_slice = token_list[-(self.window_size - 1 - i):]
            head_slice = token_list[:i]
            input_slice = tail_slice + head_slice
            is_green_append.append(self.judge_green(input_slice, token_list[i]))
        
        is_green = is_green_append + is_green
                
        return token_list, is_green
    
    def generate_and_save_train_data(self, num_samples, output_dir):
        train_data = []
        for _ in tqdm(range(num_samples)):
            length = 200
            green_ratio = random.random()
            token_list, is_green = self.generate_list_with_green_ratio(length, green_ratio)
            _, _, z_score = self.green_token_mask_and_stats(torch.tensor(token_list))
            train_data.append((tuple(token_list), tuple(is_green), z_score))

        train_data = list(set(train_data))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, 'train_data.jsonl'), 'w') as f:
            for item in train_data:
                json.dump({"Input": [int(i) for i in item[0]], "Tag": [int(i) for i in item[1]], "Output": float(item[2])}, f)
                f.write('\n')
    
    
    def generate_and_save_test_data(self, llm_name, dataset_name, output_dir, sampling_temp, max_new_tokens):
        """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
        and generate watermarked text by passing it to the generate method of the model
        as a logits processor. """
    
        print("loading llm...")
        if llm_name == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
        elif llm_name == "opt-1.3b":
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
            model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float16).cuda()
        elif llm_name == "opt-2.7b":
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b", use_fast=False)
            model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", torch_dtype=torch.float16).cuda()


        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                        delta=self.delta,
                                                        model=self.model,
                                                        window_size=self.window_size,
                                                        cache=self.cache,
                                                        bit_number=self.bit_number,
                                                        beam_size = self.beam_size,
                                                        llm_name=llm_name)
        custom_processor = CustomLogitsProcessor(llm_name=llm_name)

        gen_kwargs = dict(max_new_tokens=max_new_tokens)

        if self.beam_size == 0:
            gen_kwargs.update(dict(
                do_sample=True, 
                top_k=20,
                temperature=sampling_temp
            ))
        else:
            gen_kwargs.update(dict(
                num_beams=self.beam_size
            ))
        
        print(gen_kwargs)

        generate_with_watermark = partial(
            model.generate,
            logits_processor=LogitsProcessorList([watermark_processor]), 
            **gen_kwargs
        )

        generate_without_watermark = partial(
            model.generate, 
            logits_processor=LogitsProcessorList([custom_processor]),
            **gen_kwargs
        )

        decoded_output_with_watermark = []
        decoded_output_without_watermark = []

        print("dataset")
        print(dataset_name)

        # load dataset
        print("loading dataset...")
        if dataset_name == "c4":
            with open("./original_data/c4_validation.json") as f1:
                lines = f1.readlines()
        elif dataset_name == "dbpedia":
            with open("./original_data/dbpedia_validation.json") as f1:
                lines = f1.readlines()[500:]

        idx = 1
        for line in lines: 
            try:
                if idx > 500: # you can change it
                    break
                data = json.loads(line)
                text = data['text']
                text_tokenized = (tokenizer(text, return_tensors="pt", add_special_tokens=True)).to(device)
                prompt_length = 30
                if text_tokenized["input_ids"].shape[-1] < 230:
                    continue
                
                prompt = {}
                prompt["input_ids"] = text_tokenized["input_ids"][:, : prompt_length]
                prompt["attention_mask"] = text_tokenized["attention_mask"][:, : prompt_length]

                print("generate_with_watermark...")
                output_with_watermark = generate_with_watermark(**prompt)
                output_with_watermark = output_with_watermark[:,prompt["input_ids"].shape[-1]:]

                print("generate_without_watermark...")
                output_without_watermark = text_tokenized["input_ids"][:,prompt_length:prompt_length + 200]

                _, _, z_score = self.green_token_mask_and_stats(output_with_watermark.squeeze(0))
                decoded_output_with_watermark.append({"Input": tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0], "Tag": 1, "Z-score": z_score})
                _, _, z_score = self.green_token_mask_and_stats(output_without_watermark.squeeze(0))
                decoded_output_without_watermark.append({"Input": tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0], "Tag": 0, "Z-score": z_score})
                
                print(idx)
                idx += 1

            except StopIteration:
                break

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        
        with open(os.path.join(output_dir, 'test_data.jsonl'), 'w') as f:
            for item in decoded_output_with_watermark:
                json.dump({"Input": item["Input"], "Tag": item["Tag"], "Z-score": item["Z-score"]}, f)
                f.write('\n')
            for item in decoded_output_without_watermark:
                json.dump({"Input": item["Input"], "Tag": item["Tag"], "Z-score": item["Z-score"]}, f)
                f.write('\n')



if __name__  == "__main__":
    ## use argparse to set three parameters, bit_number, num_samples, output_dir
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="gpt2")
    parser.add_argument("--dataset_name", type=str, default="c4")
    parser.add_argument("--bit_number", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--train_num_samples", type=int, default=10000)
    parser.add_argument("--model_dir", type=str, default="model/model_16_window_3_layer_5_new.pt")
    parser.add_argument("--output_dir", type=str, default="data1")
    parser.add_argument("--use_sampling", type=bool, default=True)
    parser.add_argument("--sampling_temp", type=float, default=0.7)
    parser.add_argument("--n_beams", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--delta", type=float, default=2.0)
    args = parser.parse_args()
    watermark = Watermark(args.bit_number, args.window_size, args.layers, delta=args.delta, model_dir=args.model_dir + "combine_model.pt", beam_size=args.n_beams)
    watermark.generate_and_save_train_data(args.train_num_samples, args.output_dir)
    watermark.generate_and_save_test_data(args.llm_name, args.dataset_name, args.output_dir, args.sampling_temp, args.max_new_tokens)