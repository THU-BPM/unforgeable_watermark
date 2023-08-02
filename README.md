# A Private Watermark for Large Language Models
> https://arxiv.org/pdf/2307.16230.pdf
## conda environment
- python 3.9
- pytorch
- others：`pip install -r requirements.txt`
## four steps
### step 1: generate data for training watermark generator
We need to train the watermark generator network in order to divide the red/green list evenly (approximately half green and half red). 
```shell
python generate_data.py --bit_number 16 --window_size 3 --sample_number 2000 --output_file ./train_generator_data/train_generator_data.jsonl
```
The value of `bit_number` depends on the LLM you choose to use. For example, gpt2 has a vocabulary size of 50,257 ($2^{15}-1<50257<2^{16}-1$) and therefore we let `bit_number=16`.
### step 2: train watermark generator
```shell
python model_key.py --data_dir ./train_generator_data/train_generator_data.jsonl  --bit_number 16 --model_dir ./model/ --window_size 3 --layers 5
```
### step 3: generate training and testing data for watermark detector
- generate training data: 
  ```python
  watermark.generate_and_save_train_data(args.train_num_samples, args.output_dir)
  ```
  - LLM is not used in this step. 
  - We randomly sample 10,000 token id sequences of length 200 and calculate z-score using the watermark generator (judging whether each token is green or not).

- generate testing data: 
  ```python
  watermark.generate_and_save_test_data(args.llm_name, args.dataset_name, args.output_dir, args.sampling_temp, args.max_new_tokens)
  ```
  - LLM is used in this step. 
  - We use the validation set of c4 and dbpedia for experiments (you can find them in `./original_data`). 
  - Texts 1-500 in c4 and texts 501-1000 in dbpedia are selected to generate testing data, with prompt length of 30 and new-generated text length of 200.

```shell
python watermark_model.py --bit_number 16  --train_num_samples 10000 --dataset_name c4 --llm_name gpt2 --output_dir ./data --model_dir ./model/ --window_size 3 --layers 5 --use_sampling True --sampling_temp 0.7 --n_beams 0 --max_new_tokens 200 --delta 2.0
```
### step 4: train and test our private watermark detector
```shell
python detector.py --llm_name gpt2 --bit 16 --window_size 3 --input ./data --model_file ./model/sub_net.pt --output_model_dir ./model/ --layers 5 --z_value 4
```