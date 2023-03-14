import argparse
import os

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm.auto import tqdm
import openai

from libs import (
    get_sede,
    get_collection_method,
    load_training_inputs,
    batch,
    Collection,
    HFDescriptionGenerator,
    OpenAIDescriptionGenerator,
    ChatGPTDescriptionGenerator
    )

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2', help='Autoregressive model to use for generation')
parser.add_argument('--seq2seq', action='store_true', help='Use seq2seq model for generation')
parser.add_argument('--eightbit', action='store_true', help='Use 8-bit quantization for generation')
parser.add_argument('--collection', type=str, default='random', help='Collection method to use for fewshot example retrieval')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation')
parser.add_argument('--sample', '-s', type=int, default=None, help='Number of examples to sample from test set (default: None, use all examples)')
parser.add_argument('--random_state', type=int, default=42, help='Random state for sampling examples from test set (default: 42)')
parser.add_argument('--nshot', '-n', type=int, default=3, help='Number of examples to use for fewshot generation')
parser.add_argument('--mask_columns', type=int, default=0, help='Number of columns to mask in query (default: 0, do not mask any columns)')
parser.add_argument('--output_file', '-o', type=str, required=True, help='Output file to save generated descriptions to')

args = parser.parse_args()

os.environ['PD_RANDOM_STATE'] = str(args.random_state)

# load dataset
sede = get_sede()
df = sede['test'].to_pandas().reset_index(drop=True)

if args.sample:
    df = df.sample(n=args.sample, random_state=args.random_state).reset_index(drop=True)

# instantiate collection
collection: Collection = get_collection_method(args.collection)(sede['train'].to_pandas())

# load eval inputs
eval_inputs = load_training_inputs(
    dataset=df,
    collection=collection,
    n=args.nshot,
    mask_columns=args.mask_columns
    )

# load description generator
model_kwargs = dict(
    torch_dtype='auto',
    device_map='auto',
    load_in_8bit=args.eightbit
    )

if os.path.exists('.openai_api_key'):
    openai.api_key = open('.openai_api_key', 'r').read().strip()

if args.model in ['text-davinci-003', 'text-curie-001', 'text-babbage-001']:
    generator = OpenAIDescriptionGenerator(args.model)
elif args.model == 'chatgpt':
    generator = ChatGPTDescriptionGenerator('gpt-3.5-turbo')
else:
    model_architecture = AutoModelForSeq2SeqLM if args.seq2seq else AutoModelForCausalLM

    generator = HFDescriptionGenerator(
        model=model_architecture.from_pretrained(
            args.model,
            **model_kwargs
            ),
        tokenizer=AutoTokenizer.from_pretrained(args.model)
        )

    print('DEVICE MAP: ')
    print(generator.model.hf_device_map)

generated_descriptions = []

for input_batch in tqdm(list(batch(eval_inputs, args.batch_size))):
    generations = generator.generate_description(
        input_batch,
        generation_kwargs=dict(
            max_new_tokens=64,
            stop=['#', '\n\n'])
        )

    generated_descriptions.extend(generations)

df['model_input'] = [i.to_json() for i in eval_inputs]
df['generation'] = generated_descriptions
df.to_json(args.output_file, orient='records')
