import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from libs import (
    get_sede,
    get_collection_method,
    load_training_inputs,
    batch,
    Collection,
    DescriptionGenerator
    )

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2', help='Autoregressive model to use for generation')
parser.add_argument('--collection', type=str, default='random', help='Collection method to use for fewshot example retrieval')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation')
parser.add_argument('--sample', '-s', type=int, default=None, help='Number of examples to sample from test set (default: None, use all examples)')
parser.add_argument('--nshot', '-n', type=int, default=3, help='Number of examples to use for fewshot generation')
parser.add_argument('--output_file', '-o', type=str, required=True, help='Output file to save generated descriptions to')
                    
args = parser.parse_args()

# load dataset
sede = get_sede()
df = sede['test'].to_pandas().reset_index(drop=True)

if args.sample:
    df = df.sample(n=args.sample).reset_index(drop=True)

# instantiate collection
collection: Collection = get_collection_method(args.collection)(sede['train'].to_pandas())

# load eval inputs
eval_inputs = load_training_inputs(
    dataset=df,
    collection=collection,
    n=args.nshot)

# load description generator
generator = DescriptionGenerator(
    model=AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype='auto',
        device_map='auto'
        ),
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    )

generated_descriptions = []

for input_batch in tqdm(list(batch(eval_inputs, args.batch_size))):
    generations = generator.generate_description(
        input_batch,
        generation_kwargs=dict(
            max_new_tokens=64,
            do_sample=True,
            top_p=0.95,
            stop=['#'])
        )

    generated_descriptions.extend(generations)
    
df['examples'] = [i.examples.to_json() for i in eval_inputs]
df['generation'] = generated_descriptions
df.to_json(args.output_file, orient='records')
