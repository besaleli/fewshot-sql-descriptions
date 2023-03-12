import datasets
from transformers import (GPT2LMHeadModel, AutoTokenizer)

from libs.collection import RandomCollection
from libs.generation import DescriptionGenerator

dataset = datasets.load_dataset('sede')

collection = RandomCollection(dataset['train'].to_pandas())

model = DescriptionGenerator(
    model=GPT2LMHeadModel.from_pretrained('gpt2'),
    tokenizer=AutoTokenizer.from_pretrained('gpt2')
    )

gen = model.generate_description(
    dataset['test'][0]['QueryBody'],
    collection.retrieve(dataset['test'].to_pandas().iloc[0], 3),
    generation_kwargs=dict(
        max_new_tokens=64,
        do_sample=True,
        top_p=0.95,
        stop=['\n', '#'])
    )

print(gen)
