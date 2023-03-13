from transformers import (AutoModelForCausalLM, AutoTokenizer)

from libs.collection import TfIdfCollection
from libs.generation import DescriptionGenerator, ModelInput
from libs.dataset import get_sede

dataset = get_sede()

model = DescriptionGenerator(
    model=AutoModelForCausalLM.from_pretrained('gpt2'),
    tokenizer=AutoTokenizer.from_pretrained('gpt2')
    )

collection = TfIdfCollection(dataset['train'].to_pandas())

model_inputs = [
    ModelInput(
        row['QueryBody'],
        collection.retrieve(row, 1)
        ) for _, row in list(
            dataset['test'].to_pandas().iterrows()
            )[:5]
]

for inp in model_inputs:
    print(len(inp.examples))

gen = model.generate_description(
    model_inputs,
    generation_kwargs=dict(
        max_new_tokens=64,
        do_sample=True,
        top_p=0.95,
        stop=['#', '\n\n'])
    )

print(gen)
