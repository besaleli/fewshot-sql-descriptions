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
    ModelInput(dataset['test'][i]['QueryBody'], collection.retrieve(dataset['test'].to_pandas().iloc[i], 1)) for i in range(5)
    ]

gen = model.generate_description(
    model_inputs,
    generation_kwargs=dict(
        max_new_tokens=64,
        do_sample=True,
        top_p=0.95,
        stop=['#'])
    )

print(gen)
