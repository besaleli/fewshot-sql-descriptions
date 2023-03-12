from typing import Union, List
from dataclasses import dataclass
import pandas as pd
import torch
from transformers import (PreTrainedModel,
                          PreTrainedTokenizer,
                          StoppingCriteria,
                          StoppingCriteriaList)


generation_format = """
# Query:
```sql
{}
```

# Title:
{}
"""


class EarlyStop(StoppingCriteria):
    def __init__(self, keywords: list, tokenizer: PreTrainedTokenizer):
        self.keywords = keywords
        self.stop_ids = [tokenizer.encode(w)[0] for w in keywords]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1] in self.stop_ids


@dataclass
class ModelInput:
    query: str
    examples: pd.DataFrame


class DescriptionGenerator:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
    generation_format = generation_format.strip()
        
    def format_example(self, row: pd.Series) -> str:
        return self.generation_format.format(
            row['QueryBody'],
            row['Title']
            )
        
    def create_prompt(self, query: str, examples: Union[pd.DataFrame, None] = None):
        formatted_query = self.generation_format.format(query, '')
        
        if examples is None:
            return formatted_query
        
        formatted_examples = examples.apply(self.format_example, axis=1).to_list()
        
        return '\n\n'.join(formatted_examples + [formatted_query])
    
    def remove_prompt_from_generation(self, output: torch.LongTensor, tokenized_prompt: dict):
        return [i[len(j):] for i, j in zip(output, tokenized_prompt['input_ids'])]
    
    def postproc_early_stop(self, generations: List[str], early_stop: Union[EarlyStop, None] = None):
        if early_stop is None:
            return generations
        
        truncated_generations = generations 
        
        for keyword in early_stop.keywords:
            truncated_generations = [i[:-len(keyword)] if i.endswith(keyword) else i for i in truncated_generations]
            
        return truncated_generations
    
    def generate_description(self, model_inputs: List[ModelInput], generation_kwargs: Union[None, dict] = None):
        prompts = [
            self.create_prompt(model_input.query, model_input.examples) for model_input in model_inputs
            ]
        
        tokenized_inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True
            ).to(self.model.device)
        
        generation_kwargs = generation_kwargs or dict()
        
        if 'stop' in generation_kwargs:
            early_stop = EarlyStop(generation_kwargs.pop('stop'), self.tokenizer)
            generation_kwargs['stopping_criteria'] = StoppingCriteriaList([early_stop])
        else:
            early_stop = None
        
        if 'echo' in generation_kwargs:
            echo = generation_kwargs.pop('echo')
        else:
            echo = False
        
        with torch.no_grad():
            output = self.model.generate(**tokenized_inputs, **generation_kwargs)
        
        if not echo:
            output = self.remove_prompt_from_generation(output, tokenized_inputs)
        
        decoded_generation = [self.tokenizer.decode(i) for i in output]
        
        decoded_generation = self.postproc_early_stop(decoded_generation, early_stop=early_stop)
        
        return decoded_generation
