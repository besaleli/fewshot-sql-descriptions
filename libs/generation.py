from typing import Union
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
        self.stop_ids = [tokenizer.encode(w)[0] for w in keywords]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1] in self.stop_ids


class DescriptionGenerator:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
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
        
    def generate_description(self, query: str, examples: Union[pd.DataFrame, pd.Series, None] = None, generation_kwargs: Union[None, dict] = None):
        prompt = self.create_prompt(query, examples)
        
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        
        generation_kwargs = generation_kwargs or dict()
        
        if 'stop' in generation_kwargs:
            generation_kwargs['stopping_criteria'] = StoppingCriteriaList([EarlyStop(generation_kwargs.pop('stop'), self.tokenizer)])
        
        with torch.no_grad():
            output = self.model.generate(**tokenized_prompt, **generation_kwargs)
        
        return self.tokenizer.decode(output.squeeze())
