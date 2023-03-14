from typing import Union, List
import pandas as pd
import torch
import openai
from transformers import (PreTrainedModel,
                          PreTrainedTokenizer,
                          GPT2LMHeadModel,
                          T5ForConditionalGeneration,
                          StoppingCriteria,
                          StoppingCriteriaList)

from libs.utils import accommodate_openai


generation_format = """
# Query:
```sql
{}
```

# Describe the query above in a short sentence:
{}
"""


class EarlyStop(StoppingCriteria):
    def __init__(self, keywords: list, tokenizer: PreTrainedTokenizer):
        self.keywords = keywords
        self.stop_ids = [tokenizer.encode(w)[0] for w in keywords]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1] in self.stop_ids


class ModelInput:
    def __init__(self, query: str, examples: pd.DataFrame):
        self.query = query
        self.examples = examples
        
    def to_json(self):
        return dict(
            query=self.query,
            examples=self.examples.to_json(orient='records')
            )

class DescriptionGenerator:
    def __init__(self):
        pass
    
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
    
    def generate_description(self, model_inputs: List[ModelInput], generation_kwargs: Union[None, dict] = None):
        raise NotImplementedError('generate_description not implemented')

class HFDescriptionGenerator(DescriptionGenerator):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        super().__init__()
        
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        if type(self.model) not in [T5ForConditionalGeneration]:
            print('Warning: model is not T5ForConditionalGeneration, setting tokenizer policy to left padding')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = 'left'
    
    def remove_prompt_from_generation(self, output: torch.LongTensor, tokenized_prompt: dict):
        return [i[len(j):] for i, j in zip(output, tokenized_prompt['input_ids'])]
    
    def postproc_early_stop(self, generations: List[str], stop: Union[list, None] = None):
        if stop is None:
            return generations
        
        truncated_generations = generations 
        
        truncated_generations = [
            gen[
                :min(
                    [
                        gen.index(i) for i in (s for s in stop if s in gen)
                        ] or [len(gen)]
                    )
                ] for gen in generations
        ]
        
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
        
        if 'max_new_tokens' in generation_kwargs:
            for inp in tokenized_inputs.input_ids:
                if len(inp) + generation_kwargs['max_new_tokens'] > self.tokenizer.model_max_length:
                    print('Warning: max_new_tokens is too large for the model')
                    generation_kwargs['max_new_tokens'] = self.tokenizer.model_max_length - len(inp)
        
        generation_kwargs = generation_kwargs or dict()
        
        if 'stop' in generation_kwargs:
            stop = generation_kwargs.pop('stop')
        else:
            stop = None
        
        if 'echo' in generation_kwargs:
            echo = generation_kwargs.pop('echo')
        else:
            echo = False
        
        with torch.no_grad():
            if type(self.model) in [GPT2LMHeadModel, T5ForConditionalGeneration]:
                output = self.model.generate(**tokenized_inputs, **generation_kwargs)
            else:
                output = self.model.generate(tokenized_inputs.input_ids, **generation_kwargs)
        
        if not echo:
            output = self.remove_prompt_from_generation(output, tokenized_inputs)
        
        decoded_generation = [self.tokenizer.decode(i) for i in output]
        
        decoded_generation = self.postproc_early_stop(decoded_generation, stop=stop)
        
        return decoded_generation

class OpenAIDescriptionGenerator(DescriptionGenerator):
    def __init__(self, engine: str):
        super().__init__()
        self.engine = engine
    
    @accommodate_openai(max_tries=3, time_sleep=5)
    def generate_description(self, model_inputs: List[ModelInput], generation_kwargs: Union[None, dict] = None):
        prompts = [
            self.create_prompt(model_input.query, model_input.examples) for model_input in model_inputs
            ]
        
        if 'do_sample' in generation_kwargs:
            generation_kwargs.pop('do_sample')
            
        if 'max_new_tokens' in generation_kwargs:
            generation_kwargs['max_tokens'] = generation_kwargs.pop('max_new_tokens')
        
        completion = openai.Completion.create(
            engine=self.engine,
            prompt=prompts,
            **generation_kwargs)
        
        return [i['text'] for i in completion['choices']]
