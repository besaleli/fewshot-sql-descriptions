import random
import os
import re

import pandas as pd

from libs.utils import get_columns

random.seed(os.getenv('PD_RANDOM_STATE', 42))

class ModelInput:
    def __init__(self, query: str, examples: pd.DataFrame):
        self.query = query
        self.examples = examples
        self.col_mask_map = None
        
    def to_json(self):
        return dict(
            query=self.query,
            examples=self.examples.to_json(orient='records'),
            col_mask_map=self.col_mask_map
            )
    
    def mask_columns(self, n_to_mask: int):
        if n_to_mask > 0:
            # get all columns
            parsed_columns = get_columns(self.query)
            
            # find all columns that occur in other examples, if exist
            if len(self.examples) > 0:
                parsed_columns = list(
                    filter(
                        lambda i: any(i in q for q in self.examples['QueryBody'].to_list()),
                        parsed_columns
                        )
                    )
            
            # sample columns to mask
            columns_to_mask = random.sample(parsed_columns, min(n_to_mask, len(parsed_columns)))
            
            self.col_mask_map = {c : f'COL{i}' for i, c in enumerate(columns_to_mask)}
            
            masked_query = self.query
            for col, col_map in self.col_mask_map.items():
                masked_query = re.sub(rf'(\W){col}(\W)', rf'\1{col_map}\2', masked_query, re.IGNORECASE)
                
            self.query = masked_query
            
            if len(self.examples) > 0:
                masked_examples = []
                for example in self.examples['QueryBody']:
                    masked_query = example
                    for col, col_map in self.col_mask_map.items():
                        masked_query = re.sub(rf'(\W){col}(\W)', rf'\1{col_map}\2', masked_query, re.IGNORECASE)
                    
                    masked_examples.append(masked_query)
                
                self.examples['QueryBody'] = masked_examples
            
        return self
