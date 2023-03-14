import pandas as pd
from sql_metadata import Parser
import random

class ModelInput:
    def __init__(self, query: str, examples: pd.DataFrame):
        self.query = query
        self.examples = examples
        
    def to_json(self):
        return dict(
            query=self.query,
            examples=self.examples.to_json(orient='records')
            )
    
    def mask_columns(self, n_to_mask: int):
        if n_to_mask > 0:
            parsed_columns = Parser(self.query).columns
            columns_to_mask = random.sample(parsed_columns, min(n_to_mask, len(parsed_columns)))
            
            col_maps = {c : f'COL{i}' for i, c in enumerate(columns_to_mask)}
            
            for col, col_map in col_maps.items():
                self.query = self.query.replace(col_map, col)
            
            masked_examples = self.examples['QueryBody']
            for example in masked_examples:
                for col, col_map in col_maps.items():
                    example = str(example).replace(col, col_map)
                    
            self.examples['QueryBody'] = masked_examples
            self.examples['col_mask_map'] = [col_maps for _ in range(len(self.examples))]
