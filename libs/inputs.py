import random
import os
import re
from typing import Union

import pandas as pd
import sqlglot

from libs.utils import sample_from_dict

random.seed(os.getenv('PD_RANDOM_STATE', 42))

def _mask_columns(query: str, sample: Union[int, None] = None, maps: Union[dict, None] = None):
    parsed_query = sqlglot.parse_one(query, read='sqlite')
    columns = parsed_query.find_all(sqlglot.exp.Column)
    if maps is not None:
        col_to_anon_maps = maps
    else:
        col_to_anon_maps = sample_from_dict(
            {col.alias_or_name: f'COL{i}' for i, col in enumerate(columns)},
            sample
            )
    
    def transformer(e):
        if isinstance(e, sqlglot.exp.Column):
            if e.alias_or_name in col_to_anon_maps:
                return sqlglot.parse_one(col_to_anon_maps[e.alias_or_name], read='sqlite')
        return e
    
    return parsed_query.transform(transformer).sql(), col_to_anon_maps

def _mask_literals(query: str):
    def transformer(e):
        if isinstance(e, sqlglot.exp.Literal):
            if e.is_number:
                return sqlglot.parse_one('0')

            return sqlglot.parse_one("'VAL'")
        
        return e
    
    return sqlglot.parse_one(query).transform(transformer).sql()
    
def _mask_tables(query: str, sample: Union[int, None] = None, maps: Union[dict, None] = None):
    parsed_query = sqlglot.parse_one(query, read='sqlite')
    tables = parsed_query.find_all(sqlglot.exp.Table)
    if maps is not None:
        table_to_anon_maps = maps
    else:
        table_to_anon_maps = sample_from_dict(
            {table.name: f'TABLE{i}' for i, table in enumerate(tables)},
            sample
            )

    def transformer(e):
        if isinstance(e, sqlglot.exp.Table):
            if e.name in table_to_anon_maps:
                return sqlglot.parse_one(table_to_anon_maps[e.name], read='sqlite')
        return e
        
    return parsed_query.transform(transformer).sql(), table_to_anon_maps

class ModelInput:
    def __init__(self, query: str, examples: pd.DataFrame):
        self.query = query
        self.examples = examples
        self.col_mask_map = None
        
    def to_dict(self) -> dict:
        return dict(
            query=self.query,
            examples=self.examples.to_dict(orient='records'),
            col_mask_map=self.col_mask_map
            )
    
    def mask_columns(self, n_to_mask: int):
        if n_to_mask == 0:
            return self
        
        self.query, col_mask_map = _mask_columns(self.query, n_to_mask)
        
        masked_queries = []
        for _, row in self.examples.iterrows():
            masked_query, _ = _mask_columns(row['query'], maps=col_mask_map)
            masked_queries.append(masked_query)
        
        self.examples['query'] = masked_queries
        self.col_mask_map = col_mask_map
        
        return self
    
    def mask_literals(self):
        self.query = _mask_literals(self.query)
        
        return self
