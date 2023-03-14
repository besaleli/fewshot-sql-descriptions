import datasets
import pandas as pd
from sql_metadata import Parser

from libs.collection import Collection
from libs.generation import ModelInput

def can_parse(query):
    try:
        Parser(query).tokens
        return True
    except:
        return False

def get_sede():
    ds = datasets.load_dataset('sede')
    ds = ds.filter(lambda x: can_parse(x['QueryBody']), with_indices=False)
        
    return ds

def load_training_inputs(dataset: pd.DataFrame, collection: Collection, n: int, mask_columns: int = 0):
    return [
        ModelInput(row['QueryBody'], collection.retrieve(row, n)).mask_columns(mask_columns) for _, row in dataset.iterrows()
    ]
