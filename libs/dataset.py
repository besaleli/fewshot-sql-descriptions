import pandas as pd

from libs.collection import Collection
from libs.generation import ModelInput

def load_training_inputs(dataset: pd.DataFrame, collection: Collection, n: int, mask_columns: int = 0, mask_literals: bool = False):
    inputs = [
        ModelInput(
            row['query'],
            collection.retrieve(row, n)
            ).mask_columns(mask_columns) for _, row in dataset.iterrows()
    ]
    
    if mask_literals:
        return [i.mask_literals() for i in inputs]
    
    return inputs
