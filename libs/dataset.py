import pandas as pd

from libs.collection import Collection
from libs.generation import ModelInput

def load_training_inputs(dataset: pd.DataFrame, collection: Collection, n: int, mask_columns: int = 0):
    return [
        ModelInput(
            row['query'],
            collection.retrieve(row, n)
            ).mask_columns(mask_columns) for _, row in dataset.iterrows()
    ]
