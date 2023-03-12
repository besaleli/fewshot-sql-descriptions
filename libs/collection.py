import pandas as pd

class Collection:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        
    def retrieve(self, query: pd.Series, n: int):
        raise NotImplementedError('Not implemented yet')
    
class RandomCollection(Collection):
    def __init__(self, dataset: pd.DataFrame):
        super().__init__(dataset)
        
    def retrieve(self, query: pd.Series, n: int):
        return self.dataset.sample(n=n).reset_index(drop=True)