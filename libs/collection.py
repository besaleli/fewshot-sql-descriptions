import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from sql_metadata import Parser

class Collection:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset.reset_index(drop=True)
        
    def retrieve(self, query: pd.Series, n: int):
        raise NotImplementedError('Not implemented yet')
    
class RandomCollection(Collection):
    def __init__(self, dataset: pd.DataFrame):
        super().__init__(dataset)
        
    def retrieve(self, query: pd.Series, n: int):
        return self.dataset.sample(n=n).reset_index(drop=True)
    
class TfIdfCollection(Collection):
    def __init__(self, dataset: pd.DataFrame):
        super().__init__(dataset)
        
        print('training model...')
        self.tfidf_model = self.get_tfidf_model()
        
        print('getting vectors...')
        self.tfidf_vectors = self.get_collection_tfidf_vectors()
        
    def get_tfidf_model(self) -> TfidfVectorizer:
        tfidf_model = TfidfVectorizer(
            tokenizer=lambda x: [str(i) for i in Parser(x).tokens],
            lowercase=True
            ).fit(self.dataset['QueryBody'].to_list())
        
        return tfidf_model
    
    def get_collection_tfidf_vectors(self) -> list:
        return [
            i for i in self.tfidf_model.transform(self.dataset['QueryBody'])
            ]
        
    def fit_query(self, query: str) -> np.ndarray:
        return self.tfidf_model.transform([query]).toarray().squeeze()
    
    def retrieve(self, query: pd.Series, n: int) -> pd.DataFrame:
        query_tfidf = self.fit_query(query['QueryBody'])
        
        similarities = torch.tensor(
            [1 - cosine(query_tfidf, i.toarray().squeeze()) for i in self.tfidf_vectors]
            )
        
        candidates = torch.topk(similarities, n)
        cand_similarities, cand_idx = candidates.values, candidates.indices
        
        cand_df = self.dataset.iloc[cand_idx].reset_index(drop=True)
        cand_df['similarity'] = cand_similarities
        assert len(cand_df) == n, 'Number of candidates is not equal to n'
        
        return cand_df
