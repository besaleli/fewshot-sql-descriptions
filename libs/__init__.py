from libs.collection import *
from libs.dataset import get_sede, load_training_inputs
from libs.utils import batch
from libs.generation import DescriptionGenerator

def get_collection_method(name: str) -> Collection:
    if name == 'random':
        return RandomCollection
    if name == 'tfidf':
        return TfIdfCollection
    
    raise ValueError(f'Collection method {name} is not implemented')
