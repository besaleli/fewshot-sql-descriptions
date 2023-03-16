from libs.collection import *
from libs.dataset import load_training_inputs
from libs.utils import batch
from libs.generation import DescriptionGenerator, HFDescriptionGenerator, OpenAIDescriptionGenerator, ChatGPTDescriptionGenerator
from libs.inputs import ModelInput

def get_collection_method(name: str) -> Collection:
    if name == 'random':
        return RandomCollection
    if name == 'tfidf':
        return TfIdfCollection
    if name == 'column_jaccard':
        return ColumnJaccardIndexCollection
    
    raise ValueError(f'Collection method {name} is not implemented')
