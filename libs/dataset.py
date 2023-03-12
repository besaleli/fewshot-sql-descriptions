import datasets
from sql_metadata import Parser

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
    