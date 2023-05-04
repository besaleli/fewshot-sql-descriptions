import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
from bert_score import score
import numpy as np
import torch

df = {
    'question': [],
    'query': [],
    'generation': [],
    'retrieval_method': [],
    'nshot': [],
    'masked': [],
    'no_literals': []
}

for filename in glob.glob('results/*.json'):
    pattern = re.compile(
        r'results/text-curie-001_(\S*)_([0-9])shot_([0-9]+)mask(_no\-literals)?.json'
        )
    
    dff = pd.read_json(filename)

    retrieval_method, nshot, masked, no_literals = pattern.findall(filename)[0]
    df['question'] += dff['question'].to_list()
    df['query'] += dff['query'].to_list()
    df['generation'] += dff['generation'].to_list()
    df['retrieval_method'] += [retrieval_method] * len(dff)
    df['nshot'] += [int(nshot)] * len(dff)
    df['masked'] += [bool(int(masked))] * len(dff)
    df['no_literals'] += [bool(no_literals)] * len(dff)
    
df = pd.DataFrame(df)

print('total data points: ', len(df))

print('getting scores...')
precision, recall, f1 = score(
    df['generation'].to_list(),
    df['question'].to_list(),
    model_type='microsoft/deberta-large-mnli',
    device=torch.device('mps'),
    verbose=True
    )

df['precision'], df['recall'], df['f1'] = precision.tolist(), recall.tolist(), f1.tolist()

print('saving...')
df.to_json('data.json', orient='records')
