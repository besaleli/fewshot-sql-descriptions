from typing import Callable, Union
import random

import time
import sqlglot


def batch(l: list, n: int) -> list:
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def sample_from_dict(d: dict, sample: Union[None, int]):
    if sample in [0, None]:
        return d
    
    keys = random.sample(list(d), sample if sample < len(d) else len(d))
    return dict(zip(keys, [d[k] for k in keys]))
  
def get_columns(query: str) -> list:
    parsed_tree = sqlglot.parse_one(query, read='sqlite')
    columns = parsed_tree.find_all(sqlglot.exp.Column)
    return [i.alias_or_name for i in columns]

def accommodate_openai(max_tries: int = 5, time_sleep: int = 10) -> Callable:
    """Acommodates for OpenAI API rate limits.
    Args:
        max_tries (int, optional): Max tries to call API. Defaults to 5.
        time_sleep (int, optional): Time to sleep in between calls (seconds). Defaults to 10.
    Returns:
        Callable: Accommodated function.
    """
    def accommodater(func: Callable):
        def wrapper(*args, **kwargs):
            response = None
            try_count = 0
            while response is None:
                try:
                    response = func(*args, **kwargs)
                except Exception as exc:
                    print(f"[ERROR WITH OPENAI API] [{type(exc).__name__}] {str(exc)}")
                    if try_count > max_tries:
                        raise exc
                    time.sleep(time_sleep)
                    try_count += 1
            return response
        return wrapper
    return accommodater
