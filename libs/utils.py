from typing import Callable
import time

def batch(l: list, n: int) -> list:
    for i in range(0, len(l), n):
        yield l[i:i + n]

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
