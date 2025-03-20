import functools
import os
import pickle

def cache_result(func):
    """
    Decorator to cache function results on disk.
    """
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{func.__name__}_{hash(str(args) + str(kwargs))}.pkl"
        cache_path = os.path.join(cache_dir, key)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        result = func(*args, **kwargs)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        return result
    return wrapper
