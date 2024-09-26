from cachetools import TTLCache
import pandas as pd

cache = TTLCache(maxsize=100, ttl=3600)


def initialize_cache():
    global cache
    cache = TTLCache(maxsize=100, ttl=3600)


def get_cached_data(key):
    if key in cache:
        return cache[key]
    return None


def set_cached_data(key, value):
    cache[key] = value
