# -*- coding:utf-8 -*-

"""
    desc: Create a strong random seed. Otherwise, Python 2 would seed using
    the system time, which might be non-robust especially in the presence of
    concurrency.
    permalink: https://svn.python.org/projects/python/tags/r32/Lib/random.py  
    create: 2017.12.11
    modified by @sam.dm

"""

import hashlib
import numpy as np
import os
import random as _random
import struct
import sys


if sys.version_info < (3,):
    integer_types = (int, long)
else:
    integer_types = (int,)

def np_random(seed=None):
    if seed is not None and not (isinstance(seed, integer_types) and 0 <= seed):
        raise Exception('Seed must be a non-negative integer or omitted, not {}'.format(seed))

    seed = _seed(seed)

    rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(hash_seed(seed)))
    return rng, seed

def hash_seed(seed=None, max_bytes=8):
    """
    Args:
        seed (Optional[int]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.
    """
    if seed is None:
        seed = _seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode('utf8')).digest()
    return _bigint_from_bytes(hash[:max_bytes])

def _seed(a=None, max_bytes=8):
    """
    Args:
        a (Optional[int, str]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the seed.
    """
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        a = a.encode('utf8')
        a += hashlib.sha512(a).digest()
        a = _bigint_from_bytes(a[:max_bytes])
    elif isinstance(a, integer_types):
        a = a % 2**(8 * max_bytes)
    else:
        raise Exception('Invalid type for seed: {} ({})'.format(type(a), a))

    return a

def _bigint_from_bytes(bytes):
    sizeof_int = 4
    padding = sizeof_int - len(bytes) % sizeof_int
    bytes += b'\0' * padding
    int_count = int(len(bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum

def _int_list_from_bigint(bigint):
    # Special case 0
    if bigint < 0:
        raise Exception('Seed must be non-negative, not {}'.format(bigint))
    elif bigint == 0:
        return [0]

    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints
