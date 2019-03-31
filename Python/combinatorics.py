
from math import factorial
import numpy as np

def perm_number(v):
    return int(factorial(sum(v)) / np.product([factorial(i) for i in v]))

def _next_partition_combination(n, current_set):
    new = set()
    for k in current_set.keys():
        delta = n - k
        for previous_k_set in current_set[k]:
            for i in range(len(previous_k_set)):
                k_plus_1_set = list(previous_k_set)
                k_plus_1_set[i] += delta
                new.add(tuple(k_plus_1_set))
    return new

def partition_combinations(n, k):
    """
    Generates combinations of k distinct objects, such that there are i total objects,
    for i from 0 - n

    This is related to the permutations of the partition of an integer i,
    but with explicit 0's added for the unrepresented k's

    Args:
        n: int. last integer for which to generate partitions
        k: int.

    Returns:

    """
    tmp = tuple([0] * k)
    current_set = {0: {tmp}}
    for i in range(1, n+1):
        current_set[i] = _next_partition_combination(i, current_set)
    return current_set
