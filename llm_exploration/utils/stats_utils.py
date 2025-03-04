import numpy as np
from typing import List, Union


def get_cumulative_sum(arr: List[Union[float, int]]) -> List[float]:
    """
    Returns cumulative sum of a list.
    Given an array arr, it returns an array cum_sum_arr, such that
        cum_sum[i] = sum_{j = 0}^i arr[i]
    (i follows 0 based indexing)
    Input:
        arr (List of floats or ints):
            an array of numbers

    Output:
        cum_sum_arr (List of floats):
            an array of numbers with the above property.
    """
    cum_sum_arr = np.array(arr, dtype=np.float64)

    cum_sum = 0
    for i in range(cum_sum_arr.shape[0]):
        cum_sum += cum_sum_arr[i]
        cum_sum_arr[i] = cum_sum

    return cum_sum_arr.tolist()


def get_cumulative_average(arr: List[Union[float, int]]) -> List[float]:
    """
    Returns the cumulative average of a list.
    Given an array arr, it returns an array cum_avg_arr, such that
        cum_avg_arr[i] = sum_{j = 0}^i arr[i] / (i + 1)
    (i follows 0 based indexing)

    Input:
        arr (List of floats or ints):
            an array of numbers

    Output:
        cum_avg_arr (List of floats):
            an array of numbers with the above property.
    """
    cum_arr = np.array(arr, dtype=np.float64)

    cum_sum = 0
    for i in range(cum_arr.shape[0]):
        cum_sum += cum_arr[i]
        cum_arr[i] = cum_sum

    length_arr = [i for i in range(len(arr) + 1)][1:]
    assert len(length_arr) == cum_arr.shape[0]
    length_arr = np.array(length_arr, dtype=np.float64)

    cum_avg_arr = cum_arr / length_arr
    return cum_avg_arr.tolist()
