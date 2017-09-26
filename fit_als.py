
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import repeat
import pickle
import sys

import numpy as np
from numpy.linalg import LinAlgError

POOL_SIZE = 8


def fit_als(als, group):
    with open(als, 'rb') as f:
        als = pickle.load(f)
    if group is 'users':
        update_users(als)
    else:
        update_items(als)
    with open('als.pkl', 'wb') as f:
        pickle.dump(als, f)


def update_users(als):
    """Update the user features."""
    user_arrays = np.array_split(
        np.arange(als.ratings.shape[0]),
        POOL_SIZE
    )
    item_submats = als.make_item_submats()
    item_submat_arrays = np.array_split(item_submats, POOL_SIZE)
    rows = np.array_split(
        np.array(
            np.hsplit(als.ratings.data, als.ratings.indptr[1:-1])
        ),
        POOL_SIZE
    )
    als._update_parallel(user_arrays, item_submat_arrays, rows, 'user')


def update_items(als):
    """Update the item features."""
    item_arrays = np.array_split(
        np.arange(als.ratings.shape[1]),
        POOL_SIZE
    )
    user_submats = als.make_user_submats()
    user_submat_arrays = np.array_split(user_submats, POOL_SIZE)
    rows = np.array_split(
        np.array(
            np.hsplit(
                als.ratings.tocsc().data,
                als.ratings.tocsc().indptr[1:-1]
            )
        ),
        POOL_SIZE
    )
    als._update_parallel(item_arrays, user_submat_arrays, rows, 'item')


def _update_parallel(als, arrays, submat_arrays, rows, features):
    """Update the given features in parallel.

    Args:
        arrays (np.ndarray): Array of indices that represent which column
            of the features is being updated.
        submat_arrays (np.ndarray): Array of submatrices that will be used
            to calculate the update.
        rows (np.ndarray): Array of arrays that contain the ratings for the
            given feature column.
        features (string): The features that will be updated either 'user'
            or 'item'

    """
    with ProcessPoolExecutor() as pool:
        params = {'rank': als.rank, 'lambda_': als.lambda_}
        results = pool.map(
            als._thread_update_features,
            zip(arrays, submat_arrays, rows, repeat(params))
        )
        for result in results:
            for index, value in result.items():
                if features == 'item':
                    als.item_feats[:, index] = value
                else:
                    als.user_feats[:, index] = value


def _thread_update_features(als, args):
    """Split updates of feature matrices to multiple threads.

    Args:
        indices (np.ndarray): Array of integers representing the index of
            the user or item that is to be updated.
        submats (np.ndarray): Array of submatrices that will be used for
            updating the features.
        rows (np.ndarray): Array of rows that contain the ratings for the
            given user or item.
        params (dict): Parameters for the ALS algorithm
    Returns:
        data (dict): Dictionary of data with the user or item to be updated
            as key and the array of features as the values.

    """
    indices, submats, rows, params = args
    rank = params['rank']
    lambda_ = params['lambda_']
    data = {}
    with ThreadPoolExecutor() as pool:
        threads = {
            pool.submit(
                als._update_one,
                item_submat,
                row,
                rank,
                lambda_
            ): ind for ind, item_submat, row in zip(indices, submats, rows)
        }
    for thread in as_completed(threads):
        ind = threads[thread]
        result = thread.result()
        data[ind] = result
    return data


def _update_one(submat, row, rank, lam):
    """Update a single column for one of the feature matrices.

    Args:
        submat (np.ndarray): Submatrix of columns or rows from ratings
            corresponding to the reviews by a user or on an item.
        row (np.ndarray): Array of the ratings for the given item or user.
        rank (int): The rank of the feature arrays.
        lambda_ (float): The regularization parameter.
    Returns:
        col (np.ndarray): An array that represents a column from the
            feature matrix that is to be updated.

    """
    num_ratings = row.size
    reg_sums = submat.dot(submat.T)\
        + lam * num_ratings * np.eye(rank)
    feature_sums = submat.dot(row[np.newaxis].T)
    try:
        col = np.linalg.inv(reg_sums).dot(feature_sums)
    except LinAlgError:
        col = np.zeros((1, rank))
    return col.flatten()


if __name__ == '__main__':
    fit_als(sys.argv[0], users=sys.argv[1])
