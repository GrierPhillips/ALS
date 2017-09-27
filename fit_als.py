"""Functions for multiprocessing of ALS-wr algorithm."""

import argparse
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from itertools import repeat
import os
import pickle
import sys

import numpy as np
from numpy.linalg import LinAlgError
from scipy.sparse._sparsetools import csr_tocsc
from scipy.sparse import csr_matrix
from scipy.sparse.sputils import get_index_dtype, upcast
from sklearn.metrics import mean_squared_error


PARSER = argparse.ArgumentParser(
    description='Run the Alternating Least Squares algorithm in parallel.')
PARSER.add_argument(
    'rank', metavar='Rank', type=int,
    help='The number of latent features (rank) in the matrix decomposition.')
PARSER.add_argument(
    'tol', metavar='Tolerance', type=float,
    help='The tolerance of the stopping condition.')
PARSER.add_argument(
    'alpha', metavar='Lambda', type=float,
    help='The regularization penalty.')
PARSER.add_argument(
    '-j', '--jobs', metavar='N_jobs', type=int,
    help='Set the number of jobs to run in parallel.',
    choices=[-1] + list(range(1, os.cpu_count())))
PARSER.add_argument(
    '-rs', '--random_state', metavar='Random_State',
    help='The random state to use. Must be either a filename of a pickled ' +
    'state of a RandomState or and integer.')
PARSER.add_argument(
    '-v', '--verbose', action='store_true',
    help='Enable verbose output.')
ARGS = PARSER.parse_args()
np.seterr(divide='ignore', invalid='ignore')
if ARGS.jobs == -1:
    ARGS.jobs = os.cpu_count()
RATINGS = sps.load_npz('rat_mat.npz')
if isinstance(ARGS.random_state, str) and ARGS.random_state.endswith('.pkl'):
    with open(ARGS.random_state, 'rb') as state:
        RANDOM_STATE = pickle.load(state)
else:
    RANDOM_STATE = np.random.RandomState(ARGS.random_state)
USER_FEATS = np.zeros((ARGS.rank, RATINGS.shape[0]))
ITEM_FEATS = RANDOM_STATE.rand(ARGS.rank, RATINGS.shape[1])


def fit_als():
    """Iterate through the ALS algorithm until the fit conditions are met."""
    rmse = float('inf')
    diff = rmse
    item_avg = RATINGS.sum(0) / (RATINGS != 0).sum(0)
    item_avg[np.isnan(item_avg)] = 0
    ITEM_FEATS[0] = item_avg
    while diff > ARGS.tol:
        update()
        true = RATINGS.data
        non_zeros = RATINGS.nonzero()
        pred = np.array([
            predict_one(user, item)
            for user, item in zip(non_zeros[0], non_zeros[1])])
        new_rmse = root_mean_squared_error(true, pred)
        diff = rmse - new_rmse
        rmse = new_rmse
    np.savez('features', user=USER_FEATS, item=ITEM_FEATS)


def make_user_submats(item):
    """Get the user submatrix from a single item in the ratings matrix.

    Returns:
        submat (np.ndarray): Array containing the submatrix constructed by
            selecting the columns from the user features for the ratings
            that exist for the given column in the ratings matrix.

    """
    idx_dtype = get_index_dtype(
        (RATINGS.indptr, RATINGS.indices),
        maxval=max(RATINGS.nnz, RATINGS.shape[0]))
    indptr = np.empty(RATINGS.shape[1] + 1, dtype=idx_dtype)
    indices = np.empty(RATINGS.nnz, dtype=idx_dtype)
    data = np.empty(RATINGS.nnz, dtype=upcast(RATINGS.dtype))
    csr_tocsc(RATINGS.shape[0], RATINGS.shape[1],
              RATINGS.indptr.astype(idx_dtype),
              RATINGS.indices.astype(idx_dtype),
              RATINGS.data, indptr, indices, data)
    submat = USER_FEATS[:, indices[indptr[item]:indptr[item + 1]]]
    return submat


def make_item_submats(user):
    """Get the item submatrix from a single user in the ratings matrix.

    Returns:
        submat (np.ndarray): Array containing the submatrix constructed by
            selecting the columns from the item features for the ratings
            that exist for the given row in the ratings matrix.

    """
    idx = RATINGS.indptr
    submat = ITEM_FEATS[:, RATINGS.indices[idx[user]:idx[user + 1]]]
    return submat


def update():
    """Update the user and item features.

    Args:
        axis (int, default: 0): The axis upon which to compute the update.

    """
    user_arrays = np.array_split(np.arange(RATINGS.shape[0]), ARGS.jobs)
    _update_parallel(user_arrays)
    item_arrays = np.array_split(np.arange(RATINGS.shape[1]), ARGS.jobs)
    _update_parallel(item_arrays, user=False)


def _update_parallel(arrays, user=True):
    """Update the given features in parallel.

    Args:
        arrays (np.ndarray): Array of indices that represent which column
            of the features is being updated.
        user (bool, default: True): Boolean indicating wheter or not user
            features are being updated.

    """
    with ProcessPoolExecutor(max_workers=ARGS.jobs) as pool:
        params = {'rank': ARGS.rank, 'alpha': ARGS.alpha, 'user': user}
        results = pool.map(
            _thread_update_features,
            zip(arrays, repeat(params)))
        for result in results:
            for index, value in result.items():
                if user:
                    USER_FEATS[:, index] = value
                else:
                    ITEM_FEATS[:, index] = value


def _thread_update_features(args):
    """Split updates of feature matrices to multiple threads.

    Args:
        indices (np.ndarray): Array of integers representing the index of
            the user or item that is to be updated.
        params (dict): Parameters for the ALS algorithm.
    Returns:
        data (dict): Dictionary of data with the user or item to be updated
            as key and the array of features as the values.

    """
    indices, params = args
    data = {}
    with ThreadPoolExecutor() as pool:
        threads = {
            pool.submit(_update_one, index, **params): index
            for index in indices}
    for thread in as_completed(threads):
        ind = threads[thread]
        result = thread.result()
        data[ind] = result
    return data


def _update_one(index, **params):
    """Update a single column for one of the feature matrices.

    Args:
        index (int): Integer representing the index of the user/item that is
            to be updated.
        params (dict): Parameters for the ALS algorithm.
    Returns:
        col (np.ndarray): An array that represents a column from the
            feature matrix that is to be updated.

    """
    rank, alpha, user = params['rank'], params['alpha'], params['user']
    if user:
        submat = make_item_submats(index)
        row = RATINGS[index].data
    else:
        submat = make_user_submats(index)
        row = RATINGS[:, index].data
    num_ratings = row.size
    reg_sums = submat.dot(submat.T) + alpha * num_ratings * np.eye(rank)
    feature_sums = submat.dot(row[np.newaxis].T)
    try:
        col = np.linalg.inv(reg_sums).dot(feature_sums)
    except LinAlgError:
        col = np.zeros((1, rank))
    return col.ravel()


def predict_one(user, item):
    """Given a user and item provide the predicted rating.

    Predicted ratings for a single user, item pair can be provided by the
    fitted model by taking the dot product of the user row from the
    user_features and the item column from the item_features.

    Formula:
        rating = U_iV_j
        Where U_i is the row of features for user i and V_j is the column of
        features for item j.

    Args:
        user (int): Integer representing the user id.
        item (int): Integer representing the item id.
    Returns:
        rating (float): Float value of the predicted rating.

    """
    rating = USER_FEATS.T[user].dot(ITEM_FEATS[:, item])
    return rating


def root_mean_squared_error(true, pred):
    """Calculate the root mean sqaured error.

    Args:
        true (np.ndarray): Array like of true values.
        pred (np.ndarray): Array like of predicted values.
    Returns:
        rmse (float): Root mean squared error for the given values.

    """
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    return rmse


if __name__ == '__main__':
    fit_als()
