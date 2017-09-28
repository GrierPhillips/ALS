"""
Implementation of alternating least squares with regularization.

The alternating least squares with regularization algorithm ALS-WR was first
demonstrated in the paper Large-scale Parallel Collaborative Filtering for
the Netflix Prize. The authors discuss the method as well as how they
parallelized the algorithm in Matlab. This module implements the algorithm in
parallel in python with the built in concurrent.futures module.
"""

import os
import pickle
import subprocess

import numpy as np
import scipy.sparse as sps
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import (check_array, check_is_fitted,
                                      check_random_state)


# pylint: disable=E1101,W0212


def root_mean_squared_error(true, pred):
    """Calculate the root mean sqaured error.

    Parameters
    ----------
        true : array, shape (n_samples)
            Array of true values.
        pred : array, shape (n_samples)
            Array of predicted values.
    Returns
    -------
        rmse : float
            Root mean squared error for the given values.

    """
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    return rmse


def _check_x(X):
    if isinstance(X, tuple):
        if len(X) != 2:
            raise ValueError('Argument X should be a tuple of length 2 '
                             'containing an array for user attributes and an '
                             'array for item attributes.')
        Y = np.array(X[1])
        X = np.array(X[0])
    elif isinstance(X, DataHolder):
        Y = X.Y
        X = X.X
    else:
        raise TypeError('Type of argument X should be tuple or DataHolder, was'
                        ' {}.'.format(str(type(X)).split("'")[1]))
    if Y.ndim != 2 or X.ndim != 2:
        Y = Y.reshape(1, -1)
        X = X.reshape(1, -1)
    return X, Y


class DataHolder(object):
    """Class for packing user and item attributes into sigle data structure.

    Parameters
    ----------
    X : array-like, shape (n_samples, p_attributes)
        Array of user attributes. Each row represents a user.

    Y : array-like, shape (m_samples, q_attributes)
        Array of item attributes. Each row represents an item.

    """

    def __init__(self, X, Y):
        """Initialize instance of DataHolder."""
        self.X = X
        self.Y = Y
        self.shape = self.X.shape

    def __getitem__(self, x):
        """Return a tuple of the requested index for both X and Y."""
        return self.X[x], self.Y[x]


class ALS(BaseEstimator):
    """Implementation of Alternative Least Squares for Matrix Factorization.

    Parameters
    ----------
    rank : integer
        The number of latent features (rank) to include in the matrix
        factorization.

    alpha : float, optional (default=0.1)
        Float representing the regularization penalty.

    tolerance : float, optional (default=0.1)
        Float representing the difference in RMSE between iterations at which
        to stop factorization.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the ALS fitting process.

    Attributes
    ----------
    ratings : {array-like, sparse matrix} shape (n_samples, m_samples)
        Constant matrix representing the data to be modeled.

    item_features : array-like, shape (k_features, m_samples)
        Array of shape (rank, m_samples) where m represents the number of items
        contained in the data. Contains the latent features of items extracted
        by the factorization process.

    user_features : array-like, shape (k_features, n_samples)
        Array of shape (rank, n_samples) where n represents the number of users
        contained in the data. Contains the latent features of users extracted
        by the factorization process.

    """

    def __init__(self, rank, alpha=0.1, tolerance=0.001, random_state=None,
                 n_jobs=1, verbose=0):
        """Initialize instance of ALS."""
        self.rank = rank
        self.alpha = alpha
        self.tol = tolerance
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.data = None
        self.item_feats = None
        self.user_feats = None

    def fit(self, X):
        """Fit the model to the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, m_samlpes)
            Constant matrix representing the data to be modeled.

        Returns
        -------
        self

        """
        _, _ = self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit the model to the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, m_samples)
            Constant matrix representing the data to be modeled.

        Returns
        -------
        user_feats : array, shape (k_components, n_samples)
            The array of latent user features.

        item_feats : array, shape (k_components, m_samples)
            The array of latent item features.

        """
        data = check_array(X, accept_sparse='csr')
        random_state = check_random_state(self.random_state)
        with open('random.pkl', 'wb') as state:
            pickle.dump(random_state.get_state(), state)
        sps.save_npz('data', data)
        try:
            subprocess.run(
                ['python', 'fit_als.py', str(self.rank), str(self.tol),
                 str(self.alpha), '-rs', 'random.pkl', '-j',
                 str(self.n_jobs), '-v', str(self.verbose)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as err:
            err_msg = '\n\t'.join(err.stderr.decode().split('\n'))
            raise ValueError('Fitting ALS failed with error:\n\t{}'
                             .format(err_msg))
        with np.load('features.npz') as loader:
            self.user_feats = loader['user']
            self.item_feats = loader['item']
        for _file in ['data.npz', 'features.npz', 'random.pkl']:
            os.remove(_file)
        self.data = data
        users, items = self.data.nonzero()
        X = np.hstack((users.reshape(-1, 1), items.reshape(-1, 1)))
        y = self.data[users, items].A1
        self.reconstruction_err_ = self.score(X, y)
        return self.user_feats, self.item_feats

    def _predict(self, X):
        """Make predictions for the given arrays.

        Parameters
        ----------
        X : tuple, len = 2
            Tuple containing arrays of user indices and item indices.

        Returns
        -------
        predictions : array, shape (n_samples, m_samples)
            Array of all predicted values for the given user/item pairs.

        """
        X, Y = _check_x(X)
        X = check_array(X)
        Y = check_array(Y)
        predictions = np.array([
            self.user_feats[:, X[i]].T.dot(self.item_feats[:, Y[i]])
            for i in range(X.shape[0])])
        return predictions

    def predict_one(self, user, item):
        """Given a user and item provide the predicted rating.

        Predicted values for a single user, item pair can be provided by the
        fitted model by taking the dot product of the user column from the
        user_features and the item column from the item_features.

        Parameters
        ----------
        user : integer
            Index for the user.

        item : integer
            Index for the item.

        Returns
        -------
        prediction : float
            Predicted value at index user, item in original data.

        """
        prediction = self.user_feats.T[user].dot(self.item_feats[:, item])
        return prediction

    def predict_all(self, user):
        """Given a user provide all of the predicted values.

        Parameters
        ----------
        user : integer
            Index for the user.

        Returns
        -------
        predictions : array-like, shape (1, m_samples)
            Array containing predicted values of all items for the given user.

        """
        predictions = self.user_feats.T[user].dot(self.item_feats)
        return predictions

    def score(self, X, y):
        """Return the root mean squared error for the predicted values.

        Parameters
        ----------
        X : array-like
            Array containing row and column values for predictions.
        y : array-like
            The true values.

        Returns
        -------
        rmse : float
            The root mean squared error for the test set given the values
            predicted by the model.

        """
        check_is_fitted(self, ['item_feats', 'user_feats'])
        pred = np.array([
            self.user_feats[:, X[i][0]].T.dot(self.item_feats[:, X[i][1]])
            for i in range(X.shape[0])])
        rmse = -root_mean_squared_error(y, pred)
        return rmse

    def update_user(self, user, item, rating):
        """Update a single user's feature vector.

        When an existing user rates an item the feature vector for that user
        can be updated withot having to rebuild the entire model. Eventually,
        the entire model should be rebuilt, but this is as close to a real-time
        update as is possible.

        Parameters
        ----------
        user : integer
            Index for the user.

        item : integer
            Index for the item

        rating : integer
            The value assigned to item by user.

        """
        self.data[user, item] = rating
        submat = self.item_feats[:, self.data[user].indices]
        row = self.data[user].data
        col = self._update_one(submat, row, self.rank, self.lambda_)
        self.user_feats[:, user] = col

    def add_user(self, user_id):
        """Add a user to the model.

        When a new user is added append a new row to the data matrix and
        create a new column in user_feats. When the new user rates an item,
        the model will be ready insert the rating and use the update_user
        method to calculate the least squares approximation of the user
        features.

        Parameters
        ----------
        user_id : integer
            The index for the user.

        """
        shape = self.data._shape
        if user_id >= shape[0]:
            shape = (shape[0] + 1, shape[1])
        self.data.indptr = np.hstack(
            (self.data.indptr, self.data.indptr[-1]))
        if user_id >= self.user_feats.shape[1]:
            new_col = np.zeros((self.rank, 1))
            self.user_feats = np.hstack((self.user_feats, new_col))
