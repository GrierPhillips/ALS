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
from sklearn.utils.validation import (check_array, check_is_fitted,
                                      check_random_state)


# pylint: disable=E1101


class ALS(object):
    """Implementation of Alternative Least Squares for Matrix Factorization.

    Attributes:
        rank (int): Integer representing the rank of the matrix factorization.
        lambda_ (float, default=0.1): Float representing the regularization
            penalty.
        tolerance (float, default=0.1): Float representing the step size at
            which to stop factorization.
        ratings (np.ndarray or scipy.sparse): Array like containing model data.
        item_features (np.ndarray): Array of shape m x rank where m represents
            the number of items contained in the data. Contains the latent
            features about items extracted by the factorization process.
        user_features (np.ndarray): Array of shape m x rank where m represents
            the number of users contained in the data. Contains the latent
            features about users extracted by the factorization process.

    """

    def __init__(self, rank, alpha=0.1, tolerance=0.001, random_state=None,
                 n_jobs=1, verbose=0):
        """Create instance of ALS with given parameters.

        Args:
            rank (int): Integer representing the rank of the matrix
                factorization.
            alpha (float, default=0.1): Float representing the regularization
                term.
            tolerance (float, default=0.001): Float representing the threshold
                that a step must be below before update iterations will stop.

        """
        self.rank = rank
        self.alpha = alpha
        self.tol = tolerance
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.ratings = None
        self.item_feats = None
        self.user_feats = None

    def fit(self, X):
        """Fit the model to the given ratings.

        Args:
            X (numpy.ndarray or scipy.sparse): Ratings matrix of users x
                items.
        Returns:
            self

        """
        _, _ = self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit the model to the given ratings.

        Args:
            X : {array-like, sparse matrix}, shape (n_samples, m_samples)
                Data matrix to be decomposed.

        Returns
        -------
            user_feats : array, shape (k_components, n_samples)
                The array of latent user features.

            item_feats : array, shape (k_components, m_samples)
                The array of latent item features.

        """
        ratings = check_array(X, accept_sparse='csr')
        random_state = check_random_state(self.random_state)
        with open('random.pkl', 'wb') as state:
            pickle.dump(random_state.get_state(), state)
        sps.save_npz('ratings', ratings)
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
        for _file in ['ratings.npz', 'features.npz', 'random.pkl']:
            os.remove(_file)
        self.ratings = ratings
        users, items = self.ratings.nonzero()
        X = np.hstack((users.reshape(-1, 1), items.reshape(-1, 1)))
        y = self.ratings[users, items].A1
        self.reconstruction_err_ = self.score(X, y)
        return self.user_feats, self.item_feats

    def predict_one(self, user, item):
        """Given a user and item provide the predicted rating.

        Predicted ratings for a single user, item pair can be provided by the
        fitted model by taking the dot product of the user row from the
        user_features and the item column from the item_features.

        Formula:
            rating = UiIj
            Where Ui is the row of features for user i and Ij is the column of
            features for item j.

        Args:
            user (int): Integer representing the user id.
            item (int): Integer representing the item id.
        Returns:
            rating (float): Float value of the predicted rating.

        """
        rating = self.user_feats.T[user].dot(self.item_feats[:, item])
        return rating

    def predict_all(self, user):
        """Given a user provide all of the predicted ratings.

        Args:
            user (int): Integer representing the user id.
        Returns:
            ratings (np.ndarray): Array containing predicted values for all
                items.

        """
        ratings = self.user_feats.T[user].dot(self.item_feats)
        return ratings

    def score(self, X, y):
        """Return the root mean squared error for the predicted values.

        Args:
            X : array-like
                Array containing row and column values for predictions.
            y : array-like
                The true values.

        Returns:
            rmse (float): The root mean squared error for the test set given
                the values predicted by the model.

        """
        check_is_fitted(self, ['item_feats', 'user_feats'])
        pred = np.array([X[i][0].dot(X[i][1]) for i in range(X.shape[0])])
        rmse = -self.root_mean_squared_error(y, pred)
        return rmse

    def update_user(self, user, item, rating):
        """Update a single user's feature vector.

        When an existing user rates an item the feature vector for that user
        can be updated withot having to rebuild the entire model. Eventually,
        the entire model should be rebuilt, but this is as close to a real-time
        update as is possible.

        Args:
            user (int): Integer representing the user id.
            item (int): Integer representing the item id.
            rating (int): Integer value of the rating assigned to item by user.
        """
        self.ratings[user, item] = rating
        submat = self.item_feats[:, self.ratings[user].indices]
        row = self.ratings[user].data
        col = self._update_one(submat, row, self.rank, self.lambda_)
        self.user_feats[:, user] = col

    def add_user(self, user_id):
        """Add a user to the model.

        When a new user is added append a new row to the ratings matrix and
        create a new column in user_feats. When the new user rates an item,
        the model will be ready insert the rating and use the update_user
        method to calculate the least squares approximation of the user
        features.

        Args:
            user_id (int): The index of the user in the ratings matrix.

        """
        shape = self.ratings._shape  # pylint: disable=W0212
        if user_id >= shape[0]:
            shape = (shape[0] + 1, shape[1])
        self.ratings.indptr = np.hstack(
            (self.ratings.indptr, self.ratings.indptr[-1]))
        if user_id >= self.user_feats.shape[1]:
            new_col = np.zeros((self.rank, 1))
            self.user_feats = np.hstack((self.user_feats, new_col))
