"""
Implementation of alternating least squares with regularization.

The alternating least squares with regularization algorithm ALS-WR was first
demonstrated in the paper Large-scale Parallel Collaborative Filtering for
the Netflix Prize. The authors discuss the method as well as how they
parallelized the algorithm in Matlab. This module implements the algorithm in
parallel in python with the built in concurrent.futures module.
"""

from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from itertools import repeat
from os import cpu_count

import numpy as np
from numpy.linalg import LinAlgError
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error

# pylint: disable=E1101

POOL_SIZE = cpu_count()


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

    def __init__(self, rank, lambda_=0.1, tolerance=0.001, seed=None):
        """Create instance of als with given parameters.

        Args:
            rank (int): Integer representing the rank of the matrix
                factorization.
            lambda_ (float, default=0.1): Float representing the regularization
                term.
            tolerance (float, default=0.001): Float representing the threshold
                that a step must be below before update iterations will stop.

        """
        self.rank = rank
        self.lambda_ = lambda_
        self.tolerance = tolerance
        self.rand = np.random.RandomState(seed)
        self.ratings = None
        self.item_feats = None
        self.user_feats = None

    @staticmethod
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

    def make_item_submats(self):
        """Construct array of all the item submatrices from a ratings matrix.

        Returns:
            submats (np.ndarray): Array containing the submatrix constructed by
                selecting the columns from the item features for the ratings
                that exist for each row in the ratings matrix.

        """
        idx = self.ratings.indptr
        col_arr = self.item_feats[:, self.ratings.indices]
        submat_list = [
            col_arr[:, row:col] for row, col in zip(idx[:-1], idx[1:])
        ]
        submats = np.empty(len(submat_list), dtype=object)
        for row, submat in enumerate(submat_list):
            submats[row] = submat
        return submats

    def make_user_submats(self):
        """Construct array of all the user submatrices from a ratings matrix.

        Returns:
            submats (np.ndarray): Array containing the submatrix constructed by
                selecting the columns from the user features for the ratings
                that exist for each column in the ratings matrix.

        """
        ratings = self.ratings.tocsc()
        idx = ratings.indptr
        col_arr = self.user_feats[:, ratings.indices]
        submat_list = [
            col_arr[:, row:col] for row, col in zip(idx[:-1], idx[1:])
        ]
        submats = np.empty(len(submat_list), dtype=object)
        for row, submat in enumerate(submat_list):
            submats[row] = submat
        return submats

    def fit(self, ratings):
        """Fit the model to the given ratings.

        Args:
            ratings (numpy.ndarray or scipy.sparse): Ratings matrix of users x
                items.

        """
        self.ratings = ratings
        rmse = float('inf')
        diff = rmse
        self.item_feats = self.rand.rand(self.rank * self.ratings.shape[1])\
            .reshape((self.rank, self.ratings.shape[1]))
        course_avg = self.ratings.sum(0) / (self.ratings != 0).sum(0)
        course_avg[np.isnan(course_avg)] = 0
        self.item_feats[0] = course_avg
        self.user_feats = np.zeros(self.rank * self.ratings.shape[0])\
            .reshape((self.rank, self.ratings.shape[0]))
        while diff > self.tolerance:
            self.update_users()
            self.update_items()
            true = self.ratings.data
            non_zeros = self.ratings.nonzero()
            pred = np.array(
                [
                    self.predict_one(user, item)
                    for user, item in zip(non_zeros[0], non_zeros[1])
                ]
            )
            new_rmse = self.root_mean_squared_error(true, pred)
            diff = rmse - new_rmse
            rmse = new_rmse

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

    def score(self, true):
        """Return the root mean squared error for the predicted values.

        Args:
            true (pd.DataFrame): A pandas DataFrame structured with the
                columns, 'Rating', 'User', 'Item'.

        Returns:
            rmse (float): The root mean squared error for the test set given
                the values predicted by the model.

        """
        if not isinstance(self.item_feats, np.ndarray):
            raise Exception('The model must be fit before generating a score.')
        ratings = csr_matrix((true.Rating, (true.User, true.Item)))
        non_zeros = ratings.nonzero()
        pred = np.array(
            [
                self.predict_one(user, item)
                for user, item in zip(non_zeros[0], non_zeros[1])
            ]
        )
        rmse = self.root_mean_squared_error(ratings.data, pred)
        return rmse

    def fit_transform(self, ratings):
        """Fit model to ratings and return predicted ratings.

        Args:
            ratings (numpy.ndarray or scipy.sparse): Ratings matrix of users x
                items.
        Returns:
            predictions (numpy.ndarray): Matrix of all predicted ratings.

        """
        self.fit(ratings)
        predictions = self.user_feats.T.dot(self.item_feats)
        return predictions

    def update_users(self):
        """Update the user features."""
        user_arrays = np.array_split(
            np.arange(self.ratings.shape[0]),
            POOL_SIZE
        )
        item_submats = self.make_item_submats()
        item_submat_arrays = np.array_split(item_submats, POOL_SIZE)
        rows = np.array_split(
            np.array(
                np.hsplit(self.ratings.data, self.ratings.indptr[1:-1])
            ),
            POOL_SIZE
        )
        self._update_parallel(user_arrays, item_submat_arrays, rows, 'user')

    def update_items(self):
        """Update the item features."""
        item_arrays = np.array_split(
            np.arange(self.ratings.shape[1]),
            POOL_SIZE
        )
        user_submats = self.make_user_submats()
        user_submat_arrays = np.array_split(user_submats, POOL_SIZE)
        rows = np.array_split(
            np.array(
                np.hsplit(
                    self.ratings.tocsc().data,
                    self.ratings.tocsc().indptr[1:-1]
                )
            ),
            POOL_SIZE
        )
        self._update_parallel(item_arrays, user_submat_arrays, rows, 'item')

    def _update_parallel(self, arrays, submat_arrays, rows, features):
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
            params = {'rank': self.rank, 'lambda_': self.lambda_}
            results = pool.map(
                self._thread_update_features,
                zip(arrays, submat_arrays, rows, repeat(params))
            )
            for result in results:
                for index, value in result.items():
                    if features == 'item':
                        self.item_feats[:, index] = value
                    else:
                        self.user_feats[:, index] = value

    def _thread_update_features(self, args):
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
                    self._update_one,
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

    @staticmethod
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
