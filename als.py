"""
Implementation of alternating least squares with regularization.

The alternating least squares with regularization algorithm ALS-WR was first
demonstrated in the paper Large-scale Parallel Collaborative Filtering for
the Netflix Prize. The authors discuss the method as well as how they
parallelized the algorithm in Matlab. This module implements the algorithm in
parallel in python with the built in concurrent.futures module.
"""

from os import cpu_count
import subprocess


import numpy as np
from scipy.sparse import csr_matrix

# pylint: disable=E1101
np.seterr(divide='ignore')
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

    def fit(self, ratings):
        """Fit the model to the given ratings.

        Args:
            ratings (numpy.ndarray or scipy.sparse): Ratings matrix of users x
                items.
        Returns:
            self

        """
        ratings = check_array(ratings, accept_sparse='csr')
        random_state = check_random_state(self.random_state)
        with open('random.pkl', 'wb') as state:
            pickle.dump(random_state.get_state(), state)
        sps.save_npz('ratings', ratings)
        subprocess.run(['python', 'fit_als.py', str(self.rank),
                        str(self.tol), str(self.alpha)])
        with np.load('features.npz') as loader:
            self.user_feats = loader['user']
            self.item_feats = loader['item']
        for _file in ['ratings.npz', 'features.npz', 'random.pkl']:
            os.remove(_file)
        self.ratings = ratings
        users, items = self.ratings.nonzero()
        X = np.hstack((users, items))
        y = self.ratings[users, items].A1
        self.reconstruction_err_ = self.score(X, y)
        return self

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
        pred = np.array([
            self.predict_one(user, item)
            for user, item in zip(non_zeros[0], non_zeros[1])])
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
