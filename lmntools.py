import time
import tqdm
import pickle
import numpy as np
import numexpr as ne
import scipy as sp
import scipy.sparse
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import train_test_split


class Cache:
    def __init__(self, *args):
        for arg in args:
            setattr(self, arg, None)

    def clear(self):
        for key in self.__dict__.keys():
            self.__dict__[key] = None


class BaseLocalModels(BaseEstimator):
    """
    Abstract base class for all local models.

    This class manages the parameters of all of the M different models of the same type.
    It provides methods to in- and decrease the number of models as well as to estimate the models parameters.
    Note: The center coordinates (C) and the validity functions (Phi)
     for each model are managed in the network structure class.
    """
    DEFAULT_CASES = None

    def __init__(self):
        self.Theta_ = None
        self.M_ = 0  # number of local models
        self.p = 0   # number of parameters
    
    @property
    def M(self):
        return self.M_
    
    # --- public functions ---
    def initiate(self, N):
        raise NotImplementedError

    def update_theta(self, X, y, A, R=None, model_pointers=None):
        raise NotImplementedError
    
    def update_theta_recursive(self, x, y, A):
        raise NotImplementedError

    def increase_number_of_models(self):
        raise NotImplementedError

    def decrease_number_of_models(self):
        raise NotImplementedError

    def get_output(self, u):
        raise NotImplementedError

    @staticmethod
    def get(identifier):
        try:
            if isinstance(identifier, BaseLocalModels):
                return identifier
            elif isinstance(identifier, str):
                for subclass in BaseLocalModels.__subclasses__():
                    if identifier in subclass.DEFAULT_CASES:
                        return subclass(**subclass.DEFAULT_CASES[identifier])
            else:
                raise ValueError
        except (NameError, ValueError):
            raise ValueError("Could not interpret local model identifier: " + str(identifier))


class PolynomialRegressionModels(BaseLocalModels):
    DEFAULT_CASES = {'const': {'degree': 0}, 'linear': {'degree': 1},
                     'quadratic': {'degree': 2}, 'cubic': {'degree': 3}}

    def __init__(self, degree=1, forgetting_factor=0.99, activity_threshold=0.2,
                 init_covariance=1000, optimization='local'):
        
        super().__init__()
        self.degree = degree
        self.forgetting_factor = forgetting_factor
        self.activity_threshold = activity_threshold
        self.init_covariance = init_covariance
        self.poly = None
        self.p = None
        self.Theta_ = None
        self.P_ = None
        self.optimization = optimization

    # --- public functions ---
    @staticmethod
    def num_parameters(N, degree):
        return int(sp.special.factorial(N + degree) / (sp.special.factorial(N) * sp.special.factorial(degree)))

    def initiate(self, N):
        self.M_ = 1
        self.p = PolynomialRegressionModels.num_parameters(N, self.degree)
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=True, interaction_only=False)
        self.Theta_ = np.zeros((1, self.p))
        self.P_ = [np.identity(self.p) * self.init_covariance]  # M x N x N

    def update_theta(self, X, y, A, R=None, model_pointers=None):
        if np.isnan(A).any():
            np.nan_to_num(A, copy=False)
            print("[WARNING]: Invalid value encountered in A")
        
        k = X.shape[0]
        
        if self.optimization == 'local':
            """
            local weighted least square update
            """
            model_pointers = range(self.M_) if not model_pointers else model_pointers
            for m in model_pointers:  # for model m
                Q_m = sp.sparse.spdiags(A[m, :], diags=0, m=k, n=k)
                X_reg = self.poly.fit_transform(X)  # k x p // vandermonde matrix // regression matrix
    
                if R is not None:
                    self.Theta_[m, :] = np.linalg.lstsq(Q_m @ R @ X_reg, y @ R @ Q_m, rcond=None)[0].flatten()
                else:
                    self.Theta_[m, :] = np.linalg.lstsq(Q_m @ X_reg, y @ Q_m, rcond=None)[0].flatten()
        elif self.optimization == 'global':
            V = self.poly.fit_transform(X)  # k x p // regression matrix
            X_reg = np.zeros((k, self.p * self.M_))
            j = 0
            for m in range(self.M_):
                for i in range(self.p):
                    X_reg[:, j] = V[:, i] * A[m, :]
                    j += 1
            
            self.Theta_ = np.linalg.lstsq(X_reg, y, rcond=None)[0].reshape(self.Theta_.shape)

    def update_theta_recursive(self, x, y, A):
        """
        local recursive weighted least square update
        """
        x_reg = self.poly.fit_transform(x).T
        for m in range(self.M_):
            if A[m, :] > self.activity_threshold:
                gamma = self.P_[m] @ x_reg / (x_reg.T @ self.P_[m] @ x_reg + self.forgetting_factor * np.reciprocal(A[m, :]))
                self.Theta_[m, :] = self.Theta_[m, :] + gamma @ (y - (x_reg.T @ self.Theta_[m, :]))
                self.P_[m] = 1/self.forgetting_factor * (self.P_[m] - gamma @ x_reg.T @ self.P_[m])
    
    def increase_number_of_models(self):
        self.Theta_ = np.vstack((self.Theta_, np.zeros((1, self.p))))
        self.P_.append(np.identity(self.p) * self.init_covariance)
        self.M_ += 1

    def decrease_number_of_models(self):
        self.Theta_ = self.Theta_[0:self.M_, :]
        del self.P_[-1]
        self.M_ -= 1

    def get_output(self, u):
        return self.Theta_ @ self.poly.fit_transform(u).T


class BaseNetworkStructure(BaseEstimator):
    """
    Abstract local model base class.

    Note: this is the parent class of all network structures
    """

    def __init__(self, random_state=None):
        self.random_state = check_random_state(random_state)

        # --- Attributes which has to be set before training ---
        self.local_models = None

        # --- Attributes set during training ---
        # center coordinates
        self.C_ = None  # N X M

        self.model_range = []

        # standard deviations of the validity function
        self.Sigma_ = None

        # weighting factors from validity function
        self.A_ = None  # M x k

        # diagonal data density weighting matrix
        self.R_ = None
        
        # Number of input dimensions
        self.N_ = None
        
        # Number of (batch) samples
        self.k = None

    # --- properties ---

    @property
    def A(self):
        return self.A_
    
    @property
    def C(self):
        return self.C_

    @property
    def M(self):
        return self.local_models.M

    # --- private functions ---

    def _get_local_loss(self, X, y):
        return self.A @ (y - self.get_output(X)) ** 2  # Loss function output -> M x _

    def _get_model_volumes(self, idx=-1):
        raise NotImplementedError

    def _increase_model_complexity(self):
        raise NotImplementedError

    def _decrease_model_complexity(self):
        raise NotImplementedError

    # --- public functions ---
    def initiate(self, local_models, **kwargs):
        self.local_models = local_models
        for key, value in kwargs.items():
            setattr(self, key, value)

    def check_is_fitted(self):
        check_is_fitted(self, ['C_', 'Sigma_'])

    def check_X_y(self, X, y):
        condition = np.zeros_like(X)
        for N, bound in enumerate(self.get_network_bounds()):
            condition[:, N] = (X[:, N] > (bound[0] - bound[0])) & (X[:, N] < (bound[1] + bound[1]))

        mask = np.multiply.reduce(condition, axis=1).astype(bool)
        return X[mask, :], y[mask]
        
    def get_network_bounds(self):
        bounds = [[np.inf, -np.inf] for dimension in range(self.N_)]
        for lm_range in self.model_range:
            for n, dimension_range in enumerate(lm_range):
                bounds[n][0] = dimension_range[0] if dimension_range[0] < bounds[n][0] else bounds[n][0]
                bounds[n][1] = dimension_range[1] if dimension_range[1] > bounds[n][1] else bounds[n][1]
        return bounds

    def merging(self):
        pass

    def get_global_loss(self, X, y):
        return np.sum((y - self.get_output(X)) ** 2)

    def validity_function(self, X, A=None):
        A = self.A_ if A is None else A
        exponent = np.zeros((self.M, X.shape[0]))  # M x k
        for m in range(self.M):
            np.sum((X - self.C_.T[m, :]) ** 2 / (2 * (self.Sigma_.T[m, :] ** 2)), out=exponent[m, :], axis=1)
        mu = ne.evaluate('exp(-1.0 * exponent)')
        mu_sum = np.sum(mu, axis=0)  # summation along M-axis -> k
        np.divide(mu, mu_sum, out=A)

    def fit(self, X, y, X_val, y_val, input_range):
        return self.batch_learning(self, X, y, X_val, y_val, input_range)
    
    def batch_learning(self, X, y, X_val, y_val, input_range):
        raise NotImplementedError

    def online_learning(self, x, y, X_total, y_total, **kwargs):
        raise NotImplementedError

    def get_output(self, u, A=None):
        A = self.A_ if A is None else A
        return np.sum(self.local_models.get_output(u) * A, axis=0)

    @staticmethod
    def get(identifier):
        try:
            if isinstance(identifier, BaseNetworkStructure):
                return identifier
            elif isinstance(identifier, str):
                return eval(identifier.lower().capitalize())()
            else:
                raise ValueError
        except (NameError, ValueError):
            raise ValueError("Could not interpret network structure identifier: " + str(identifier))


class Lolimot(BaseNetworkStructure):
    """
    A heuristic and incremental tree-construction algorithm that partitions the input space by axis-orthogonal splits.
    """
    def __init__(self, smoothness=0.33, smoothing='proportional', sigma=0.4, sigma_limit=10e-18,
                 refinement='loser', output_constrains=None):

        super().__init__()
        self.smoothness = smoothness
        self.smoothing = smoothing
        self.sigma = sigma
        self.sigma_limit = sigma_limit
        self.refinement = refinement

        self.output_constrains = output_constrains

        self.split_duration = []
        self.local_loss = None
        # number of online samples per model
        self.M_counter = np.zeros(1)

        self.cache = Cache('model_range_prev', 'C_prev', 'Sigma_prev', 'Theta_prev')

    # --- private functions ---

    def _get_model_volumes(self, idx=-1):
        volumes = []
        if idx == -1:
            for r in self.model_range:
                volumes.append(np.prod([np.abs(np.subtract(*r_n)) for r_n in r]))
            return volumes
        else:
            return np.prod([np.abs(np.subtract(*r_n)) for r_n in self.model_range[idx]])
    
    def _increase_model_complexity(self):
        self.local_models.increase_number_of_models()
        self.A_ = np.vstack((self.A_, np.zeros((1, self.k))))
        self.C_ = np.hstack((self.C_, np.zeros((self.N_, 1))))
        self.Sigma_ = np.hstack((self.Sigma_, np.zeros((self.N_, 1))))
        self.model_range.append([() for _ in range(self.N_)])
        self.M_counter = np.vstack((self.M_counter, np.zeros(1)))
        
    def _decrease_model_complexity(self):
        self.local_models.decrease_number_of_models()
        self.A_ = self.A_[0:self.M, :]
        self.C_ = self.C_[:, 0:self.M]
        self.Sigma_ = self.Sigma_[:, 0:self.M]
        self.model_range.pop()
        self.M_counter = self.M_counter[0:self.M]

    def _get_sigma(self, ranges):
        if self.smoothing == 'const':
            return self.sigma
        elif self.smoothing == 'proportional':
            new_sigmas = np.array(list(map(lambda r: np.abs(np.subtract(*r)) * self.smoothness, ranges)))
            new_sigmas[new_sigmas < self.sigma_limit] = self.sigma_limit
            return new_sigmas.tolist()

        else:
            raise ValueError(f"Inadmissible smoothing parameter: '{self.smoothing}'")

    def _save_params(self):
        self.cache.model_range_prev = deepcopy(self.model_range)
        self.cache.C_prev = np.copy(self.C_)
        self.cache.Sigma_prev = np.copy(self.Sigma_)
        self.cache.Theta_prev = np.copy(self.local_models.Theta_)

    def _recover_params(self):
        self.model_range = deepcopy(self.cache.model_range_prev)
        self.C_ = np.copy(self.cache.C_prev)
        self.Sigma_ = np.copy(self.cache.Sigma_prev)
        self.local_models.Theta_ = np.copy(self.cache.Theta_prev)

    def _get_model_idx_for_refinement(self, X, y, X_val):
        self.local_loss = self._get_local_loss(X, y)
        if self.refinement == 'loser' and self.output_constrains is None:  # loser refinement
            return np.argmax(self.local_loss)

        elif self.refinement == 'limited':  # lower limit for model volume
            idxs = np.flip(np.argsort(self.local_loss), axis=0)
            i = 0
            while not self._get_model_volumes(idxs[i]) > 1e-10:
                i += 1
                continue
            return idxs[i]

        elif self.refinement == 'probability':  # probability approach
            if self.M > 1:
                scaler = MinMaxScaler(feature_range=(np.min(self.local_loss), np.max(self.local_loss)))
                volumes = self._get_model_volumes()
                p = self.local_loss + np.log10((1 - scaler.fit_transform([volumes]))[0])
            else:
                p = self.local_loss

            return self.random_state.choice(list(range(self.M)), p=p/np.sum(p))

        elif self.output_constrains is not None:  # constrained output
            A = np.zeros((self.M, X_val.shape[0]))
            self.validity_function(X, A)
            y = self.get_output(X_val, A)
            constrain_violation = A @ (np.logical_or(
                (y < self.output_constrains[0]), (y > self.output_constrains[1])))
            return np.argmax(constrain_violation + np.multiply(self.local_loss, 0.01))
        else:
            raise NotImplementedError

    def _split_along(self, j, l, m, X, y):
        # (a) ... split component model along j in two halves
        self.model_range[m] = deepcopy(self.model_range[l])
        r = self.model_range[l][j]
        ranges = [(np.min(r), np.mean(r)), (np.mean(r), np.max(r))]
        self.model_range[l][j], self.model_range[m][j] = ranges

        self.C_[:, m] = deepcopy(self.C_[:, l])
        self.C_[j, (l, m)] = list(map(lambda x: np.mean(x), ranges))

        self.Sigma_[:, m] = deepcopy(self.Sigma_[:, l])
        self.Sigma_[j, (l, m)] = self._get_sigma(ranges)

        # (b) ... calculate validity functions all models
        self.validity_function(X, self.A_)

        # (c) ... get models' parameter
        self.local_models.update_theta(X, y, self.A_, self.R_, (l, m))
    
    def _split_model(self, X, y, r):
        self._increase_model_complexity()
        m = self.M - 1  # 'm' denotes the most recent added model

        L_global = np.zeros(self.N_)  # global model loss for every split attempt
        self._save_params()

        # 3. for every input dimension ...
        for j in range(self.N_):
            self._split_along(j, r, m, X, y)

            # (d) ... calculate the tree's output error
            L_global[j] = self.get_global_loss(X, y)

            # Undo changes 'from _split_along'
            self._recover_params()

        # 4. find best division (split) and apply
        j = np.argmin(L_global)

        self._split_along(j, r, m, X, y)
    
    # --- public functions ---

    def batch_learning(self, X, y, X_val, y_val, input_range):
        start_time = time.time()

        # 1. Initialize global model
        if self.M == 0:
         
            self.k, self.N_ = X.shape
            self.local_models.initiate(self.N_)
            self.C_ = np.zeros((self.N_, 1))
            self.Sigma_ = np.zeros((self.N_, 1))

            self.C_[:, 0] = [np.mean(r) for r in input_range]
            self.Sigma_[:, 0] = self._get_sigma(input_range)
            self.A_ = np.zeros((1, self.k))
            self.validity_function(X, self.A_)
            self.model_range.append(deepcopy(input_range))
            self.local_models.update_theta(X, y, self.A_, self.R_)
            self.local_loss = self._get_local_loss(X, y)
            return True

        # 2. Find worst LLM
        r = self._get_model_idx_for_refinement(X, y, X_val)  # model 'r' is considered for further refinement
        try:
            self._split_model(X, y, r)
            self.split_duration.append(time.time() - start_time)
            return True

        except np.linalg.LinAlgError:
            print(f"[WARNING]: Training was aborted because of singular matrix with M={self.M}")
            return False
        
        except FloatingPointError:
            print(f"[WARNING]: Training was aborted because Sigma values a too small. M={self.M}")
            return False

    def online_learning(self, x, y, X_total, y_total, recursive=False, **kwargs):
        A = np.zeros((self.M, 1))
        self.validity_function(x, A)
        model_idx = np.argmax(A)

        if recursive:
            self.local_models.update_theta_recursive(x, y, A)
        else:
            self.k = X_total.shape[0]
            self.A_ = np.zeros((self.M, self.k))
            self.validity_function(X_total, self.A_)
            self.local_models.update_theta(X_total, y_total, self.A_)

        local_loss = self._get_local_loss(x, y)[model_idx]
        max_loss = np.max(self.local_loss) if np.max(self.local_loss) > 1e-8 else 1e-8

        if local_loss > max_loss:
            self.k = X_total.shape[0]
            self.A_ = np.zeros((self.M, self.k))

            self.validity_function(X_total, self.A_)
            number_of_samples = np.sum(self.A_[model_idx, :])
            if number_of_samples < 1.0:
                # self.local_loss = self._get_local_loss(X_total, y)
                return

            self._split_model(X_total, y_total, model_idx)

    def finalize(self, X, y):
        self.local_models.update_theta(X, y, self.A_)
        raise DeprecationWarning


class Grid(BaseNetworkStructure):
    def __init__(self, smoothness=0.33, max_models=10):
        super().__init__()
        self.smoothness = smoothness
        self.max_models = max_models

    # --- public functions ---
    def batch_learning(self, X, y, X_val, y_val, input_range):
        self.k, self.N_ = X.shape
        self.local_models.initiate(self.N_)
        for _ in range(self.max_models-1):
            self.local_models.increase_number_of_models()
        self.A_ = np.zeros((self.max_models, self.k))
        C = np.meshgrid(*[np.linspace(start, stop, int(np.power(self.max_models, 1/self.N_).round())) for start, stop in input_range])
        self.C_ = np.reshape(*C, (self.N_, self.max_models))
        delta_c = np.abs(self.C_[0, 0] - self.C_[0, 1])

        self.model_range = [ tuple(zip((self.C_[:, i] - delta_c/2).tolist(), (self.C_[:, i] + delta_c/2).tolist())) for i in range(self.max_models)]

        self.Sigma_ = np.ones((self.N_, self.max_models)) * self.smoothness * delta_c
        self.validity_function(X, self.A_)
        self.local_models.update_theta(X, y, self.A_)
        return True


class LMNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, network='lolimot', local_models='linear', model_complexity=10, training_tol=1e-8, plotter=None,
                 early_stopping=False, early_stopping_tol=0.075, kde_bandwidth=0, validation_size=None, notebook=True,
                 random_state=None, verbosity=0, **kwargs):

        self.random_state = check_random_state(random_state)

        self.local_models = BaseLocalModels.get(local_models)
        self.network = BaseNetworkStructure.get(network)

        for key, val in kwargs.items():
            try:
                self.local_models.set_params(**{key: val})
            except ValueError:
                try:
                    self.network.set_params(**{key: val})
                except ValueError:
                    continue

        self.model_complexity = model_complexity
        self.training_tol = training_tol
        self.early_stopping = early_stopping
        self.early_stopping_tol = early_stopping_tol
        self.kde_bandwidth = kde_bandwidth
        self.validation_size = validation_size

        self.kde = None
        self.validation_loss = []
        self.global_loss = []
        self.training_duration = 0

        self.verbosity = verbosity
        self.notebook = notebook
        self.plotter = plotter
    
    @property
    def M(self):
        return self.network.M
    
    @property
    def C(self):
        return self.network.C

    @property
    def N(self):
        return self.network.N_
    
    @property
    def bounds(self):
        return self.network.get_network_bounds()
    
    def _get_global_loss(self, X, y):
        return self.network.get_global_loss(X, y)
    
    def _get_validation_loss(self, X, y):
        return np.sum((y - self.predict(X)) ** 2)

    def _stopping_condition_met(self):
        if self.network.M == 0:
            return False
        
        # determine different stopping conditions
        complexity_reached = self.network.M >= self.model_complexity
        tolerance_reached = self.training_tol > self.global_loss[-1]
        if self.early_stopping and self.validation_loss[-4:-1]:
            current_mean_val_loss = np.mean(self.validation_loss[-4:-1])
            overfitting = ((self.validation_loss[-1] - current_mean_val_loss) >
                           self.early_stopping_tol * current_mean_val_loss) or\
                          ((self.validation_loss[-1] - np.min(self.validation_loss)) >
                           self.early_stopping_tol * np.min(self.validation_loss) * 1.5)
        else:
            # there are not enough information yet to determine overfitting
            overfitting = False

        # in case of the fullfilment of a stopping condition print the following message
        if complexity_reached and self.verbosity:
            print(f"[INFO]: Training finished with a maximal model complexity:={self.model_complexity}.")
        elif tolerance_reached and self.verbosity:
            print(f"[INFO]: Training finished because global loss is smaller than tol:={self.training_tol}.")
        elif (overfitting and self.early_stopping) and self.verbosity:
            print(f"[INFO]: Early stopping of the training.")
    
        return complexity_reached or tolerance_reached or overfitting
    
    def _make_validation_split(self, X, y):
        X_val, y_val = [None, None]
        if self.validation_size is not None and not np.isclose(self.validation_size, (0.0,)):
            if isinstance(self.validation_size, float):
                # proportion of the dataset to include in the validation split
                train_size = 1.0 - self.validation_size
            else:
                # absolute number of validation samples
                train_size = int(X.shape[0] - self.validation_size)

            X, X_val, y, y_val = train_test_split(X, y, train_size=train_size,
                                                  test_size=self.validation_size,
                                                  random_state=self.random_state,
                                                  shuffle=True)
        return X, X_val, y, y_val
    
    # --- public functions ---
    
    def get_validation_split(self, X, y):
        return self._make_validation_split(X, y)
    
    def fit(self, X, y, input_range=None, output_constrains=None, additional_validation_set=None, store_data=True):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples,)
            Target values

        input_range : list of tuples, optional
            Range of values for each input dimension N for which the model should be trained.
            If not passed, it will be determined when fitting to data.

        output_constrains: tuple, optional
            A tuple with the lower and the upper bound of the output. Is involved when splitting models.

        additional_validation_set: list, optional
            Data set to estimate the generalization ability of the model during training.
            Should be like [X_val, y_val].
        
        store_data: bool, default: True
            If True training data will be stored after finishing training (for subsequent online training)

        Returns
        -------
        self : returns an instance of self.
        """
        # --- resetting log from former trainings ---
        for attr in [self.global_loss, self.validation_loss]:
            attr.clear()

        # --- input checks ---
        X, y = check_X_y(X, y, y_numeric=True)

        if (self.early_stopping_tol != 0.075 and not self.early_stopping) and self.verbosity:
            print("[WARNING]: A tolerance for early stopping was set, but early stopping is disabled!")

        N = X.shape[1]
        if input_range:
            assert N == len(input_range), \
                f"Dimension N from 'input_range' and 'X' does not agree: {N} neq {len(input_range)}"

        # --- validation split ---
        X, X_val, y, y_val = self._make_validation_split(X, y)

        if additional_validation_set and X_val is not None:
            X_val_add, y_val_add = check_X_y(*additional_validation_set)
            X_val = np.vstack((X_val, X_val_add))
            y_val = np.vstack((y_val, y_val_add))
        elif additional_validation_set:
            X_val, y_val = check_X_y(*additional_validation_set)

        # --- tracking training duration
        self.training_duration = 0
        start_time = time.time()  # start tracking the training duration

        # --- initialising model parameter ---
        if not input_range:
            input_range = []
            for j in range(N):
                input_range.append((X[:, j].min(), X[:, j].max()))
        else:
            input_range = input_range

        if not np.isclose(self.kde_bandwidth, (0.0,)):
            # --- estimating the pdf of the data ---
            self.kde_ = KernelDensity(bandwidth=self.kde_bandwidth, kernel='gaussian').fit(X)
            rep_dens = np.reciprocal(np.exp(self.kde_.score_samples(X)))
            R = sp.sparse.spdiags(rep_dens / np.max(rep_dens), diags=0, m=X.shape[0], n=X.shape[0])
        else:
            R = None

        self.network.initiate(deepcopy(self.local_models), output_constrains=output_constrains,
                              R_=R, max_models=self.model_complexity)

        # --- model fitting ---

        tqdm.tqdm.monitor_interval = 0  # disable the monitor thread because bug in tqdm #481
        if self.notebook:
            pbar = tqdm.tqdm_notebook(total=self.model_complexity, disable=not(bool(self.verbosity)))
        else:
            pbar = tqdm.tqdm(total=self.model_complexity, disable=not(bool(self.verbosity)))

        while not self._stopping_condition_met():
                sucess_flag = self.network.batch_learning(X, y, X_val, y_val, input_range)
                pbar.update(1)
                
                # --- training logging ---
                self.global_loss.append(self._get_global_loss(X, y))
                if X_val is not None:
                    self.validation_loss.append(self._get_validation_loss(X_val, y_val))

                if self.plotter:
                    plot_data = {}
                    plot_data.update({'training_loss': self.global_loss[-1]})

                    if X_val is not None:
                        plot_data.update({'validation_loss': self.validation_loss[-1]})

                    self.plotter.update(plot_data)
                
                # --- stop training if network encounters an error ---
                if not sucess_flag:
                    break

        pbar.close()
        # ----------------------

        self.training_duration = time.time() - start_time
        # print(f"[INFO] Finished model training after {time.time() - start_time:.4f} seconds.")

        if store_data:
            # --- for online training ---
            self.store_data = True
            self.X, self.y = X, y

        return self
    
    def online_training(self, x, y, **kwargs):
        if not self.store_data:
            raise Exception("Call fit with parameter 'store_data=True' before online training.")
            
        x, y = check_X_y(x, y, y_numeric=True)

        self.X = np.vstack((self.X, x))
        self.y = np.hstack((self.y, y))

        while self.X.shape[0] > kwargs.get('max_size', 100) and "F" in kwargs:
            idx = np.argmax(kwargs["F"])
            self.X = np.delete(self.X, idx, axis=0)
            self.y = np.delete(self.y, idx)

        self.network.online_learning(x=x, y=y, X_total=self.X, y_total=self.y, **kwargs)
        return True

    def update_local_loss(self):
        self.network.local_loss = self.network._get_local_loss(self.X, self.y)

    def predict(self, X, return_A=False):
        self.network.check_is_fitted()

        # Input validation
        X = check_array(X)
        A = np.zeros((self.network.M, X.shape[0]))
        self.network.validity_function(X, A)
        if not return_A:
            return self.network.get_output(X, A)
        else:
            return self.network.get_output(X, A), A

    def save(self, filename='LMN.pkl'):
        del self.plotter
        with open(filename, 'wb') as output:  # overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename='LMN.pkl'):
        with open(filename, 'rb') as input_file:
            return pickle.load(input_file)