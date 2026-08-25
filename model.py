"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np

def impute_nan_with_mean(X):
    # Create a copy so we don't modify the original array
    X = X.copy()

    # Compute the mean of each column while ignoring NaN values
    col_mean = np.nanmean(X, axis=0)

    # If a column is entirely NaN, its mean will be NaN.
    # Replace those NaN means with 0.0
    col_mean[np.isnan(col_mean)] = 0.0

    # Create a boolean mask indicating where NaN values are located
    mask = np.isnan(X)

    # Get the row and column indices of every NaN value
    rows, cols = np.where(mask)

    # Replace each NaN with the mean of its corresponding column
    X[rows, cols] = col_mean[cols]

    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1 
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.clip(X ,lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator / (denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    # X.shape = (N, F)
    # col.shape = (N,)
    # result.shape = (N, F+1)
    col_2d = col.reshape(-1, 1)
    return np.concatenate([X, col_2d], axis=1)

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    #  N is the number of samples and C is the number of unique categories
    labels = np.array(labels)
    unique_ = np.unique(labels)
    output = (labels[:, None] == unique_[None, :]).astype(float)
    return output

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    bias = np.ones((X.shape[0], 1))
    return np.hstack([bias, X])

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    rng = np.random.default_rng(seed)
    return rng.permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n = len(indices) 
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = indices[0 : n_train]
    val = indices[n_train : n_train + n_val]
    test = indices[n_train + n_val :]
    return train , val , test

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X_sub = X[indices]
    y_sub = y[indices]
    return X_sub, y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    # X.shape =  (N, D) 
    # y.shape = (N,)
    # theta.shape = (D,)
    A = X.T @ X           # A.shape = (D,D)
    b = X.T @ y           # b.shape = (D,)
    theta = np.linalg.solve(A, b)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    # X.shape = (N,D)
    # theta.shape = (D,)
    # pred.shape = (N,)
    y_hat = X @ theta
    return y_hat

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return rmse

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    SS_res = np.sum((y_true - y_pred)**2)
    SS_tot = np.sum((y_true - y_true.mean())**2)
    if SS_tot == 0 :
        return 0.0 
    else :
        return 1.0 - (SS_res / SS_tot)

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    r = y_true - y_pred
    mean = float(np.mean(r))
    std = float(np.std(r))
    median_abs = float(np.median(np.abs(r)))
    return {'mean':mean ,'std':std ,'median_abs':median_abs}

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    # X.shape = (N,F)
    X_imp = impute_nan_with_mean(X)
    lower , upper = compute_iqr_bounds(X_imp, k=iqr_k)
    result = clip_columns(X_imp, lower, upper)
    return result

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]
    X = append_column(X_num, make_ratio_feature(numerator, denominator))
    if cat_labels is not None:
        X = np.hstack([X, one_hot_encode(cat_labels)])
    return X

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

