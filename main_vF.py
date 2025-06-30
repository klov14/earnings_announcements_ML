#%% Packages
import pandas as pd
import os 
import random
from typing import List

import optuna
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
#%% Settings

SEED = 187
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#%% Functions

def add_technical_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add key technical indicators, plus high–low and open–close ranges,
    then drop the raw high, low, and open columns (keep only close).
    """
    df = prices.sort_values(['stock_id', 'date']).copy()

    # 1) new range features
    df['high_low']   = df['high'] - df['low']
    df['open_close'] = df['close'] - df['open']

    # 2) drop raw columns we no longer need
    df = df.drop(columns=['high', 'low', 'open'])

    # 3) group for transforms
    grp = df.groupby('stock_id')

    # 4) Returns & momentum
    df['daily_ret']     = grp['close'].pct_change(fill_method=None)
    df['roc_10']        = grp['close'].transform(lambda x: x.pct_change(10, fill_method=None))
    df['momentum_10']   = grp['close'].transform(lambda x: x.diff(10))

    # 5) Moving averages
    df['sma_10'] = grp['close'].transform(lambda x: x.rolling(10).mean())
    df['ema_10'] = grp['close'].transform(lambda x: x.ewm(span=10, adjust=False).mean())

    # 6) ATR(14) using high_low
    df['atr_14'] = grp['high_low'].transform(lambda x: x.rolling(14).mean())

    # 7) Bollinger Band width on close
    std20 = grp['close'].transform(lambda x: x.rolling(20).std())
    sma20 = grp['close'].transform(lambda x: x.rolling(20).mean())
    df['bb_width'] = (2 * std20) / sma20

    # 8) On-Balance Volume
    signed_vol = np.sign(df['daily_ret'].fillna(0)) * df['volume']
    df['obv']    = grp['volume'].transform(lambda x: signed_vol.loc[x.index].cumsum())

    # 9) RSI
    delta     = df['close'].diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    avg_gain  = grp['close'].transform(lambda x: gain.loc[x.index].rolling(14).mean())
    avg_loss  = grp['close'].transform(lambda x: loss.loc[x.index].rolling(14).mean())
    rs        = avg_gain / (avg_loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # 10) MACD
    ema12 = grp['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = grp['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['macd']        = ema12 - ema26
    df['macd_signal'] = grp['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['macd_hist']   = df['macd'] - df['macd_signal']

    return df

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

def _macd_hist(series: pd.Series,
               fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd - macd.ewm(span=signal, adjust=False).mean()

def _price_factors(prices: pd.DataFrame) -> pd.DataFrame:
    pr = prices.sort_values(["stock_id", "date"]).copy()
    g  = pr.groupby("stock_id", group_keys=False)

    # momentum & volatility
    pr["lr_1d"]   = g["close"].transform(lambda s: np.log(s / s.shift(1)))
    pr["lr_5d"]   = g["close"].transform(lambda s: np.log(s / s.shift(5)))
    pr["lr_21d"]  = g["close"].transform(lambda s: np.log(s / s.shift(21)))
    pr["vol_21d"] = g["lr_1d"].transform(lambda s: s.rolling(21).std())

    # RSI and MACD histogram
    pr["rsi14"]  = g["close"].transform(_rsi)
    pr["macd_h"] = g["close"].transform(_macd_hist)

    # Bollinger-band width
    mean20 = g["close"].transform(lambda s: s.rolling(20).mean())
    std20  = g["close"].transform(lambda s: s.rolling(20).std())
    pr["bb_width"] = (mean20 + 2*std20 - (mean20 - 2*std20)) / mean20

    # ATR14 
    pr["atr14"] = g.apply(
        lambda df: (df["high"] - df["low"]).rolling(14).mean()
    ).reset_index(level=0, drop=True)

    # shift all factor columns by +1 day per stock to avoid look-ahead
    fac_cols = ["lr_1d", "lr_5d", "lr_21d",
                "vol_21d", "rsi14", "macd_h",
                "bb_width", "atr14"]
    pr[fac_cols] = pr.groupby("stock_id", group_keys=False)[fac_cols].shift(1)

    return pr[["stock_id", "date"] + fac_cols]

def _cs_zscore(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """Cross-sectional z-score each trading day """
    out = df.copy()
    for c in factor_cols:
        out[c + "_z"] = (out[c] - out.groupby("date")[c].transform("mean")) \
                        /  out.groupby("date")[c].transform("std")
    return out.drop(columns=factor_cols)


def engineer_x(X_train: pd.DataFrame,
               all_prices: pd.DataFrame,
               *,
               id_col: str = "stock_id",
               date_col: str = "date",
               cross_sectional: bool = True) -> pd.DataFrame:
    """
    Returns an enhanced X whose row order/length exactly matches X_train.

    Parameters
    ----------
    X_train        : DataFrame already parsed (columns separated by ';')
    all_prices     : Daily OHLCV + market_cap, already parsed
    id_col / date_col : column names for the merge keys
    cross_sectional   : if True, z-score each factor per date
    """
    # basic hygiene
    x = X_train.copy()
    pr = all_prices.copy()

    # guarantee correct types
    x[id_col] = x[id_col].astype(int)
    x[date_col] = pd.to_datetime(x[date_col])

    pr[id_col] = pr[id_col].astype(int)
    pr[date_col] = pd.to_datetime(pr[date_col])
    numeric = ["open", "high", "low", "close", "volume", "market_cap"]
    pr[numeric] = pr[numeric].apply(pd.to_numeric, errors="coerce")

    # build price-based factors 
    price_fac = _price_factors(pr)
    fac_cols  = [c for c in price_fac.columns if c not in (id_col, date_col)]

    if cross_sectional:
        price_fac = _cs_zscore(price_fac, fac_cols)

    # merge onto announcement rows 
    X_enh = x.merge(price_fac, on=[id_col, date_col], how="left")

    # simple missing-value median-policy 
    num_cols = X_enh.select_dtypes("number").columns
    for col in num_cols:
        X_enh[col] = (X_enh.groupby(id_col)[col].ffill()
                                  .fillna(X_enh[col].median()))

    return X_enh

def encode_string_categories(
    df: pd.DataFrame,
    cols: List[str]
) -> pd.DataFrame:
    """
    Convert specified string columns in df to numeric category codes starting at 1.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : List[str]
        List of column names to encode.
    
    Returns
    -------
    pd.DataFrame
        Copy of df with specified columns encoded as integers 1..n_categories.
    """
    df_encoded = df.copy()
    for col in cols:
        df_encoded[col] = (
            df_encoded[col]
            .astype('category')
            .cat.codes
            .add(1)  
        )
    return df_encoded

def drop_invalid_samples(X, y):
    """
    Remove samples from X and y where X contains NaN or inf.
    Supports pandas DataFrame (2D) or numpy arrays of shape (N, M) or (N, T, D).
    """
    # Build mask of valid rows
    if isinstance(X, pd.DataFrame):
        bad = X.isna().any(axis=1) | X.isin([np.inf, -np.inf]).any(axis=1)
        mask = (~bad).to_numpy()
        X_clean = X.iloc[mask].reset_index(drop=True)
    else:
        # numpy array: collapse all but the first axis
        if X.ndim == 2:
            mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        elif X.ndim == 3:
            mask = ~(np.isnan(X) | np.isinf(X)).any(axis=(1, 2))
        else:
            raise ValueError(f"Unsupported array dimension: X.ndim={X.ndim}")
        X_clean = X[mask]

    # Apply same mask to y
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y_clean = y.iloc[mask].reset_index(drop=True)
    else:
        y_clean = y[mask]

    return X_clean, y_clean

def drop_invalid_sample_X(X):
    """
    Remove samples from X and y where X contains NaN or inf.
    Supports pandas DataFrame (2D) or numpy arrays of shape (N, M) or (N, T, D).
    """
    # Build mask of valid rows
    if isinstance(X, pd.DataFrame):
        bad = X.isna().any(axis=1) | X.isin([np.inf, -np.inf]).any(axis=1)
        mask = (~bad).to_numpy()
        X_clean = X.iloc[mask].reset_index(drop=True)
    else:
        # collapse all but the first axis
        if X.ndim == 2:
            mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        elif X.ndim == 3:
            mask = ~(np.isnan(X) | np.isinf(X)).any(axis=(1, 2))
        else:
            raise ValueError(f"Unsupported array dimension: X.ndim={X.ndim}")
        X_clean = X[mask]

    return X_clean

def split_train_oos(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float,
    date_col : str = 'date',
):
    """
    Sort X (and y) by a date column, then split into in-sample and out-of-sample.

    Args:
        X             (pd.DataFrame): feature DataFrame including `date_col`.
        y             (pd.Series):   target aligned with X (same index).
        date_col      (str):         name of the datetime column in X.
        test_fraction (float):       fraction of samples to reserve as most recent OOS.

    Returns:
        X_train (pd.DataFrame): in-sample features (date_col dropped)
        y_train (pd.Series):    in-sample targets
        X_oos   (pd.DataFrame): out-of-sample features (date_col dropped)
        y_oos   (pd.Series):    out-of-sample targets
    """
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be between 0 and 1")

    # Ensure date column is datetime
    X = X.copy()
    X[date_col] = pd.to_datetime(X[date_col])

    # Sort by date ascending
    X_sorted = X.sort_values(by=date_col)
    y_sorted = y.loc[X_sorted.index]

    N = len(X_sorted)
    n_oos = int(np.floor(N * test_fraction))
    if n_oos == 0 or n_oos == N:
        raise ValueError("test_fraction results in empty train or oos set")

    split_idx = N - n_oos
    # Split features and drop date column
    X_train = X_sorted.iloc[:split_idx].drop(columns=[date_col])
    X_oos   = X_sorted.iloc[split_idx:].drop(columns=[date_col])

    # Split targets
    y_train = y_sorted.iloc[:split_idx]
    y_oos   = y_sorted.iloc[split_idx:]

    return X_train, y_train, X_oos, y_oos

def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.drop(columns=cols, errors="ignore")


def plot_predicted_vs_true(y_true, y_pred, title="Predicted vs. True Values"):
    """
    Plot predicted vs. true values for regression results.

    Args:
        y_true (array-like): Ground truth target values
        y_pred (array-like): Predicted values from the model
        title   (str)      : Plot title (optional)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)],
             [min(y_true), max(y_true)],
             color='red', linestyle='--', label='Ideal prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
def build_price_histories(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    lookback: int = 29
):
    """
    For each event in `events` (stock_id, date), build a lookback-window
    of the last `lookback` trading‐day rows from `prices`.

    Args:
        events   (pd.DataFrame): must contain columns ["stock_id","date"]
        prices   (pd.DataFrame): must contain ["stock_id","date"] plus feature cols
        lookback (int): number of prior rows to include (T-1 ... T-lookback)

    Returns:
        panel      (np.ndarray): shape (N_events, lookback, D_features)
        feat_cols  (List[str]) : the feature column names
        events_idx (pd.Index)   : original index of `events`
    """
    # 1) prep
    events = events.copy()
    events['date'] = pd.to_datetime(events['date'])
    prices = prices.copy()
    prices['date'] = pd.to_datetime(prices['date'])

    # feature columns
    feat_cols = [c for c in prices.columns 
                 if c not in ('stock_id','date')]
    D = len(feat_cols)
    N = len(events)

    # group prices by stock once
    grp = prices.sort_values('date').groupby('stock_id')

    # empty panel
    panel = np.full((N, lookback, D), np.nan, dtype=float)

    # 2) iterate events
    for i, (sid, ev_date) in enumerate(zip(events['stock_id'], events['date'])):
        # get all prior rows for this stock
        try:
            hist = grp.get_group(sid)
        except KeyError:
            continue  # no data for this stock_id
        hist = hist.loc[hist['date'] < ev_date].sort_values('date')
        # take last `lookback` rows
        vals = hist[feat_cols].values[-lookback:]
        L = vals.shape[0]
        if L > 0:
            panel[i, lookback - L : lookback, :] = vals

    return panel, feat_cols, events.index


def perform_pca(df: pd.DataFrame, explained_variance: float) -> pd.DataFrame:
    """
    Perform PCA on `df`, retaining enough components to explain
    at least `explained_variance` of total variance.

    Args:
        df                (pd.DataFrame): Input data (numeric columns).
        explained_variance (float): Fraction between 0 and 1 of variance to retain.

    Returns:
        pd.DataFrame: Transformed data with PCA components named 'PC1', 'PC2', …
    """
    # 1. Scale features to zero mean, unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    # 2. Fit PCA to desired variance threshold
    pca = PCA(n_components=explained_variance, svd_solver='full')
    X_pca = pca.fit_transform(X_scaled)

    # 3. Build DataFrame of principal components
    cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, index=df.index, columns=cols)

    return df_pca

#%% Load data

all_data = pd.read_parquet('all_data.parquet', engine='fastparquet')
all_prices = pd.read_parquet('all_prices.parquet', engine='fastparquet')
X = pd.read_parquet('X_train.parquet', engine='fastparquet')
y = pd.read_parquet('y_train.parquet', engine='fastparquet')
DummyX_test = pd.read_parquet('DummyX_test.parquet', engine='fastparquet')

# Ensure correct formatting
X['date'] = pd.to_datetime(X['date'])
y['date'] = pd.to_datetime(y['date'])
DummyX_test['date'] = pd.to_datetime(DummyX_test['date'])

# Initialise for evaluation
evaluation_results = []

LOOKBACK_WINDOW = 30

#%% Feature engineering

X = engineer_x(X, all_prices)
DummyX_test = engineer_x(DummyX_test, all_prices)

# Transform text categories to numbers
X = X.drop(columns=['gics_sector','gics_group', 'gics_subindustry'])
DummyX_test = DummyX_test.drop(columns=['gics_sector','gics_group', 'gics_subindustry'])

# Quantify industry codes
X = encode_string_categories(X, ['gics_industry'])
DummyX_test = encode_string_categories(DummyX_test, ['gics_industry'])

# Add technical indicators to prices
all_prices = add_technical_indicators(all_prices)

# Build 3D look-back window
panel, features, idx = build_price_histories(X, all_prices, lookback=LOOKBACK_WINDOW)
panel_dummy, features_dummy, idx_dummy = build_price_histories(DummyX_test, all_prices, lookback=LOOKBACK_WINDOW)

# Flatten panel 
panel_2d = panel.reshape(panel.shape[0], -1)  
panel_dummy_2d = panel_dummy.reshape(panel_dummy.shape[0], -1)

# Combine with X 
X_flat = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
panel_df = pd.DataFrame(panel_2d, index=X_flat.index)  

DummyX_test_flat = pd.DataFrame(DummyX_test) if not isinstance(DummyX_test, pd.DataFrame) else DummyX_test
panel_df_dummy = pd.DataFrame(panel_dummy_2d, index=DummyX_test_flat.index) 

# Concatenate along columns
X_combined = pd.concat([X_flat, panel_df], axis=1)
X_dummy_combined = pd.concat([DummyX_test_flat, panel_df_dummy], axis=1)

#%% Finalise training data

# Drop all rows with NaNs
X, y = drop_invalid_samples(X_combined, y)
X_dummy = drop_invalid_sample_X(X_dummy_combined)

# Ensure column names are strings
X.columns = X.columns.map(str)
y.columns = y.columns.map(str)
X_dummy.columns = X_dummy.columns.map(str)

# Separate non-numeric columns
X_non_numeric = X.select_dtypes(exclude=["number"])
X_numeric = X.select_dtypes(include=["number"])
X_dummy_non_numeric = X_dummy.select_dtypes(exclude=["number"])
X_dummy_numeric = X_dummy.select_dtypes(include=["number"])


# Perform PCA on numeric part
# 1. Initialize the PCA object
pca = PCA(n_components=0.97)

# 2. Fit PCA on the TRAINING data and transform it 
X_pca_values = pca.fit_transform(X_numeric)
X_pca = pd.DataFrame(X_pca_values)

# 3. Use the same fitted PCA to transform the dummy data 
X_dummy_pca_values = pca.transform(X_dummy_numeric)
X_dummy_pca = pd.DataFrame(X_dummy_pca_values)


# Recombine PCA output with non-numeric columns 
X = pd.concat([X_non_numeric.reset_index(drop=True), X_pca.reset_index(drop=True)], axis=1)
X_dummy = pd.concat([X_dummy_non_numeric.reset_index(drop=True), X_dummy_pca.reset_index(drop=True)], axis=1)

# Keep entire dataset for the training of the best and final model at the end
X_full = X
y_full = y 

# Create OOS and training dataset
X, y, X_oos, y_oos = split_train_oos(X, y, test_fraction=0.2)

#%% Linear Regression - Prepare

# Adjust X and y for usage in regression
def preprocess_linear_regression(X, y):
    X_regression = drop_columns(X, ["stock_id", "date", "MASK_EARNINGS", 'value_smart_eps'])
    y_regression = y['forward_return']
    return X_regression, y_regression
    
X_regression, y_regression = preprocess_linear_regression(X, y)

#%% Linear Regression - Train

def train_linear_regression(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Train a simple linear regression model.

    Args:
        X (np.ndarray): Features, shape (N, D)
        y (np.ndarray): Target, shape (N,)

    Returns:
        Pipeline: Fitted pipeline (scaler → linear regression)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LinearRegression())
    ])
    model.fit(X, y)
    return model

regression_model = train_linear_regression(X_regression.values, y_regression.values)

#%% Linear Regression - Evaluate

def evaluate_regression_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a regression model on given test data.

    Args:
        model      : trained regression model with .predict()
        X_test     : test features (2D array)
        y_test     : true target values
        model_name : name to identify the model

    Returns:
        dict with model_name, rmse, mae, r2, mape
    """

    y_pred = model.predict(X_test)
    
    plot_predicted_vs_true(y_test, y_pred)

    return {
        "model": model,
        "model_name": model_name,
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "mape": mean_absolute_percentage_error(y_test, y_pred) * 100,
        "prediction" : y_pred
    }

X_oos_regression, y_oos_regression = preprocess_linear_regression(X_oos, y_oos)

result_linear_regression = evaluate_regression_model(
    regression_model,
    X_oos_regression,
    y_oos_regression,
    "linear_regression"
)

evaluation_results.append(result_linear_regression)

#%% Random Forest - Prepare

def preprocess_rf(X, y):
    X = drop_columns(X, ["stock_id", "date", "MASK_EARNINGS", 'value_smart_eps'])
    y = y['forward_return']
    return X, y

X_rf, y_rf = preprocess_rf(X, y)

#%% Random Forest - Train

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": 2
    }
    model = RandomForestRegressor(**params)
    scores = cross_val_score(
        model,
        X_rf,
        y_rf,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    return np.mean(scores)

# create study and optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

# train final model with best hyperparameters
best_params = study.best_params
rf_model = RandomForestRegressor(
    **best_params,
    random_state=SEED,
    n_jobs=-1,
    verbose=1
)
rf_model.fit(X_rf, y_rf)

print("Best hyperparameters:", best_params)


#%% Random Forest - Evaluate

X_oos_rf, y_oos_rf = preprocess_rf(X_oos, y_oos)

result_rf = evaluate_regression_model(
    rf_model,
    X_oos_regression,
    y_oos_regression,
    "random_forest"
)

evaluation_results.append(result_rf)

#%% XGBoost - Prepare

def preprocess_xgboost(X, y):
    X = drop_columns(X, ["stock_id", "date", "MASK_EARNINGS", 'value_smart_eps'])
    y = y['forward_return']
    return X, y

X_xgb, y_xgb = preprocess_xgboost(X, y)

#%% XGBoost - Train

X_arr = np.asarray(X_xgb, dtype=float)
y_arr = np.asarray(y_xgb, dtype=float)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": SEED,
        "verbosity": 2,
        "n_jobs": -1
    }
    model = XGBRegressor(**params)
    cv = KFold(n_splits=3, shuffle=True, random_state=SEED)

    scores = cross_val_score(
        model,
        X_arr,
        y_arr,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    rmse = np.mean(np.sqrt(-scores))
    return rmse

study = optuna.create_study(direction="minimize")

study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)

# final model
best = study.best_params
xgb_model = XGBRegressor(**best, random_state=SEED, verbosity=2, n_jobs=-1)
xgb_model.fit(X_arr, y_arr, verbose=2)

#%% XGBoost - Evaluate

X_oos_xgb, y_oos_xgb = preprocess_xgboost(X_oos, y_oos)

result_xgboost = evaluate_regression_model(
    xgb_model,
    X_oos_regression,
    y_oos_regression,
    "XGBoost"
)

evaluation_results.append(result_xgboost)

#%% Neural Net - Execute

# preprocess
def preprocess_nn(X: pd.DataFrame, y: pd.DataFrame):
    """Drop non-numeric columns and keep the three targets."""
    X = X.drop(
        columns=["stock_id", "date", "MASK_EARNINGS", "value_smart_eps"],
        errors="ignore",
    )
    y = y[["auxiliary_target_1", "auxiliary_target_2", "forward_return"]]
    return X, y

X_nn, y_nn = preprocess_nn(X, y)

# split 
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_nn,
    y_nn,
    test_size=0.2,
    shuffle=False,          
    random_state=SEED,
)

# scale using train only
x_scaler = StandardScaler().fit(X_train_raw.values)
y_scaler = StandardScaler().fit(y_train_raw.values)

X_train = x_scaler.transform(X_train_raw.values)
X_val   = x_scaler.transform(X_val_raw.values)

y_train = y_scaler.transform(y_train_raw.values)
y_val   = y_scaler.transform(y_val_raw.values)

# build neural net
def build_ffn(input_dim: int, output_dim: int = 3):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dense(16, activation="relu"),
        BatchNormalization(),
        Dense(8, activation="relu"),
        BatchNormalization(),
        Dense(output_dim)
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return model


model = build_ffn(X_train.shape[1], output_dim=3)
model.summary()

# early stopping
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
)

def eval_forward_return(y_true_scaled, y_pred_scaled, scaler, name: str):
    """Evaluate only the forward_return (3rd column)."""
    y_true = scaler.inverse_transform(y_true_scaled)[:, 2]
    y_pred = scaler.inverse_transform(y_pred_scaled)[:, 2]
    return {
        "model": model,
        "model_name": name,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "prediction" : y_pred
    }


# validation metrics
val_pred_scaled = model.predict(X_val)
result_nn_val = eval_forward_return(y_val, val_pred_scaled, y_scaler, "neural_net")
print("Validation metrics:", result_nn_val)

# OOS evaluation
X_oos_nn, y_oos_nn = preprocess_nn(X_oos, y_oos)

X_oos_scaled = x_scaler.transform(X_oos_nn.values)
y_oos_scaled = y_scaler.transform(y_oos_nn.values)

y_oos_pred_scaled = model.predict(X_oos_scaled)
result_neural_net = eval_forward_return(
    y_oos_scaled, y_oos_pred_scaled, y_scaler, "neural_net"
)

evaluation_results.append(result_neural_net)

# optional scatter plot
plot_predicted_vs_true(
    y_scaler.inverse_transform(y_oos_scaled)[:, 2],
    y_scaler.inverse_transform(y_oos_pred_scaled)[:, 2],
    title="Neural Net: Predicted vs True (forward_return)",
)

#%% Support Vector Machine - Execute 

def preprocess_svm(X: pd.DataFrame, y: pd.DataFrame):
    X = X.drop(
        columns=["stock_id", "date", "MASK_EARNINGS", "value_smart_eps"],
        errors="ignore",
    )
    y = y["forward_return"]         
    return X, y

# 1) Preprocess train and OOS sets
X_svm, y_svm = preprocess_svm(X, y)
X_oos_svm, y_oos_svm = preprocess_svm(X_oos, y_oos)

# 2) Fit scaler only on the in-sample features
x_scaler_svm = StandardScaler().fit(X_svm.values)

# 3) Transform both in-sample and OOS
X_svm_scaled = x_scaler_svm.transform(X_svm.values)
X_oos_svm_scaled = x_scaler_svm.transform(X_oos_svm.values)

# 4) Prepare targets as 1-D floats
y_svm = y_svm.values.astype(float)
y_oos_svm   = y_oos_svm.values.astype(float)

def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    epsilon: float = 0.1,
    gamma: str | float = "scale"
) -> SVR:
    model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
    model.fit(X, y)
    return model

# Train
svm_model = train_svm(X_svm, y_svm)

# Predict
y_oos_pred_svm = svm_model.predict(X_oos_svm)
y_oos_true = y_oos_svm  

# Evaluate
result_svm_oos = {
    "model": model,
    "model_name": "svm",
    "rmse": float(np.sqrt(mean_squared_error(y_oos_true, y_oos_pred_svm))),
    "mae":  float(mean_absolute_error(y_oos_true, y_oos_pred_svm)),
    "r2":   float(r2_score(y_oos_true, y_oos_pred_svm)),
    "mape": float(mean_absolute_percentage_error(y_oos_true, y_oos_pred_svm) * 100),
    "prediction" : y_oos_pred_svm
}

evaluation_results.append(result_svm_oos)

# Scatter Plot
plot_predicted_vs_true(
    y_oos_true,
    y_oos_pred_svm,
    title="SVM: Predicted vs True (forward_return)"
)

#%% Lasso Regression - Execute

def preprocess_lasso(X: pd.DataFrame, y: pd.DataFrame):
    X = X.drop(
        columns=["stock_id", "date", "MASK_EARNINGS", "value_smart_eps"],
        errors="ignore",
    )
    y = y["forward_return"]           
    return X, y

# 1) preprocess in-sample and OOS
X_lasso,    y_lasso    = preprocess_lasso(X,     y)
X_oos_lasso, y_oos_lasso = preprocess_lasso(X_oos, y_oos)

# 2) fit scaler on in-sample features only
x_scaler_lasso = StandardScaler().fit(X_lasso.values)

# 3) transform both sets
X_lasso_scaled     = x_scaler_lasso.transform(X_lasso.values)
X_oos_lasso_scaled = x_scaler_lasso.transform(X_oos_lasso.values)

# 4) targets
y_lasso      = y_lasso.values.astype(float)
y_oos_lasso  = y_oos_lasso.values.astype(float)


def train_lasso(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list[float] | None = None,
    cv: int = 10,
    max_iter: int = 10000,
    tol: float = 1e-3
) -> LassoCV:
    model = LassoCV(
        alphas=alphas,
        cv=cv,
        max_iter=max_iter,
        tol=tol,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X, y)
    print("Chosen alpha:", model.alpha_)
    return model

lasso_model = train_lasso(X_lasso_scaled, y_lasso)

y_oos_pred_lasso = lasso_model.predict(X_oos_lasso_scaled)
y_oos_true = y_oos_lasso

result_lasso_oos = {
    "model": model,
    "model_name": "lasso",
    "rmse": float(np.sqrt(mean_squared_error(y_oos_true, y_oos_pred_lasso))),
    "mae":  float(mean_absolute_error(y_oos_true, y_oos_pred_lasso)),
    "r2":   float(r2_score(y_oos_true, y_oos_pred_lasso)),
    "mape": float(mean_absolute_percentage_error(y_oos_true, y_oos_pred_lasso) * 100),
    "prediction" : y_oos_pred_lasso
}

evaluation_results.append(result_lasso_oos)

plot_predicted_vs_true(
    y_oos_true,
    y_oos_pred_lasso,
    title="Lasso: Predicted vs True (forward_return)"
)

#%% Final Evaluation

def plot_model_evaluation_metrics(evaluation_results):
    """
    Plots RMSE, MAE, R², and MAPE for each model in separate subplots,
    with y-limits set from min–10% to max+10% of the metric range.

    Args:
        evaluation_results (list of dict): List containing evaluation results with keys:
            - model_name
            - rmse
            - mae
            - r2
            - mape
    """
    # Convert to DataFrame
    df = pd.DataFrame(evaluation_results)

    # Melt the dataframe for easier plotting
    df_melted = df.melt(
        id_vars="model_name",
        value_vars=["rmse", "mae", "r2", "mape"],
        var_name="metric",
        value_name="value"
    )

    # Prepare subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    metrics = ["rmse", "mae", "r2", "mape"]

    for ax, metric in zip(axes, metrics):
        data = df_melted[df_melted["metric"] == metric]
        sns.barplot(
            data=data,
            x="model_name",
            y="value",
            ax=ax,
            palette="tab10"
        )
        ax.set_title(metric.upper())
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=45)

        # compute padding
        vals = data["value"]
        vmin, vmax = vals.min(), vals.max()
        dr = vmax - vmin
        pad = dr * 0.1 if dr != 0 else abs(vmax) * 0.1 or 1.0
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.set_ylabel(metric.upper())
        ax.grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()


def select_best_model(evaluation_results, metric="r2"):
    """
    Select the best model according to a chosen metric.

    Args:
        evaluation_results (list of dict): Each dict must contain:
            - "model": the fitted estimator
            - "model_name": string identifier
            - metrics like "rmse", "mae", "r2", "mape"
        metric (str): which metric to use for selection.
            - For "r2", higher is better.
            - For all others, lower is better.

    Returns:
        best_model (estimator): the fitted model with best performance.
    """
    # Validate
    if not evaluation_results:
        raise ValueError("evaluation_results is empty")
    if metric not in evaluation_results[0]:
        raise KeyError(f"Metric '{metric}' not found in results")

    # Determine comparator
    reverse = metric == "r2"

    # Find best entry
    best = max(evaluation_results, key=lambda r: r[metric]) if reverse \
           else min(evaluation_results, key=lambda r: r[metric])

    print(f"Best model by {metric}: {best['model_name']} ({metric}={best[metric]:.4f})")
    return best["model"], best["prediction"]

plot_model_evaluation_metrics(evaluation_results)

best_model, best_prediction = select_best_model(evaluation_results, metric="r2")

#%% Backtesting

def plot_comparative_equity(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    invest_frac: float = 0.5,
    initial_capital: float = 1.0,
    freq: int = 252
):
    """
    Plot equity curves for three strategies:
      • blue: your model’s predictions (y_pred_model)
      • green: always long  (+1)
      • red:   always short (-1)

    Args:
        y_true         (array): actual forward returns, shape (N,)
        y_pred_model   (array): model’s predicted returns, shape (N,)
        invest_frac    (float): fraction of equity to risk per trade
        initial_capital(float): starting portfolio value
        freq           (int):   annualization factor (for metrics calc—but here just plotting)
    """

    # helper to compute equity curve
    def compute_equity(y_true, y_pred, invest_frac, initial_capital):
        sign = np.where(y_pred > 0, 1.0, -1.0)
        profit = invest_frac * sign * y_true
        equity = initial_capital * np.cumprod(1 + profit)
        return equity

    # Model equity
    eq_model = compute_equity(y_true, y_pred_model, invest_frac, initial_capital)
    # Always long equity
    eq_long  = compute_equity(y_true, np.ones_like(y_true), invest_frac, initial_capital)
    # Always short equity
    eq_short = compute_equity(y_true, -np.ones_like(y_true), invest_frac, initial_capital)

    # Build DataFrame for plotting
    df = pd.DataFrame({
        "Model (blue)":     eq_model,
        "Always Long (green)":  eq_long,
        "Always Short (red)":   eq_short
    })

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["Model (blue)"],     color="blue",  label="Model")
    plt.plot(df["Always Long (green)"], color="green", label="Always Long")
    plt.plot(df["Always Short (red)"], color="red",   label="Always Short")
    plt.title("Equity Curves: Model vs. Always Long vs. Always Short")
    plt.xlabel("Trade Number")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    
for i in range(0,6):
    plot_comparative_equity(
        y_true=y_oos["forward_return"].values,
        y_pred_model = evaluation_results[i]['prediction'],
        invest_frac=0.1,
        initial_capital=1000.0
    )

