#%%
# type: ignore
%load_ext autoreload
%autoreload 2
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import random
import pickle as pkl

from hf_utils import DataDecidePaths

# This import requires ddpred package to be installed separately
# NOTE: This notebook requires ddpred for dataset creation functionality
try:
    from ddpred.data import create_datasets
except ImportError as e:
    print("Error: ddpred package required but not found.")
    print("Please install ddpred package: pip install -e /path/to/ddpred")
    print(f"Import error: {e}")
    raise

# %%
# Dump Datasets
for max_params_str in ["150M", "300M", "530M", "750M"]:
    dataset_path = DataDecidePaths().dataset_path(max_params_str)
    split_features_dfs, split_target_dfs, _, sub_features_dfs = create_datasets(
        max_train_params=max_params_str,
        metric_col="pile-valppl",
        verbose=False,
    )
    with open(dataset_path, "wb") as f:
        dataset = {
            'split_features_dfs': split_features_dfs,   
            'split_target_dfs': split_target_dfs,
        }
        pkl.dump(dataset, f)
        print(f"Dumped dataset to {dataset_path}")

    sub_dataset_path = DataDecidePaths().dataset_path(max_params_str + "_sub_features")
    with open(sub_dataset_path, "wb") as f:
        pkl.dump(sub_features_dfs, f)
        print(f"Dumped sub-features dataset to {sub_dataset_path}")

# %%
# Load Datasets
def load_dataset(max_params_str: str):
    dataset_path = DataDecidePaths().dataset_path(max_params_str)
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, "rb") as f:
        dataset = pkl.load(f)
        return dataset['split_features_dfs'], dataset['split_target_dfs']

# Options: "150M", "300M", "530M", "750M"
load_param_str = "150M"
fixed_seed = 42
stopping_rounds = 10
reg_params = dict(
    objective          = 'regression',
    metric             = 'rmse',
    num_leaves         = 5,
    learning_rate      = 0.01,
    min_data_in_leaf   = 10,
    feature_fraction   = 0.5,
    bagging_fraction   = 0.7,
    bagging_freq       = 1,
    max_bin            = 50,
    min_gain_to_split  = 0.1,
    reg_alpha          = 1.0,
    reg_lambda         = 1.0,
    n_estimators       = 50,
    random_state       = fixed_seed,
    bagging_seed       = fixed_seed,          # Seed for bagging
    feature_fraction_seed = fixed_seed,       # Seed for feature sampling
    deterministic      = True,        # Force deterministic behavior
    verbosity          = -1,
  	#device_type        = 'cuda',
)

random.seed(42)
np.random.seed(42)

split_features_dfs, split_target_dfs = load_dataset(load_param_str)
X_train = split_features_dfs['train'].drop(columns=['params', 'data'])
y_train = split_target_dfs['train']['log_final_pile-valppl']
X_val = split_features_dfs['val'].drop(columns=['params', 'data'])
y_val = split_target_dfs['val']['log_final_pile-valppl']
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

# Make the regressor and fit it
reg_ppl = lgb.LGBMRegressor(**reg_params)
reg_ppl.fit(
    X_train, y_train,
    eval_set   = [(X_train, y_train), (X_val, y_val)],
    eval_names = ['train', 'val'],
    eval_metric= 'rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=True)],
)
#print("Training history:", reg_ppl.evals_result_)

# Plot the training metrics
lgb.plot_metric(reg_ppl)

#%%hi
def load_dataset(max_params_str: str):
    dataset_path = DataDecidePaths().dataset_path(max_params_str)
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, "rb") as f:
        dataset = pkl.load(f)
        return dataset['split_features_dfs'], dataset['split_target_dfs']

def prep_ranking_data(split_features, split_targets, col="log_final_pile-valppl"):
    feats = split_features.sort_values(by='params').reset_index(drop=True)
    targets_df = split_targets.sort_values(by='params').reset_index(drop=True)

    targets_df['rank'] = targets_df.groupby('params')[col].rank(method='dense', ascending=False).astype(int)

    num_data_per_group = feats[['params', 'data']].groupby('params').count().reset_index(drop=True)['data'].tolist()
    feats = feats.drop(columns=['params', 'data'])
    targets = targets_df['rank']

    print(num_data_per_group)
    print(feats.shape, targets.shape)
    print("Target range:", targets.min(), targets.max())
    return feats, targets, num_data_per_group

load_param_str = "150M"
fixed_seed = 42
random.seed(42)
np.random.seed(42)
split_features_dfs, split_target_dfs = load_dataset(load_param_str)
train_feats, train_targets, train_data_per_group = prep_ranking_data(split_features_dfs['train'], split_target_dfs['train'])
val_feats, val_targets, val_data_per_group = prep_ranking_data(split_features_dfs['eval'], split_target_dfs['eval'])

ranker = lgb.LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    learning_rate=0.02,          # Moderate learning rate
    num_leaves=10,               # Moderate complexity
    min_data_in_leaf=2,          # Allow some flexibility
    feature_fraction=0.7,        # Use most features
    bagging_fraction=0.8,        # Use most data
    reg_alpha=0.5,               # Light regularization
    reg_lambda=0.5,              # Light regularization
    n_estimators=100,
    random_state=42,
    ndcg_at=[1, 2, 3],
    verbosity=1,                 # See training progress
)

ranker.fit(
    train_feats, train_targets,
    group=train_data_per_group,              # [3, 3] = 3 datasets for each model size
    eval_set=[(train_feats, train_targets), (val_feats, val_targets)],
    eval_names=['train', 'val'],
    eval_group=[train_data_per_group, val_data_per_group],
    eval_at=[1, 2, 3],
    #callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)]
)

lgb.plot_metric(ranker, metric='ndcg@3')

# %%
lgb.plot_metric(ranker, metric='ndcg@2')

# %%
lgb.plot_metric(ranker, metric='ndcg@1')
# Prep val data
#%%



# %%







#%%
lgb.plot_metric(reg_ppl.evals_result_)


# %%
y_val_pred = reg_ppl.predict(X_val)
y_train_pred = reg_ppl.predict(X_train)
print("Validation RMSE:", np.sqrt(mean_squared_error(y_val, y_val_pred)))
print("Validation R²:", r2_score(y_val, y_val_pred))


#%%
# Feature importance
print("Feature importances:", reg_ppl.feature_importances_)

# Plot feature importance
lgb.plot_importance(reg_ppl, max_num_features=10)

# %%
# %%
# %%













# %%
