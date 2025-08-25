#!/usr/bin/env python3
"""
Debug window feature extraction to understand shape issues.

NOTE: This script requires the ddpred package to be installed for full functionality.
The ddpred package provides data processing and feature extraction capabilities
that complement the hf_utils dataset loading functions.

To install ddpred: pip install -e /path/to/ddpred
"""

from hf_utils import DataDecide

# These imports require ddpred package to be installed separately
try:
    import ddpred.data as dd_data
    import ddpred.features as dd_feats
except ImportError as e:
    print("Error: ddpred package required but not found.")
    print("Please install ddpred package: pip install -e /path/to/ddpred")
    print(f"Import error: {e}")
    exit(1)

# Setup identical to main script
dd = DataDecide()
base_df = dd_data.prep_base_df(
    dd,
    filter_by_max_step=True,
    add_lr_cols=True,
    return_means=True,
)

feature_config = dd_feats.FeatureConfig(
    dd,
    windows=dd_feats.EarlyWindowData(
        wiggle_room=0,
        drop_max_step=False,
        percentage_windows=["[0, 25]", "[25, 50]"],
    ),
    target_feature_values=[70, 80, 90, 100],
    axis_types=["xlinylin"],
    basic_features={
        "first_val": [True, False],
        "last_val": [True, True],
        "val_first_last_slope": [True, True],
    },
)

# Simulate what the trainer does
print("=== SIMULATING TRAINER STEPS ===")
updated_df = base_df.copy()
updated_df["group"] = updated_df["data"] + "_" + updated_df["params"]
print(f"Updated DF shape: {updated_df.shape}")

# Step 1: Extract fixed features
print("\n=== STEP 1: Fixed Features ===")
keep_cols = ["params", "data", "group"]
feature_df = dd_feats.extract_fixed_features(
    feature_config.fixed_features,
    updated_df,
    keep_cols=keep_cols,
)
print(f"Fixed features DF shape: {feature_df.shape}")
print(f"Fixed features columns: {list(feature_df.columns)}")

# Step 2: Create window DFs
print("\n=== STEP 2: Window DFs ===")
feature_config.windows.create_dfs(updated_df)
print(f"Window names: {list(feature_config.windows.dfs.keys())}")
for name, window_df in feature_config.windows.dfs.items():
    print(f"  {name}: {window_df.shape}")

# Step 3: Extract window features
print("\n=== STEP 3: Window Features ===")
window_features = dd_feats.extract_window_features(
    feature_config.windows.dfs,
    list(feature_config.window_features.values()),
    "pile-valppl",
    dd_feats.extract_basic_features,
    only_feat_names=dd_feats.BASIC_FEATURES,
    x_axis_key=feature_config.x_axis_key,
)
print(f"Window features DF shape: {window_features.shape}")
print(f"Window features columns: {list(window_features.columns)}")
print("Window features key counts:")
print(f"  Unique params: {window_features['params'].nunique()}")
print(f"  Unique data: {window_features['data'].nunique()}")
print(
    f"  Unique (params, data): {window_features[['params', 'data']].drop_duplicates().shape[0]}"
)

# Sample data
print("\nSample window features:")
print(window_features[["params", "data"]].head(10))

# Step 4: Create target features
print("\n=== STEP 4: Target Features ===")
target_features_df, target_values_df = feature_config.extract_target_features(
    updated_df, "pile-valppl"
)
print(f"Target features DF shape: {target_features_df.shape}")
print(f"Target features columns: {list(target_features_df.columns)}")
print(f"Target values DF shape: {target_values_df.shape}")

# Check what columns are in common for merging
print("\n=== MERGE ANALYSIS ===")
feature_cols = set(feature_df.columns)
window_cols = set(window_features.columns)
target_cols = set(target_features_df.columns)

print(f"Fixed feature columns: {feature_cols}")
print(f"Window feature columns: {window_cols}")
print(f"Target feature columns: {target_cols}")
print(f"Common between fixed & window: {feature_cols & window_cols}")
print(f"Common between fixed & target: {feature_cols & target_cols}")
print(f"Common between window & target: {window_cols & target_cols}")

# Try the actual merges
print("\n=== TESTING MERGES ===")
# Merge 1: fixed + window (on params, data)
merged1 = feature_df.merge(window_features, on=["params", "data"], how="outer")
print(f"After fixed + window merge: {merged1.shape}")

# Merge 2: add target features (the problematic merge)
merge_cols = [col for col in target_features_df.columns if col in merged1.columns]
print(f"Merge columns for target: {merge_cols}")
merged2 = merged1.merge(target_features_df, on=merge_cols, how="left")
print(f"After target merge: {merged2.shape}")

print(
    f"Final feature DF would have {merged2.shape[0]} rows vs {target_values_df.shape[0]} target rows"
)
