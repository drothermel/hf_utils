#!/usr/bin/env python3
"""
Debug script to understand the shape misalignment issue.

NOTE: This script requires the ddpred package to be installed for full functionality.
The ddpred package provides data processing, feature extraction, and training capabilities
that complement the hf_utils dataset loading functions.

To install ddpred: pip install -e /path/to/ddpred
"""

from hf_utils import DataDecide

# These imports require ddpred package to be installed separately
try:
    import ddpred.data as dd_data
    import ddpred.features as dd_feats
    from ddpred.trainers.trainer import Trainer
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

trainer = Trainer(
    dd=dd,
    feature_config=feature_config,
    base_df=base_df,
    target_col="pile-valppl",
)

# Debug the shape issue
trainer.setup_trainer()

print("=== DETAILED SHAPE ANALYSIS ===")
print(f"Base DF shape: {base_df.shape}")
print(f"Updated DF shape: {trainer.updated_df.shape}")
print(f"Feature DF shape: {trainer.feature_df.shape}")
print(f"Target DF shape: {trainer.target_df.shape}")

print("\n=== KEY COLUMN ANALYSIS ===")
print("Base DF key columns:")
print(f"  Unique params: {base_df['params'].nunique()}")
print(f"  Unique data: {base_df['data'].nunique()}")
print(
    f"  Unique (params, data): {base_df[['params', 'data']].drop_duplicates().shape[0]}"
)

print("\nFeature DF key columns:")
print(f"  Unique params: {trainer.feature_df['params'].nunique()}")
print(f"  Unique data: {trainer.feature_df['data'].nunique()}")
print(
    f"  Unique (params, data): {trainer.feature_df[['params', 'data']].drop_duplicates().shape[0]}"
)
if "target_percentage" in trainer.feature_df.columns:
    print(
        f"  Unique target_percentage: {trainer.feature_df['target_percentage'].nunique()}"
    )
    print(
        f"  Target percentage values: {sorted(trainer.feature_df['target_percentage'].unique())}"
    )

print("\nTarget DF key columns:")
print(f"  Target DF columns: {list(trainer.target_df.columns)}")
print(f"  Target DF index type: {type(trainer.target_df.index)}")
print(f"  Target DF index names: {trainer.target_df.index.names}")

print("\n=== SAMPLE DATA ===")
print("Feature DF sample:")
print(
    trainer.feature_df[["params", "data", "target_percentage", "target_step"]].head(10)
)
print("\nTarget DF sample:")
print(trainer.target_df.head(10))

print("\n=== FEATURE DF GROUPING ANALYSIS ===")
feature_groupby = trainer.feature_df.groupby(["params", "data"]).size()
print("Rows per (params, data) combination in feature_df:")
print(f"  Min: {feature_groupby.min()}")
print(f"  Max: {feature_groupby.max()}")
print(f"  Mean: {feature_groupby.mean():.1f}")
print(f"  Unique counts: {feature_groupby.value_counts().head()}")

print("\n=== TARGET FEATURE VALUES ANALYSIS ===")
print(f"Target feature values in config: {feature_config.target_feature_values}")
target_groupby = trainer.feature_df.groupby(
    ["params", "data", "target_percentage"]
).size()
print("Rows per (params, data, target_percentage):")
print(f"  Min: {target_groupby.min()}")
print(f"  Max: {target_groupby.max()}")
print(f"  Mean: {target_groupby.mean():.1f}")

# Check expected vs actual calculations
expected_feature_rows = (
    350
    * len(feature_config.target_feature_values)
    * len(feature_config.axis_types)
    * len(feature_config.windows.window_names)
)
print("\n=== EXPECTED CALCULATIONS ===")
print(
    f"Expected feature rows: 350 × {len(feature_config.target_feature_values)} × {len(feature_config.axis_types)} × {len(feature_config.windows.window_names)} = {expected_feature_rows}"
)
print(f"Actual feature rows: {len(trainer.feature_df)}")
print(
    f"Expected target rows: 350 × {len(feature_config.target_feature_values)} = {350 * len(feature_config.target_feature_values)}"
)
print(f"Actual target rows: {len(trainer.target_df)}")

print("\n=== TRYING GET_TRAINING_DFS ===")
try:
    X, y, groups = trainer.get_training_dfs()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"groups shape: {groups.shape}")
except Exception as e:
    print(f"Error in get_training_dfs: {e}")
