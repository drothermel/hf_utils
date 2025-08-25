#!/usr/bin/env python3
"""
Feature Correlation Analysis Script

Analyzes correlations between features and targets across different model sizes.
Creates visualizations showing how feature correlations vary with model size.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from typing import List, Tuple, Optional
from hf_utils import DataDecidePaths
import argparse


def load_all_data(target_col: str = "log_final_pile-valppl") -> pd.DataFrame:
    """
    Load the complete dataset from dataset_750M.pkl which contains all model sizes.

    Args:
        target_col: Target column name

    Returns:
        Combined DataFrame with columns: params, data, features..., target
    """
    # Load the complete dataset
    dataset_path = DataDecidePaths().dataset_path("750M")
    print(f"Loading complete dataset from {dataset_path}")

    with open(dataset_path, "rb") as f:
        dataset = pkl.load(f)
        split_features_dfs = dataset["split_features_dfs"]
        split_target_dfs = dataset["split_target_dfs"]

    combined_data = []

    # Combine train, val, and eval splits
    for split in ["train", "val", "eval"]:
        if split in split_features_dfs and split in split_target_dfs:
            features_df = split_features_dfs[split].copy()
            targets_df = split_target_dfs[split].copy()

            # Add target and split info
            features_df["target"] = targets_df[target_col]
            features_df["split"] = split

            combined_data.append(features_df)

    if not combined_data:
        raise ValueError("No datasets could be loaded successfully")

    combined_df = pd.concat(combined_data, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Model sizes in params column: {sorted(combined_df['params'].unique())}")
    print(f"Data types: {combined_df['data'].unique()}")

    return combined_df


def compute_feature_correlations(
    df: pd.DataFrame, features: List[str], groupby_strategy: str = "params"
) -> pd.DataFrame:
    """
    Compute correlations between features and target using different grouping strategies.

    Args:
        df: Combined dataset
        features: List of feature names to analyze
        groupby_strategy: 'params' (across datasets for each model size),
                         'data' (across model sizes for each dataset),
                         'all' (all data together)

    Returns:
        DataFrame with columns depending on strategy
    """
    correlation_results = []

    # Get feature columns (exclude metadata columns)
    feature_columns = [
        col for col in df.columns if col not in ["params", "data", "target", "split"]
    ]

    # Filter to requested features if specified
    if features:
        feature_columns = [col for col in feature_columns if col in features]
        missing_features = set(features) - set(feature_columns)
        if missing_features:
            print(f"Warning: Features not found: {missing_features}")

    print(
        f"Analyzing {len(feature_columns)} features with strategy '{groupby_strategy}'"
    )

    if groupby_strategy == "params":
        # Group by model size, compute correlation across different datasets
        for params, group in df.groupby("params"):
            if len(group) < 2:
                continue
            for feature in feature_columns:
                if feature in group.columns and not group[feature].isna().all():
                    corr = group[feature].corr(group["target"])
                    if not pd.isna(corr):
                        correlation_results.append(
                            {
                                "params": params,
                                "feature": feature,
                                "correlation": corr,
                                "n_samples": len(group),
                            }
                        )

    elif groupby_strategy == "data":
        # Group by dataset, compute correlation across different model sizes
        for data, group in df.groupby("data"):
            if len(group) < 2:
                continue
            for feature in feature_columns:
                if feature in group.columns and not group[feature].isna().all():
                    corr = group[feature].corr(group["target"])
                    if not pd.isna(corr):
                        correlation_results.append(
                            {
                                "data": data,
                                "feature": feature,
                                "correlation": corr,
                                "n_samples": len(group),
                            }
                        )

    elif groupby_strategy == "all":
        # Compute correlation across all data (no grouping)
        for feature in feature_columns:
            if feature in df.columns and not df[feature].isna().all():
                corr = df[feature].corr(df["target"])
                if not pd.isna(corr):
                    correlation_results.append(
                        {"feature": feature, "correlation": corr, "n_samples": len(df)}
                    )

    return pd.DataFrame(correlation_results)


def extract_model_size_numeric(params_str: str) -> float:
    """Extract numeric value from params string for plotting."""
    import re

    # Extract number and convert units
    match = re.match(r"(\d+(?:\.\d+)?)([KMGT]?)", params_str.upper())
    if not match:
        return float("inf")  # Put unknown sizes at the end

    number, unit = match.groups()
    number = float(number)

    multipliers = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
    return number * multipliers.get(unit, 1)


def plot_correlation_vs_model_size(
    corr_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Create scatter plot of correlation vs model size, colored by feature.

    Args:
        corr_df: DataFrame from compute_feature_correlations
        features: List of features to plot (None for all)
        output_path: Path to save plot (None to display)
        figsize: Figure size tuple
    """
    if features:
        corr_df = corr_df[corr_df["feature"].isin(features)]

    # Check if we have params column (strategy='params')
    if "params" not in corr_df.columns:
        print("No 'params' column found - cannot plot vs model size")
        return

    # Add numeric model size for x-axis
    corr_df = corr_df.copy()
    corr_df["params_numeric"] = corr_df["params"].apply(extract_model_size_numeric)
    corr_df = corr_df.sort_values("params_numeric")

    # Create the plot
    plt.figure(figsize=figsize)

    # Use seaborn for nice colors
    unique_features = corr_df["feature"].unique()
    palette = sns.color_palette("husl", len(unique_features))

    for i, feature in enumerate(unique_features):
        feature_data = corr_df[corr_df["feature"] == feature]
        plt.scatter(
            feature_data["params_numeric"],
            feature_data["correlation"],
            label=feature,
            alpha=0.7,
            s=60,
            color=palette[i],
        )

    plt.xlabel("Model Size (Parameters)")
    plt.ylabel("Correlation with Target")
    plt.title(
        "Feature-Target Correlations vs Model Size\n(Correlations computed across datasets within each model size)"
    )
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Format x-axis labels
    ax = plt.gca()
    x_ticks = sorted(corr_df["params_numeric"].unique())
    x_labels = []
    for tick in x_ticks:
        # Find corresponding string
        matching_rows = corr_df[corr_df["params_numeric"] == tick]
        if not matching_rows.empty:
            x_labels.append(matching_rows.iloc[0]["params"])
        else:
            x_labels.append(f"{tick:.0e}")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)

    # Legend
    if len(unique_features) <= 20:  # Only show legend if not too many features
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        print(f"Too many features ({len(unique_features)}) for legend")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature correlations across model sizes"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Specific features to analyze (default: all)",
    )
    parser.add_argument(
        "--target-col", default="log_final_pile-valppl", help="Target column name"
    )
    parser.add_argument(
        "--output", default=None, help="Output path for plot (default: display)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Show only top N features by mean absolute correlation",
    )
    parser.add_argument(
        "--strategy",
        choices=["params", "data", "all"],
        default="params",
        help="Grouping strategy: params (across datasets per model size), data (across model sizes per dataset), all (no grouping)",
    )

    args = parser.parse_args()

    # Load complete dataset
    print("Loading dataset...")
    combined_df = load_all_data(args.target_col)

    # Compute correlations
    print("Computing correlations...")
    corr_df = compute_feature_correlations(combined_df, args.features, args.strategy)

    if corr_df.empty:
        print("No correlations could be computed!")
        return

    # Filter to top N features if requested
    if args.top_n:
        feature_mean_corr = corr_df.groupby("feature")["correlation"].apply(
            lambda x: abs(x).mean()
        )
        top_features = feature_mean_corr.nlargest(args.top_n).index.tolist()
        print(f"Top {args.top_n} features by mean absolute correlation:")
        for feat in top_features:
            print(f"  {feat}: {feature_mean_corr[feat]:.3f}")
        corr_df = corr_df[corr_df["feature"].isin(top_features)]

    # Create plot
    print("Creating plot...")
    if args.strategy == "params":
        plot_correlation_vs_model_size(corr_df, output_path=args.output)
    else:
        print(
            f"Plotting not implemented for strategy '{args.strategy}' - showing data instead"
        )
        print(corr_df.head(20))

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total correlations computed: {len(corr_df)}")
    print(f"Features analyzed: {corr_df['feature'].nunique()}")

    if "params" in corr_df.columns:
        print(
            f"Model sizes: {sorted(corr_df['params'].unique(), key=extract_model_size_numeric)}"
        )
    if "data" in corr_df.columns:
        print(f"Datasets: {sorted(corr_df['data'].unique())}")

    # Show strongest correlations
    print("\nStrongest positive correlations:")
    top_pos = corr_df.nlargest(5, "correlation")
    for _, row in top_pos.iterrows():
        group_info = []
        if "params" in row:
            group_info.append(f"params={row['params']}")
        if "data" in row:
            group_info.append(f"data={row['data']}")
        group_str = ", ".join(group_info) if group_info else "all data"
        print(f"  {row['feature']} ({group_str}): {row['correlation']:.3f}")

    print("\nStrongest negative correlations:")
    top_neg = corr_df.nsmallest(5, "correlation")
    for _, row in top_neg.iterrows():
        group_info = []
        if "params" in row:
            group_info.append(f"params={row['params']}")
        if "data" in row:
            group_info.append(f"data={row['data']}")
        group_str = ", ".join(group_info) if group_info else "all data"
        print(f"  {row['feature']} ({group_str}): {row['correlation']:.3f}")


if __name__ == "__main__":
    main()
