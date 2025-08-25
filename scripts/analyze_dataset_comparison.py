#!/usr/bin/env python3
"""
Comprehensive dataset comparison analysis script.

This script:
1. Loads perplexity and downstream evaluation datasets
2. Performs join analysis to compare dataset overlap
3. Creates visualization plots for both steps and tokens
4. Returns a combined dataframe with correct_prob values
"""

import warnings

# Suppress urllib3 OpenSSL warnings early
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+.*")

from dotenv import load_dotenv  # noqa: E402
from pprint import pprint  # noqa: E402

from hf_utils import (  # noqa: E402
    load_datadecide_perplexity_results,
    load_datadecide_downstream_results_parsed,
)

# Load environment variables
load_dotenv()


def params_to_millions(param_str):
    """Convert params to millions."""
    if param_str.endswith("M"):
        return float(param_str[:-1])
    elif param_str.endswith("B"):
        return float(param_str[:-1]) * 1000  # billions to millions
    else:
        return float(param_str) / 1e6


def load_and_remap_ppl_eval():
    print()
    print("\n>> 1. Loading perplexity evaluation dataset <<\n")
    ppl_df = load_datadecide_perplexity_results(force_reload=False)
    ppl_df["params_millions"] = ppl_df["params"].apply(params_to_millions)

    # Apply seed mapping for perplexity data
    seed_mapping = {
        "default": 0,
        "small aux 2": 1,
        "small aux 3": 2,
        "large aux 2": 3,
        "large aux 3": 4,
    }
    ppl_df["seed"] = ppl_df["seed"].map(seed_mapping)
    col_mapping = {
        "eval/wikitext_103-validation/Perplexity": "wikitext_103-valppl",
        "eval/pile-validation/Perplexity": "pile-valppl",
        "eval/c4_en-validation/Perplexity": "c4_en-valppl",
        "eval/m2d2_s2orc-validation/Perplexity": "m2d2_s2orc-valppl",
        "eval/ice-validation/Perplexity": "ice-valppl",
        "eval/dolma_wiki-validation/Perplexity": "dolma_wiki-valppl",
        "eval/dolma_stack-validation/Perplexity": "dolma_stack-valppl",
        "eval/dolma_reddit-validation/Perplexity": "dolma_reddit-valppl",
        "eval/dolma_pes2o-validation/Perplexity": "dolma_pes2o-valppl",
        "eval/dolma_common-crawl-validation/Perplexity": "dolma_common-crawl-valppl",
        "eval/dolma_books-validation/Perplexity": "dolma_books-valppl",
    }
    ppl_df = ppl_df.rename(columns=col_mapping)
    ppl_df = ppl_df[
        ["step", "data", "params", "seed", "params_millions"]
        + list(col_mapping.values())
    ]
    print(f"Perplexity dataset shape: {ppl_df.shape}")
    print("Columns:")
    pprint(list(ppl_df.columns))
    return ppl_df


def load_and_remap_downstream_eval():
    print()
    print("\n>> 2. Loading downstream evaluation dataset <<\n")
    dwn_df = load_datadecide_downstream_results_parsed(force_reload=False)
    dwn_df["params_millions"] = dwn_df["params"].apply(params_to_millions)
    dwn_df = dwn_df[
        [
            "params",
            "params_millions",
            "data",
            "task",
            "step",
            "seed",
            "tokens",
            "compute",
            "correct_prob",
        ]
    ]

    # Apply seed mapping for downstream data
    seed_mapping = {
        "default": 0,
        "small aux 2": 1,
        "small aux 3": 2,
        "large aux 2": 3,
        "large aux 3": 4,
    }
    dwn_df["seed"] = dwn_df["seed"].map(seed_mapping)
    print(f"Downstream dataset shape: {dwn_df.shape}")
    print("Columns:")
    pprint(list(dwn_df.columns))
    return dwn_df


def extract_param_step_token_mapping(dwn_df):
    """Extract the (param, step, token) mapping from downstream dataset."""
    print()
    print("\n>> 3. Extracting parameter-step-token mapping\n")

    # Create mapping from (params, step) -> tokens
    step_token_mapping = dwn_df[["params", "step", "tokens"]].drop_duplicates()

    print(
        f"   Found {len(step_token_mapping)} unique (param, step, token) combinations"
    )

    # Verify consistency: each (param, step) should map to exactly one token value
    consistency_check = step_token_mapping.groupby(["params", "step"])[
        "tokens"
    ].nunique()
    inconsistent_mappings = consistency_check[consistency_check > 1]

    if len(inconsistent_mappings) > 0:
        print(
            f"   WARNING: Found {len(inconsistent_mappings)} inconsistent (param, step) -> token mappings"
        )
        print(f"   First few inconsistencies: {inconsistent_mappings.head()}")
    else:
        print("   ✓ All (param, step) -> token mappings are consistent")

    return step_token_mapping


def add_tokens_to_perplexity_df(ppl_df, step_token_mapping):
    """Add token column to perplexity dataframe using step-token mapping."""
    print()
    print("\n>> 4. Adding tokens to perplexity dataset\n")

    original_shape = ppl_df.shape

    # Merge on params and step to add tokens
    ppl_with_tokens = ppl_df.merge(
        step_token_mapping[["params", "step", "tokens"]],
        on=["params", "step"],
        how="left",
    )

    # Check how many rows got token values
    rows_with_tokens = ppl_with_tokens["tokens"].notna().sum()
    total_rows = len(ppl_with_tokens)

    print(f"   Original perplexity dataset: {original_shape[0]} rows")
    print(f"   After adding tokens: {ppl_with_tokens.shape[0]} rows")
    print(
        f"   Rows with token values: {rows_with_tokens}/{total_rows} ({100 * rows_with_tokens / total_rows:.1f}%)"
    )

    return ppl_with_tokens


def join_datasets_with_tokens(ppl_with_tokens, dwn_df):
    """Join the two datasets on data, params, seed, step."""
    print()
    print("\n>> 5. Joining datasets\n")

    # Perform inner join on the key columns
    join_cols = ["data", "params", "seed", "step"]

    merged_df = ppl_with_tokens.merge(
        dwn_df, on=join_cols, how="outer", suffixes=("_ppl", "_dwn"), indicator=True
    )

    print(f"   Merged dataset shape: {merged_df.shape}")

    # Count join results
    join_counts = merged_df["_merge"].value_counts()
    print("   Join results:")
    for merge_type, count in join_counts.items():
        print(f"     {merge_type}: {count} rows")

    return merged_df


def analyze_joined_datasets(merged_df):
    """Analyze the joined dataset to understand overlap and coverage."""
    print()
    print("\n>> 6. Analyzing joined datasets\n")

    # Basic statistics
    total_rows = len(merged_df)
    both_datasets = merged_df[merged_df["_merge"] == "both"]
    ppl_only = merged_df[merged_df["_merge"] == "left_only"]
    dwn_only = merged_df[merged_df["_merge"] == "right_only"]

    print(f"    Total combinations: {total_rows}")
    print(
        f"    In both datasets: {len(both_datasets)} ({100 * len(both_datasets) / total_rows:.1f}%)"
    )
    print(
        f"    Perplexity only: {len(ppl_only)} ({100 * len(ppl_only) / total_rows:.1f}%)"
    )
    print(
        f"    Downstream only: {len(dwn_only)} ({100 * len(dwn_only) / total_rows:.1f}%)"
    )

    # Check token consistency in merged data
    both_with_tokens = both_datasets.dropna(subset=["tokens_ppl", "tokens_dwn"])
    if len(both_with_tokens) > 0:
        token_matches = (
            both_with_tokens["tokens_ppl"] == both_with_tokens["tokens_dwn"]
        ).sum()
        print(
            f"   Token consistency in overlapping data: {token_matches}/{len(both_with_tokens)} ({100 * token_matches / len(both_with_tokens):.1f}%)"
        )

    # Group-level analysis
    print("\n   Analysis by (data, params, seed) groups:")

    # Get max steps/tokens for each group in each dataset
    ppl_max = (
        merged_df[merged_df["_merge"].isin(["both", "left_only"])]
        .groupby(["data", "params", "seed"])
        .agg({"step": "max", "tokens_ppl": "max"})
        .rename(columns={"step": "max_step_ppl", "tokens_ppl": "max_tokens_ppl"})
    )

    dwn_max = (
        merged_df[merged_df["_merge"].isin(["both", "right_only"])]
        .groupby(["data", "params", "seed"])
        .agg({"step": "max", "tokens_dwn": "max"})
        .rename(columns={"step": "max_step_dwn", "tokens_dwn": "max_tokens_dwn"})
    )

    # Combine max values
    group_analysis = ppl_max.merge(
        dwn_max, left_index=True, right_index=True, how="outer"
    )

    # Count groups by coverage
    both_groups = group_analysis.dropna(subset=["max_step_ppl", "max_step_dwn"])
    ppl_only_groups = group_analysis[group_analysis["max_step_dwn"].isna()]
    dwn_only_groups = group_analysis[group_analysis["max_step_ppl"].isna()]

    print(f"      Groups in both datasets: {len(both_groups)}")
    print(f"      Groups in perplexity only: {len(ppl_only_groups)}")
    print(f"      Groups in downstream only: {len(dwn_only_groups)}")

    return {
        "merged_data": merged_df,
        "both_datasets": both_datasets,
        "ppl_only": ppl_only,
        "dwn_only": dwn_only,
        "group_analysis": group_analysis,
        "summary": {
            "total_rows": total_rows,
            "both_count": len(both_datasets),
            "ppl_only_count": len(ppl_only),
            "dwn_only_count": len(dwn_only),
            "both_groups": len(both_groups),
            "ppl_only_groups": len(ppl_only_groups),
            "dwn_only_groups": len(dwn_only_groups),
        },
    }


def create_combined_dataframe(analysis_results):
    """Create a clean combined dataframe from analysis results."""
    print()
    print("\n>> 7. Creating combined dataframe\n")

    # Get rows that exist in both datasets
    both_datasets = analysis_results["both_datasets"]
    combined_df = both_datasets.copy()

    # Handle duplicate columns
    combined_df = handle_duplicate_columns(combined_df)

    # Remove the merge indicator column
    combined_df = combined_df.drop(columns=["_merge"])

    # Reorder columns for better readability
    combined_df = reorder_columns(combined_df)

    # Print summary
    print(f"   Combined dataset shape: {combined_df.shape}")
    print(f"   Combined dataset columns: {list(combined_df.columns)}")

    print("\n>> 8. Sample of combined dataframe:\n")
    print(combined_df.head(10))

    return combined_df


def handle_duplicate_columns(combined_df):
    """Handle duplicate columns by verifying they are identical before removing."""
    duplicate_cols = {}

    for col in combined_df.columns:
        if col.endswith("_ppl"):
            base_col = col[:-4]
            dwn_col = base_col + "_dwn"
            if dwn_col in combined_df.columns:
                # Check if the columns are actually identical
                ppl_values = combined_df[col]
                dwn_values = combined_df[dwn_col]

                # Handle NaN values properly in comparison
                are_identical = (
                    (ppl_values == dwn_values) | (ppl_values.isna() & dwn_values.isna())
                ).all()

                if are_identical:
                    print(f"   ✓ Columns {col} and {dwn_col} are identical")
                    # For key columns, use the _ppl version and drop _dwn
                    if base_col in [
                        "data",
                        "params",
                        "seed",
                        "step",
                        "params_millions",
                    ]:
                        combined_df[base_col] = combined_df[col]
                        duplicate_cols[col] = base_col
                        duplicate_cols[dwn_col] = None  # Mark for removal
                    # For tokens, use _dwn version since it's the authoritative source
                    elif base_col == "tokens":
                        combined_df["tokens"] = combined_df[dwn_col]
                        duplicate_cols[col] = None
                        duplicate_cols[dwn_col] = "tokens"
                else:
                    # Handle non-identical columns
                    combined_df = handle_non_identical_columns(
                        combined_df, col, dwn_col, base_col, ppl_values, dwn_values
                    )
                    duplicate_cols[col] = f"{base_col}_perplexity"
                    duplicate_cols[dwn_col] = f"{base_col}_downstream"

    # Remove original duplicate columns (keeping the renamed ones)
    cols_to_drop = [col for col, new_name in duplicate_cols.items() if new_name is None]
    if cols_to_drop:
        print(f"   Dropping identical duplicate columns: {cols_to_drop}")
        combined_df = combined_df.drop(columns=cols_to_drop)

    return combined_df


def handle_non_identical_columns(
    combined_df, ppl_col, dwn_col, base_col, ppl_values, dwn_values
):
    """Handle columns that are not identical by showing differences and keeping both."""
    # Find differences for debugging
    diff_mask = ~((ppl_values == dwn_values) | (ppl_values.isna() & dwn_values.isna()))
    num_differences = diff_mask.sum()
    print(
        f"   ⚠️  WARNING: Columns {ppl_col} and {dwn_col} differ in {num_differences} rows"
    )

    if num_differences <= 10:  # Show details if not too many differences
        diff_rows = combined_df[diff_mask][
            ["data", "params", "seed", "step", ppl_col, dwn_col]
        ]
        print("   Differences:")
        print(diff_rows.head(10))

    # Keep both columns with descriptive names
    combined_df[f"{base_col}_perplexity"] = combined_df[ppl_col]
    combined_df[f"{base_col}_downstream"] = combined_df[dwn_col]

    return combined_df


def reorder_columns(combined_df):
    """Reorder columns for better readability."""
    key_cols = [
        "data",
        "params",
        "params_millions",
        "seed",
        "step",
        "tokens",
        "correct_prob",
    ]
    available_key_cols = [col for col in key_cols if col in combined_df.columns]
    other_cols = [col for col in combined_df.columns if col not in available_key_cols]
    return combined_df[available_key_cols + other_cols]


def print_completion_message():
    """Print the analysis completion message."""
    print()
    print("\n" + "=" * 80)
    print("= ANALYSIS COMPLETE")
    print("=" * 80)


def main():
    """Run the complete dataset comparison analysis."""
    print()
    print("=" * 80)
    print("= DATASET COMPARISON ANALYSIS   ")
    print("=" * 80)
    print()

    ppl_df = load_and_remap_ppl_eval()
    dwn_df = load_and_remap_downstream_eval()

    # Extract step-token mapping from downstream data
    step_token_mapping = extract_param_step_token_mapping(dwn_df)

    # Add tokens to perplexity data
    ppl_with_tokens = add_tokens_to_perplexity_df(ppl_df, step_token_mapping)

    # Join the datasets
    merged_df = join_datasets_with_tokens(ppl_with_tokens, dwn_df)

    # Analyze the joined datasets
    analysis_results = analyze_joined_datasets(merged_df)

    # Create final combined dataframe
    combined_df = create_combined_dataframe(analysis_results)

    # Print completion message
    print_completion_message()

    return combined_df


if __name__ == "__main__":
    combined_df = main()
