"""
HuggingFace utilities package.

This package provides utilities for working with HuggingFace datasets and models,
particularly for the DataDecide framework.
"""

import pandas as pd
from .datadecide import DataDecide, DataDecidePaths


def load_datadecide_perplexity_results(force_reload: bool = False) -> pd.DataFrame:
    """Load the parsed perplexity evaluation results."""
    dd = DataDecide(force_reload=force_reload, verbose=False)
    dd.load_df("ppl_parsed_df")
    return dd._loaded_dfs["ppl_parsed_df"]


def load_datadecide_downstream_results_parsed(
    force_reload: bool = False,
) -> pd.DataFrame:
    """Load the parsed downstream evaluation results."""
    dd = DataDecide(force_reload=force_reload, verbose=False)
    dd.load_df("dwn_parsed_df")
    return dd._loaded_dfs["dwn_parsed_df"]


__all__ = [
    "DataDecide",
    "DataDecidePaths",
    "load_datadecide_perplexity_results",
    "load_datadecide_downstream_results_parsed",
]
