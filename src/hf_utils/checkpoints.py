"""Comprehensive checkpoint downloading and analysis functions."""

import torch
from huggingface_hub import hf_hub_download
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .branches import extract_step_from_branch, get_checkpoint_branches
from .configs import download_config_file, analyze_model_config, print_config_analysis
from .weights import analyze_model_weights, print_weight_analysis


def download_optimizer_checkpoint(
    repo_id: str, branch: str = "main", local_dir: Optional[str] = None
) -> Tuple[Optional[str], bool, str]:
    """Download optimizer checkpoint from Hugging Face repository.

    Args:
        repo_id: HuggingFace repository ID (e.g., "allenai/DataDecide-falcon-and-cc-qc-tulu-10p-60M")
        branch: Branch/revision to download from (e.g., "step0-seed-default")
        local_dir: Local directory to save file (optional)

    Returns:
        Tuple of (file_path, success, error_message)
        - file_path: Path to downloaded file if successful, None otherwise
        - success: True if download succeeded, False otherwise
        - error_message: Error description if failed, empty string if succeeded
    """
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename="training/optim.pt",
            revision=branch,
            local_dir=local_dir,
        )
        print(f"✅ Downloaded optimizer checkpoint to: {file_path}")
        return file_path, True, ""
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Entry Not Found" in error_msg:
            print(f"⚠️  No optimizer checkpoint found for branch {branch}")
            return None, False, "Optimizer checkpoint not available"
        else:
            print(f"❌ Error downloading checkpoint: {e}")
            return None, False, error_msg


def analyze_optimizer_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load and analyze PyTorch optimizer checkpoint.

    Args:
        checkpoint_path: Path to the optimizer checkpoint file

    Returns:
        Dictionary containing optimizer state information
    """
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        analysis = {
            "checkpoint_keys": list(checkpoint.keys())
            if isinstance(checkpoint, dict)
            else ["non_dict_checkpoint"],
            "checkpoint_type": type(checkpoint).__name__,
            "optimizer_info": {},
            "learning_rate_info": {},
        }

        # If it's a dictionary, examine common optimizer state keys
        if isinstance(checkpoint, dict):
            # Common keys in optimizer checkpoints
            for key in ["state", "param_groups", "optimizer", "lr_scheduler"]:
                if key in checkpoint:
                    analysis["optimizer_info"][key] = analyze_optimizer_component(
                        checkpoint[key], key
                    )

            # Look for learning rate information
            analysis["learning_rate_info"] = extract_learning_rate_info(checkpoint)

        return analysis

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return {"error": str(e)}


def analyze_optimizer_component(component: Any, component_name: str) -> Dict[str, Any]:
    """Analyze a specific component of the optimizer checkpoint."""
    info = {
        "type": type(component).__name__,
        "size": len(component) if hasattr(component, "__len__") else "N/A",
    }

    if component_name == "param_groups" and isinstance(component, list):
        # Extract parameter group information
        info["num_param_groups"] = len(component)
        if component:
            first_group = component[0]
            if isinstance(first_group, dict):
                info["param_group_keys"] = list(first_group.keys())
                # Look for learning rate in first parameter group
                if "lr" in first_group:
                    info["learning_rate"] = first_group["lr"]
                if "initial_lr" in first_group:
                    info["initial_learning_rate"] = first_group["initial_lr"]

    elif component_name == "state" and isinstance(component, dict):
        info["num_parameters"] = len(component)
        if component:
            # Look at first parameter's state
            first_param_state = next(iter(component.values()))
            if isinstance(first_param_state, dict):
                info["state_keys"] = list(first_param_state.keys())

    return info


def extract_learning_rate_info(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all learning rate related information from checkpoint."""
    lr_info = {}

    # Check param_groups for learning rates
    if "param_groups" in checkpoint and isinstance(checkpoint["param_groups"], list):
        lr_info["param_group_lrs"] = []
        for i, group in enumerate(checkpoint["param_groups"]):
            if isinstance(group, dict):
                group_lr_info = {"group_index": i}
                if "lr" in group:
                    group_lr_info["current_lr"] = group["lr"]
                if "initial_lr" in group:
                    group_lr_info["initial_lr"] = group["initial_lr"]
                if "weight_decay" in group:
                    group_lr_info["weight_decay"] = group["weight_decay"]
                if "momentum" in group:
                    group_lr_info["momentum"] = group["momentum"]
                lr_info["param_group_lrs"].append(group_lr_info)

    # Check for learning rate scheduler state
    if "lr_scheduler" in checkpoint:
        lr_info["lr_scheduler_present"] = True
        scheduler = checkpoint["lr_scheduler"]
        if isinstance(scheduler, dict):
            lr_info["lr_scheduler_keys"] = list(scheduler.keys())
            if "last_epoch" in scheduler:
                lr_info["scheduler_last_epoch"] = scheduler["last_epoch"]

    return lr_info


def download_all_checkpoint_components(
    repo_id: str,
    branch: str,
    include_weights: bool = False,
    weight_files: Optional[List[str]] = None,
    local_dir: Optional[str] = None,
) -> Dict[str, Tuple[Optional[str], bool, str]]:
    """Download all checkpoint components (optimizer, config, optionally weights).

    Args:
        repo_id: HuggingFace repository ID
        branch: Branch name to download from
        include_weights: Whether to download model weights
        weight_files: Specific weight files to download (optional)
        local_dir: Local directory to save files (optional)

    Returns:
        Dictionary mapping component names to (file_path, success, error_message) tuples
    """
    results = {}

    # Download optimizer checkpoint
    print("📥 Downloading optimizer checkpoint...")
    results["optimizer"] = download_optimizer_checkpoint(repo_id, branch, local_dir)

    # Download config file
    print("📥 Downloading config.json...")
    results["config"] = download_config_file(repo_id, branch, local_dir)

    # Download weights if requested
    if include_weights:
        from .weights import discover_model_weight_files, download_model_weights

        if weight_files is None:
            weight_files = discover_model_weight_files(repo_id, branch)

        results["weights"] = {}
        for weight_file in weight_files:
            print(f"📥 Downloading {weight_file}...")
            results["weights"][weight_file] = download_model_weights(
                repo_id, branch, weight_file, local_dir
            )

    return results


def analyze_complete_checkpoint(
    repo_id: str,
    branch: str,
    include_weights: bool = False,
    weight_files: Optional[List[str]] = None,
    delete_weights_after: bool = False,
) -> Dict[str, Any]:
    """Complete comprehensive checkpoint analysis.

    Args:
        repo_id: HuggingFace repository ID
        branch: Branch name to analyze
        include_weights: Whether to analyze model weights
        weight_files: Specific weight files to analyze (optional)
        delete_weights_after: Delete weight files after analysis

    Returns:
        Dictionary with complete checkpoint analysis
    """
    print(f"🔍 Comprehensive checkpoint analysis: {branch}")

    # Base analysis structure
    analysis = {
        "branch": branch,
        "step": extract_step_from_branch(branch),
        "components": {
            "optimizer": {"available": False},
            "config": {"available": False},
            "weights": {"available": False},
        },
    }

    # Download and analyze optimizer
    print("\n📊 OPTIMIZER ANALYSIS:")
    optimizer_path, optimizer_success, optimizer_error = download_optimizer_checkpoint(
        repo_id, branch
    )
    analysis["components"]["optimizer"]["available"] = optimizer_success

    if optimizer_success and optimizer_path:
        optimizer_analysis = analyze_optimizer_checkpoint(optimizer_path)
        analysis["components"]["optimizer"].update(optimizer_analysis)
    else:
        analysis["components"]["optimizer"]["error"] = optimizer_error

    # Download and analyze config
    print("\n⚙️  CONFIG ANALYSIS:")
    config_path, config_success, config_error = download_config_file(repo_id, branch)
    analysis["components"]["config"]["available"] = config_success

    if config_success and config_path:
        config_analysis = analyze_model_config(config_path)
        analysis["components"]["config"].update(config_analysis)
    else:
        analysis["components"]["config"]["error"] = config_error

    # Download and analyze weights if requested
    if include_weights:
        print("\n🏋️  WEIGHTS ANALYSIS:")
        weights_analysis = analyze_model_weights(
            repo_id, branch, weight_files, delete_weights_after
        )
        analysis["components"]["weights"] = weights_analysis

    return analysis


def process_single_checkpoint(
    repo_id: str,
    branch: str,
    include_weights: bool = False,
    delete_weights_after: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Process a single checkpoint with comprehensive analysis.

    Args:
        repo_id: HuggingFace repository ID
        branch: Branch name to process
        include_weights: Whether to analyze model weights
        delete_weights_after: Delete weight files after analysis

    Returns:
        Tuple of (branch_name, analysis_dict)
    """
    try:
        print(f"Processing checkpoint: {branch}")
        analysis = analyze_complete_checkpoint(
            repo_id, branch, include_weights, delete_weights_after=delete_weights_after
        )
        return branch, analysis

    except Exception as e:
        error_analysis = {
            "branch": branch,
            "step": extract_step_from_branch(branch),
            "error": str(e),
            "components": {
                "optimizer": {"available": False, "error": str(e)},
                "config": {"available": False, "error": str(e)},
                "weights": {"available": False, "error": str(e)},
            },
        }
        print(f"❌ Error processing {branch}: {e}")
        return branch, error_analysis


def process_all_checkpoints(
    repo_id: str,
    max_workers: int = 4,
    include_weights: bool = False,
    delete_weights_after: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Process all checkpoints in parallel with comprehensive analysis.

    Args:
        repo_id: HuggingFace repository ID
        max_workers: Maximum number of parallel downloads
        include_weights: Whether to analyze model weights
        delete_weights_after: Delete weight files after analysis

    Returns:
        Dictionary mapping branch names to their analyses
    """
    # Use library function to get checkpoint branches
    branches = get_checkpoint_branches(repo_id)

    if not branches:
        print("No checkpoint branches found")
        return {}

    print(f"Processing {len(branches)} checkpoints with {max_workers} workers...")

    results = {}

    # Process checkpoints in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_branch = {
            executor.submit(
                process_single_checkpoint,
                repo_id,
                branch,
                include_weights,
                delete_weights_after,
            ): branch
            for branch in branches
        }

        # Collect results as they complete
        for future in as_completed(future_to_branch):
            branch, analysis = future.result()
            results[branch] = analysis

    return results


def create_comprehensive_summary(
    all_analyses: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Create a comprehensive DataFrame summary across all checkpoints.

    Args:
        all_analyses: Dictionary of all checkpoint analyses

    Returns:
        DataFrame with comprehensive checkpoint progression
    """
    summary_data = []

    for branch, analysis in all_analyses.items():
        if "error" in analysis:
            continue

        row = {
            "branch": branch,
            "step": analysis.get("step", 0),
            "optimizer_available": False,
            "config_available": False,
            "weights_available": False,
            "current_lr": None,
            "weight_decay": None,
            "momentum": None,
            "num_param_groups": None,
            "model_type": None,
            "hidden_size": None,
            "num_layers": None,
            "vocab_size": None,
            "estimated_params_millions": None,
            "total_weight_params_millions": None,
            "optimizer_error": "",
            "config_error": "",
            "weights_error": "",
        }

        # Extract component availability
        components = analysis.get("components", {})

        # Optimizer information
        optimizer_comp = components.get("optimizer", {})
        row["optimizer_available"] = optimizer_comp.get("available", False)
        if row["optimizer_available"]:
            lr_info = optimizer_comp.get("learning_rate_info", {})
            if "param_group_lrs" in lr_info and lr_info["param_group_lrs"]:
                first_group = lr_info["param_group_lrs"][0]
                row["current_lr"] = first_group.get("current_lr")
                row["weight_decay"] = first_group.get("weight_decay")
                row["momentum"] = first_group.get("momentum")
                row["num_param_groups"] = len(lr_info["param_group_lrs"])
        else:
            row["optimizer_error"] = optimizer_comp.get("error", "")

        # Config information
        config_comp = components.get("config", {})
        row["config_available"] = config_comp.get("available", False)
        if row["config_available"]:
            arch_info = config_comp.get("architecture_info", {})
            row["model_type"] = arch_info.get("model_type")
            row["hidden_size"] = arch_info.get("hidden_size")
            row["num_layers"] = arch_info.get("num_layers")
            row["vocab_size"] = arch_info.get("vocab_size")
            if "estimated_parameters" in arch_info:
                est_params = arch_info["estimated_parameters"]
                if "estimated_total_millions" in est_params:
                    row["estimated_params_millions"] = est_params[
                        "estimated_total_millions"
                    ]
        else:
            row["config_error"] = config_comp.get("error", "")

        # Weights information
        weights_comp = components.get("weights", {})
        row["weights_available"] = weights_comp.get("weights_available", False)
        if row["weights_available"]:
            summary = weights_comp.get("summary", {})
            row["total_weight_params_millions"] = summary.get(
                "total_parameters_millions"
            )
        else:
            row["weights_error"] = weights_comp.get("error", "")

        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    return df.sort_values("step") if not df.empty else df


def create_learning_rate_summary(
    all_analyses: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Create a learning rate focused DataFrame for backward compatibility.

    Args:
        all_analyses: Dictionary of all checkpoint analyses

    Returns:
        DataFrame with learning rate progression
    """
    comprehensive_df = create_comprehensive_summary(all_analyses)

    # Select only learning rate relevant columns
    lr_columns = [
        "branch",
        "step",
        "optimizer_available",
        "current_lr",
        "weight_decay",
        "momentum",
        "num_param_groups",
        "optimizer_error",
    ]

    return (
        comprehensive_df[lr_columns] if not comprehensive_df.empty else comprehensive_df
    )


def generate_analysis_filename(branch: str) -> str:
    """Generate standardized filename for checkpoint analysis JSON.

    Args:
        branch: Branch name (e.g., "step1250-seed-default")

    Returns:
        Sanitized filename for JSON output
    """
    sanitized = branch.replace("/", "_").replace("-", "_")
    return f"optimizer_analysis_{sanitized}.json"


def save_checkpoint_analysis(
    analysis: Dict[str, Any], branch: str, output_dir: Optional[str] = None
) -> str:
    """Save checkpoint analysis to JSON file.

    Args:
        analysis: Analysis dictionary to save
        branch: Branch name for filename generation
        output_dir: Optional output directory (defaults to current directory)

    Returns:
        Path to saved file
    """
    filename = generate_analysis_filename(branch)
    if output_dir:
        filepath = Path(output_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
    else:
        filepath = Path(filename)

    with open(filepath, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    return str(filepath)


def print_unavailable_checkpoint_info(branch: str, error_msg: str) -> None:
    """Pretty print information about unavailable checkpoint.

    Args:
        branch: Branch name
        error_msg: Error message explaining why checkpoint is unavailable
    """
    print(f"\n⚠️  CHECKPOINT ANALYSIS - {branch}")
    print("=" * (len(branch) + 25))
    print(f"📊 Branch: {branch}")
    print(f"📈 Step: {extract_step_from_branch(branch)}")
    print("❌ Optimizer Status: Not Available")
    print(f"💬 Reason: {error_msg}")


def analyze_single_checkpoint_with_reporting(
    repo_id: str,
    branch: str,
    include_weights: bool = False,
    delete_weights_after: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Complete single checkpoint analysis workflow with comprehensive reporting and saving.

    Args:
        repo_id: HuggingFace repository ID
        branch: Branch name to analyze
        include_weights: Whether to analyze model weights
        delete_weights_after: Delete weight files after analysis
        output_dir: Optional output directory for JSON file

    Returns:
        Complete analysis dictionary
    """
    print(f"🚀 Comprehensive checkpoint analysis: {repo_id} (branch: {branch})")

    # Perform comprehensive analysis
    analysis = analyze_complete_checkpoint(
        repo_id, branch, include_weights, delete_weights_after=delete_weights_after
    )

    # Print detailed reports for each component
    print_comprehensive_analysis(analysis, branch)

    # Save analysis
    output_file = save_checkpoint_analysis(analysis, branch, output_dir)
    print(f"\n💾 Complete analysis saved to: {output_file}")

    return analysis


def print_comprehensive_analysis(
    analysis: Dict[str, Any], branch_name: str = None
) -> None:
    """Print comprehensive analysis results for all components.

    Args:
        analysis: Complete checkpoint analysis dictionary
        branch_name: Branch name for header (optional)
    """
    components = analysis.get("components", {})

    # Print optimizer analysis
    optimizer_comp = components.get("optimizer", {})
    if optimizer_comp.get("available", False):
        print_optimizer_analysis(optimizer_comp, branch_name)
    else:
        print_unavailable_checkpoint_info(
            branch_name or "unknown", optimizer_comp.get("error", "")
        )

    # Print config analysis
    config_comp = components.get("config", {})
    if config_comp.get("available", False):
        print_config_analysis(config_comp, branch_name)
    else:
        print_unavailable_config_info(
            branch_name or "unknown", config_comp.get("error", "")
        )

    # Print weights analysis if available
    weights_comp = components.get("weights", {})
    if weights_comp.get("weights_available", False):
        print_weight_analysis(weights_comp, branch_name)
    elif "error" in weights_comp:
        print_unavailable_weights_info(
            branch_name or "unknown", weights_comp.get("error", "")
        )


def print_unavailable_config_info(branch: str, error_msg: str) -> None:
    """Pretty print information about unavailable config.

    Args:
        branch: Branch name
        error_msg: Error message explaining why config is unavailable
    """
    print(f"\n⚠️  CONFIG ANALYSIS - {branch}")
    print("=" * (len(branch) + 20))
    print(f"📊 Branch: {branch}")
    print(f"📈 Step: {extract_step_from_branch(branch)}")
    print("❌ Config Status: Not Available")
    print(f"💬 Reason: {error_msg}")


def print_unavailable_weights_info(branch: str, error_msg: str) -> None:
    """Pretty print information about unavailable weights.

    Args:
        branch: Branch name
        error_msg: Error message explaining why weights are unavailable
    """
    print(f"\n⚠️  WEIGHTS ANALYSIS - {branch}")
    print("=" * (len(branch) + 20))
    print(f"📊 Branch: {branch}")
    print(f"📈 Step: {extract_step_from_branch(branch)}")
    print("❌ Weights Status: Not Available")
    print(f"💬 Reason: {error_msg}")


def save_all_analyses_outputs(
    all_analyses: Dict[str, Dict[str, Any]], output_dir: Optional[str] = None
) -> Tuple[str, str, str]:
    """Save all checkpoint analyses to comprehensive CSV summary and JSON files.

    Args:
        all_analyses: Dictionary of all checkpoint analyses
        output_dir: Optional output directory (defaults to current directory)

    Returns:
        Tuple of (comprehensive_csv_path, lr_csv_path, json_path) for saved files
    """
    from pathlib import Path

    # Determine output paths
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        comprehensive_csv_path = output_path / "comprehensive_checkpoint_summary.csv"
        lr_csv_path = output_path / "learning_rate_summary.csv"
        json_path = output_path / "all_checkpoint_analyses.json"
    else:
        comprehensive_csv_path = Path("comprehensive_checkpoint_summary.csv")
        lr_csv_path = Path("learning_rate_summary.csv")
        json_path = Path("all_checkpoint_analyses.json")

    # Create and save comprehensive summary
    comprehensive_summary = create_comprehensive_summary(all_analyses)
    if not comprehensive_summary.empty:
        comprehensive_summary.to_csv(comprehensive_csv_path, index=False)
        print(f"💾 Comprehensive summary saved to: {comprehensive_csv_path}")

    # Create and save learning rate summary for compatibility
    lr_summary = create_learning_rate_summary(all_analyses)
    if not lr_summary.empty:
        lr_summary.to_csv(lr_csv_path, index=False)
        print(f"💾 Learning rate summary saved to: {lr_csv_path}")

    # Save full analyses to JSON
    with open(json_path, "w") as f:
        json.dump(all_analyses, f, indent=2, default=str)
    print(f"💾 Full analyses saved to: {json_path}")

    return str(comprehensive_csv_path), str(lr_csv_path), str(json_path)


def print_analysis_summary(all_analyses: Dict[str, Dict[str, Any]]) -> None:
    """Print summary statistics and learning rate progression.

    Args:
        all_analyses: Dictionary of all checkpoint analyses
    """
    print(f"\n🎉 Successfully processed {len(all_analyses)} checkpoints")

    # Create and display learning rate summary
    lr_summary = create_learning_rate_summary(all_analyses)

    if not lr_summary.empty:
        print("\n📊 LEARNING RATE PROGRESSION SUMMARY:")
        print(lr_summary.to_string(index=False, float_format="%.2e"))
    else:
        print("\n⚠️  No learning rate data available in any checkpoints")


def print_detailed_sample_analyses(
    all_analyses: Dict[str, Dict[str, Any]], sample_count: int = 3
) -> None:
    """Print detailed analysis for first N checkpoints (by step number).

    Args:
        all_analyses: Dictionary of all checkpoint analyses
        sample_count: Number of checkpoints to show detailed analysis for
    """
    sorted_branches = sorted(all_analyses.keys(), key=extract_step_from_branch)
    print(
        f"\n📋 Detailed analysis for first {min(sample_count, len(sorted_branches))} checkpoints:"
    )

    shown_count = 0
    for branch in sorted_branches:
        if shown_count >= sample_count:
            break
        if "error" not in all_analyses[branch]:
            print_optimizer_analysis(all_analyses[branch], branch)
            shown_count += 1

    if shown_count == 0:
        print("⚠️  No successful checkpoint analyses to show")


def process_all_checkpoints_with_reporting(
    repo_id: str,
    max_workers: int = 6,
    include_weights: bool = False,
    delete_weights_after: bool = False,
    output_dir: Optional[str] = None,
    sample_count: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """Complete batch checkpoint processing workflow with comprehensive reporting and file output.

    Args:
        repo_id: HuggingFace repository ID
        max_workers: Maximum number of parallel downloads
        include_weights: Whether to analyze model weights
        delete_weights_after: Delete weight files after analysis
        output_dir: Optional output directory for files
        sample_count: Number of detailed analyses to show

    Returns:
        Dictionary of all checkpoint analyses
    """
    print(f"🚀 Discovering and analyzing ALL checkpoints from {repo_id}")
    if include_weights:
        print(
            "⚠️  Including model weights analysis - this may take significant time and storage!"
        )

    # Process all checkpoints
    all_analyses = process_all_checkpoints(
        repo_id,
        max_workers=max_workers,
        include_weights=include_weights,
        delete_weights_after=delete_weights_after,
    )

    if not all_analyses:
        print("No checkpoints found to analyze")
        return {}

    # Print summary statistics
    print_analysis_summary(all_analyses)

    # Save output files
    save_all_analyses_outputs(all_analyses, output_dir)

    # Show detailed analyses for sample checkpoints
    print_detailed_sample_analyses(all_analyses, sample_count)

    return all_analyses


def print_optimizer_analysis(analysis: Dict[str, Any], branch_name: str = None) -> None:
    """Pretty print the optimizer analysis results."""
    header = (
        f"OPTIMIZER CHECKPOINT ANALYSIS - {branch_name}"
        if branch_name
        else "OPTIMIZER CHECKPOINT ANALYSIS"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
        return

    if "step" in analysis:
        print(f"📍 Step: {analysis['step']}")
    print(f"📊 Checkpoint Type: {analysis['checkpoint_type']}")
    print(f"🔑 Top-level Keys: {', '.join(analysis['checkpoint_keys'])}")

    print("\n📈 LEARNING RATE INFORMATION:")
    lr_info = analysis["learning_rate_info"]
    if "param_group_lrs" in lr_info:
        for group_info in lr_info["param_group_lrs"]:
            print(f"  Group {group_info['group_index']}:")
            if "current_lr" in group_info:
                print(f"    Current LR: {group_info['current_lr']}")
            if "initial_lr" in group_info:
                print(f"    Initial LR: {group_info['initial_lr']}")
            if "weight_decay" in group_info:
                print(f"    Weight Decay: {group_info['weight_decay']}")
            if "momentum" in group_info:
                print(f"    Momentum: {group_info['momentum']}")

    if lr_info.get("lr_scheduler_present"):
        print("  📅 LR Scheduler: Present")
        if "scheduler_last_epoch" in lr_info:
            print(f"    Last Epoch: {lr_info['scheduler_last_epoch']}")

    print("\n🔧 OPTIMIZER COMPONENTS:")
    for comp_name, comp_info in analysis["optimizer_info"].items():
        print(f"  {comp_name}:")
        print(f"    Type: {comp_info.get('type', 'Unknown')}")
        print(f"    Size: {comp_info.get('size', 'Unknown')}")

        if comp_name == "param_groups":
            print(f"    Num Groups: {comp_info.get('num_param_groups', 'Unknown')}")
            if "learning_rate" in comp_info:
                print(f"    Learning Rate: {comp_info['learning_rate']}")

        elif comp_name == "state":
            print(f"    Num Parameters: {comp_info.get('num_parameters', 'Unknown')}")
            if "state_keys" in comp_info:
                print(f"    State Keys: {', '.join(comp_info['state_keys'])}")
