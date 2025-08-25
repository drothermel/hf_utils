"""Model weights downloading and analysis functions."""

import torch
import os
from huggingface_hub import hf_hub_download, list_repo_files
from typing import Dict, Any, Optional, List, Tuple


def discover_model_weight_files(repo_id: str, branch: str = "main") -> List[str]:
    """Discover available model weight files in the repository.

    Args:
        repo_id: HuggingFace repository ID
        branch: Branch/revision to check

    Returns:
        List of weight file names found
    """
    try:
        all_files = list_repo_files(repo_id=repo_id, revision=branch)

        # Common weight file patterns
        weight_patterns = [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model-00001-of-00001.bin",
        ]

        # Find files matching weight patterns
        weight_files = []
        for file in all_files:
            if any(pattern in file for pattern in weight_patterns):
                weight_files.append(file)
            # Also catch sharded models
            elif ("pytorch_model-" in file and file.endswith(".bin")) or file.endswith(
                ".safetensors"
            ):
                weight_files.append(file)

        # Sort by filename for consistent ordering
        return sorted(weight_files)

    except Exception as e:
        print(f"❌ Error discovering weight files: {e}")
        return []


def download_model_weights(
    repo_id: str, branch: str, filename: str, local_dir: Optional[str] = None
) -> Tuple[Optional[str], bool, str]:
    """Download a specific model weight file.

    Args:
        repo_id: HuggingFace repository ID
        branch: Branch/revision to download from
        filename: Name of the weight file to download
        local_dir: Local directory to save file (optional)

    Returns:
        Tuple of (file_path, success, error_message)
    """
    try:
        file_path = hf_hub_download(
            repo_id=repo_id, filename=filename, revision=branch, local_dir=local_dir
        )
        print(f"✅ Downloaded {filename} to: {file_path}")
        return file_path, True, ""
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Entry Not Found" in error_msg:
            print(f"⚠️  Weight file {filename} not found for branch {branch}")
            return None, False, f"Weight file {filename} not available"
        else:
            print(f"❌ Error downloading {filename}: {e}")
            return None, False, error_msg


def calculate_weight_statistics(weight_path: str) -> Dict[str, Any]:
    """Calculate comprehensive statistics for model weights.

    Args:
        weight_path: Path to the weight file

    Returns:
        Dictionary containing weight statistics
    """
    try:
        # Load the weights
        if weight_path.endswith(".safetensors"):
            # Handle safetensors format
            try:
                from safetensors import safe_open

                weights = {}
                with safe_open(weight_path, framework="pt") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
            except ImportError:
                return {"error": "safetensors library not available"}
        else:
            # Handle PyTorch .bin format
            weights = torch.load(weight_path, map_location="cpu")

        if not isinstance(weights, dict):
            return {"error": "Weights file is not a dictionary"}

        stats = {
            "file_path": weight_path,
            "file_size_mb": round(os.path.getsize(weight_path) / (1024 * 1024), 2),
            "num_tensors": len(weights),
            "tensor_info": {},
            "parameter_stats": {},
            "layer_analysis": analyze_layer_structure(weights),
        }

        # Analyze each tensor
        total_params = 0
        tensor_stats = []

        for name, tensor in weights.items():
            if torch.is_tensor(tensor):
                tensor_params = tensor.numel()
                total_params += tensor_params

                tensor_info = {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "parameters": tensor_params,
                    "size_mb": round(
                        tensor.numel() * tensor.element_size() / (1024 * 1024), 3
                    ),
                    "statistics": calculate_tensor_stats(tensor),
                }
                tensor_stats.append(tensor_info)

        stats["tensor_info"] = tensor_stats
        stats["parameter_stats"] = {
            "total_parameters": total_params,
            "total_parameters_millions": round(total_params / 1_000_000, 2),
            "total_parameters_billions": round(total_params / 1_000_000_000, 3),
        }

        # Overall distribution analysis
        stats["global_statistics"] = calculate_global_weight_stats(weights)

        return stats

    except Exception as e:
        return {"error": f"Failed to analyze weights: {str(e)}"}


def calculate_tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    """Calculate statistics for a single tensor.

    Args:
        tensor: PyTorch tensor to analyze

    Returns:
        Dictionary with tensor statistics
    """
    try:
        # Convert to float for calculations
        flat_tensor = tensor.flatten().float()

        stats = {
            "mean": float(torch.mean(flat_tensor)),
            "std": float(torch.std(flat_tensor)),
            "min": float(torch.min(flat_tensor)),
            "max": float(torch.max(flat_tensor)),
            "median": float(torch.median(flat_tensor)),
            "abs_mean": float(torch.mean(torch.abs(flat_tensor))),
            "zero_fraction": float(torch.sum(flat_tensor == 0.0) / flat_tensor.numel()),
        }

        # Add percentiles
        percentiles = [25, 75, 90, 95, 99]
        for p in percentiles:
            stats[f"percentile_{p}"] = float(torch.quantile(flat_tensor, p / 100.0))

        return stats
    except Exception as e:
        return {"error": f"Failed to calculate tensor stats: {str(e)}"}


def analyze_layer_structure(weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Analyze the layer structure from weight names.

    Args:
        weights: Dictionary of weight tensors

    Returns:
        Dictionary with layer structure analysis
    """
    layer_info = {
        "embedding_layers": [],
        "transformer_layers": [],
        "output_layers": [],
        "layer_norm_layers": [],
        "attention_layers": [],
        "feedforward_layers": [],
        "other_layers": [],
    }

    layer_counts = {
        "total_layers": 0,
        "attention_heads": 0,
        "feedforward_layers": 0,
        "layer_norms": 0,
    }

    for name, tensor in weights.items():
        if not torch.is_tensor(tensor):
            continue

        name_lower = name.lower()

        # Categorize layers
        if "embed" in name_lower:
            layer_info["embedding_layers"].append(name)
        elif (
            "ln" in name_lower
            or "layer_norm" in name_lower
            or "layernorm" in name_lower
        ):
            layer_info["layer_norm_layers"].append(name)
            layer_counts["layer_norms"] += 1
        elif any(
            attn_key in name_lower
            for attn_key in ["attn", "attention", "self_attention"]
        ):
            layer_info["attention_layers"].append(name)
        elif any(
            ff_key in name_lower
            for ff_key in ["mlp", "ffn", "feed_forward", "intermediate"]
        ):
            layer_info["feedforward_layers"].append(name)
            layer_counts["feedforward_layers"] += 1
        elif "lm_head" in name_lower or "output" in name_lower:
            layer_info["output_layers"].append(name)
        elif "transformer" in name_lower or "layer" in name_lower:
            layer_info["transformer_layers"].append(name)
        else:
            layer_info["other_layers"].append(name)

    # Estimate number of transformer layers
    transformer_layers = set()
    for name in weights.keys():
        # Look for layer indices in names like "transformer.h.0", "model.layers.0", etc.
        import re

        layer_matches = re.findall(r"(?:layer|h)\.(\d+)\.", name.lower())
        if layer_matches:
            transformer_layers.update(int(match) for match in layer_matches)

    layer_counts["estimated_transformer_layers"] = len(transformer_layers)
    layer_counts["total_layers"] = len(weights)

    return {
        "layer_categorization": layer_info,
        "layer_counts": layer_counts,
        "transformer_layer_indices": sorted(list(transformer_layers))
        if transformer_layers
        else [],
    }


def calculate_global_weight_stats(weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Calculate global statistics across all model weights.

    Args:
        weights: Dictionary of all weight tensors

    Returns:
        Dictionary with global weight statistics
    """
    try:
        # Collect all weights into a single tensor for global stats
        all_weights = []
        for tensor in weights.values():
            if torch.is_tensor(tensor):
                all_weights.append(tensor.flatten().float())

        if not all_weights:
            return {"error": "No tensors found"}

        global_tensor = torch.cat(all_weights)

        global_stats = {
            "global_mean": float(torch.mean(global_tensor)),
            "global_std": float(torch.std(global_tensor)),
            "global_min": float(torch.min(global_tensor)),
            "global_max": float(torch.max(global_tensor)),
            "global_abs_mean": float(torch.mean(torch.abs(global_tensor))),
            "global_zero_fraction": float(
                torch.sum(global_tensor == 0.0) / global_tensor.numel()
            ),
        }

        # Global percentiles
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        for p in percentiles:
            global_stats[f"global_percentile_{p}"] = float(
                torch.quantile(global_tensor, p / 100.0)
            )

        return global_stats
    except Exception as e:
        return {"error": f"Failed to calculate global stats: {str(e)}"}


def analyze_model_weights(
    repo_id: str,
    branch: str,
    weight_files: Optional[List[str]] = None,
    delete_after_analysis: bool = False,
) -> Dict[str, Any]:
    """Complete model weights analysis workflow.

    Args:
        repo_id: HuggingFace repository ID
        branch: Branch to analyze
        weight_files: Specific weight files to analyze (optional, discovers all if None)
        delete_after_analysis: Whether to delete weight files after analysis

    Returns:
        Dictionary with complete weights analysis
    """
    print(f"🔍 Analyzing model weights for {repo_id} (branch: {branch})")

    # Discover weight files if not specified
    if weight_files is None:
        weight_files = discover_model_weight_files(repo_id, branch)

    if not weight_files:
        return {
            "weights_available": False,
            "error": "No weight files found",
            "discovered_files": [],
        }

    print(f"📁 Found {len(weight_files)} weight file(s): {', '.join(weight_files)}")

    analysis = {
        "weights_available": True,
        "discovered_files": weight_files,
        "file_analyses": {},
        "summary": {},
    }

    total_params = 0
    total_size_mb = 0
    downloaded_files = []

    try:
        # Analyze each weight file
        for weight_file in weight_files:
            print(f"\n📊 Analyzing {weight_file}...")

            # Download the file
            file_path, success, error_msg = download_model_weights(
                repo_id, branch, weight_file
            )

            if success and file_path:
                downloaded_files.append(file_path)

                # Calculate statistics
                file_stats = calculate_weight_statistics(file_path)
                analysis["file_analyses"][weight_file] = file_stats

                # Accumulate totals
                if "parameter_stats" in file_stats:
                    total_params += file_stats["parameter_stats"].get(
                        "total_parameters", 0
                    )
                if "file_size_mb" in file_stats:
                    total_size_mb += file_stats["file_size_mb"]
            else:
                analysis["file_analyses"][weight_file] = {
                    "error": error_msg,
                    "downloaded": False,
                }

        # Create summary
        analysis["summary"] = {
            "total_files_analyzed": len(
                [f for f in analysis["file_analyses"].values() if "error" not in f]
            ),
            "total_parameters": total_params,
            "total_parameters_millions": round(total_params / 1_000_000, 2),
            "total_parameters_billions": round(total_params / 1_000_000_000, 3),
            "total_size_mb": round(total_size_mb, 2),
            "total_size_gb": round(total_size_mb / 1024, 3),
        }

    finally:
        # Clean up downloaded files if requested
        if delete_after_analysis and downloaded_files:
            print(
                f"\n🗑️  Cleaning up {len(downloaded_files)} downloaded weight files..."
            )
            for file_path in downloaded_files:
                try:
                    os.remove(file_path)
                    print(f"✅ Deleted: {file_path}")
                except Exception as e:
                    print(f"⚠️  Failed to delete {file_path}: {e}")

    return analysis


def print_weight_analysis(analysis: Dict[str, Any], branch_name: str = None) -> None:
    """Pretty print the weights analysis results.

    Args:
        analysis: Weights analysis dictionary
        branch_name: Branch name for header (optional)
    """
    header = (
        f"MODEL WEIGHTS ANALYSIS - {branch_name}"
        if branch_name
        else "MODEL WEIGHTS ANALYSIS"
    )
    print(f"\n{'=' * len(header)}")
    print(header)
    print("=" * len(header))

    if not analysis.get("weights_available", True):
        print(f"❌ No weights available: {analysis.get('error', 'Unknown reason')}")
        return

    # Summary statistics
    summary = analysis.get("summary", {})
    if summary:
        print("\n📊 SUMMARY:")
        print(f"  Files Analyzed: {summary.get('total_files_analyzed', 0)}")
        print(
            f"  Total Parameters: {summary.get('total_parameters_billions', 0):.2f}B ({summary.get('total_parameters', 0):,})"
        )
        print(
            f"  Total Size: {summary.get('total_size_gb', 0):.2f} GB ({summary.get('total_size_mb', 0):.1f} MB)"
        )

    # Per-file analysis
    file_analyses = analysis.get("file_analyses", {})
    if file_analyses:
        print(f"\n📁 FILE ANALYSES ({len(file_analyses)} files):")

        for filename, file_analysis in file_analyses.items():
            if "error" in file_analysis:
                print(f"\n  ❌ {filename}: {file_analysis['error']}")
                continue

            print(f"\n  📄 {filename}:")
            print(f"    Size: {file_analysis.get('file_size_mb', 0):.1f} MB")
            print(f"    Tensors: {file_analysis.get('num_tensors', 0)}")

            param_stats = file_analysis.get("parameter_stats", {})
            if param_stats:
                print(
                    f"    Parameters: {param_stats.get('total_parameters_millions', 0):.1f}M"
                )

            # Global statistics
            global_stats = file_analysis.get("global_statistics", {})
            if global_stats and "error" not in global_stats:
                print(f"    Global Mean: {global_stats.get('global_mean', 0):.6f}")
                print(f"    Global Std: {global_stats.get('global_std', 0):.6f}")
                print(
                    f"    Zero Fraction: {global_stats.get('global_zero_fraction', 0):.4f}"
                )

            # Layer structure
            layer_analysis = file_analysis.get("layer_analysis", {})
            if layer_analysis:
                layer_counts = layer_analysis.get("layer_counts", {})
                if layer_counts:
                    est_layers = layer_counts.get("estimated_transformer_layers", 0)
                    if est_layers > 0:
                        print(f"    Estimated Transformer Layers: {est_layers}")

    discovered_files = analysis.get("discovered_files", [])
    if discovered_files:
        print(f"\n🔍 DISCOVERED FILES: {', '.join(discovered_files)}")
