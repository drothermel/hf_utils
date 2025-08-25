"""HuggingFace data management and persistence utilities."""

# import warnings
# Suppress urllib3 OpenSSL warnings early
# warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+.*")

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from datasets import load_dataset
import pandas as pd


def load_or_download_dataset(
    path: Path, repo_id: str, split: str = "train"
) -> pd.DataFrame:
    download_dataset(path, repo_id, split, force_reload=False)
    return pd.read_parquet(path)


def download_dataset(
    path: Path, repo_id: str, split: str = "train", force_reload: bool = False
) -> None:
    if force_reload or not path.exists():
        raw_df = load_dataset(repo_id, split="train")
        raw_df.to_parquet(path)


# -------------------- OLD CODE --------------------


def get_data_path(data_type, name):
    repo_name = sanitize_repo_name(name)
    return f"/Users/daniellerothermel/drotherm/repos/ddpred/data/huggingface/{data_type}/{repo_name}.json"


def sanitize_repo_name(repo_id: str) -> str:
    """Convert repository ID to filesystem-safe name.

    Args:
        repo_id: HuggingFace repository ID (e.g., "allenai/model-name")

    Returns:
        Sanitized name (e.g., "allenai--model-name")
    """
    return repo_id.replace("/", "--").replace(" ", "-")


def load_branch_data(repo_id: str) -> Optional[Dict[str, Any]]:
    """Load branch data from persistent storage.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        Branch metadata dictionary, or None if not found
    """
    file_path = get_data_path(data_type="branches", name=repo_id)

    if not file_path.exists():
        return None

    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load branch data for {repo_id}: {e}")
        return None


def save_branch_data(repo_id: str, data: Dict[str, Any]) -> None:
    """Save branch data to persistent storage.

    Args:
        repo_id: HuggingFace repository ID
        data: Branch metadata dictionary to save
    """
    file_path = get_data_path(data_type="branches", name=repo_id)

    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp if not present
    if "last_updated" not in data:
        data["last_updated"] = datetime.now(timezone.utc).isoformat()

    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Update metadata
        update_metadata(repo_id)

    except IOError as e:
        print(f"Warning: Could not save branch data for {repo_id}: {e}")


def get_metadata_path() -> Path:
    """Get path to the metadata file."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "data" / "huggingface" / "branches" / "metadata.json"


def load_metadata() -> Dict[str, Any]:
    """Load the metadata file."""
    metadata_path = get_metadata_path()

    if not metadata_path.exists():
        return {"repositories": {}}

    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"repositories": {}}


def save_metadata(metadata: Dict[str, Any]) -> None:
    """Save the metadata file."""
    metadata_path = get_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save metadata: {e}")


def update_metadata(repo_id: str) -> None:
    """Update metadata with repository information.

    Args:
        repo_id: HuggingFace repository ID
    """
    metadata = load_metadata()

    metadata["repositories"][repo_id] = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "file_path": f"{sanitize_repo_name(repo_id)}.json",
    }

    save_metadata(metadata)


def list_cached_repositories() -> List[str]:
    """List all repositories that have cached branch data.

    Returns:
        List of repository IDs
    """
    metadata = load_metadata()
    return list(metadata.get("repositories", {}).keys())


def get_cache_info(repo_id: str) -> Optional[Dict[str, Any]]:
    """Get cache information for a repository.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        Cache info dictionary or None if not cached
    """
    metadata = load_metadata()
    return metadata.get("repositories", {}).get(repo_id)


def clear_branch_cache(repo_id: str) -> bool:
    """Clear cached branch data for a repository.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        True if cache was cleared, False if no cache existed
    """
    file_path = get_data_path(data_type="branches", name=repo_id)
    if file_path.exists():
        file_path.unlink()

        # Remove from metadata
        metadata = load_metadata()
        if repo_id in metadata.get("repositories", {}):
            del metadata["repositories"][repo_id]
            save_metadata(metadata)

        return True

    return False
