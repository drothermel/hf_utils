"""Core branch discovery and parsing functions for HuggingFace repositories."""

import re
from typing import Any, Dict, List, Tuple

from huggingface_hub import list_repo_refs


def get_all_repo_branches(repo_id: str) -> List[str]:
    """Get all branch names from a HuggingFace repository.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        List of all branch names
    """
    refs = list_repo_refs(repo_id)
    return [ref.name for ref in refs.branches]


def is_checkpoint_branch(branch: str) -> bool:
    """Check if branch name matches checkpoint pattern.

    Args:
        branch: Branch name to check

    Returns:
        True if branch follows step-seed pattern
    """
    pattern = re.compile(r"^step\d+-seed-.+$")
    return bool(pattern.match(branch))


def get_checkpoint_branches(repo_id: str) -> List[str]:
    """Get all checkpoint branches from a repository.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        List of checkpoint branch names, sorted by step number
    """
    all_branches = get_all_repo_branches(repo_id)
    checkpoint_branches = [b for b in all_branches if is_checkpoint_branch(b)]
    return sort_branches_by_step(checkpoint_branches)


def parse_branch_name(branch: str) -> Dict[str, Any]:
    """Parse branch name into components.

    Args:
        branch: Branch name to parse

    Returns:
        Dictionary with step, seed, and validity info
    """
    result = {"branch": branch, "valid": False, "step": None, "seed": None}

    if not is_checkpoint_branch(branch):
        return result

    # Extract step number
    step_match = re.search(r"step(\d+)-", branch)
    if step_match:
        result["step"] = int(step_match.group(1))

    # Extract seed configuration
    seed_match = re.search(r"seed-(.+)$", branch)
    if seed_match:
        result["seed"] = seed_match.group(1)

    result["valid"] = result["step"] is not None and result["seed"] is not None
    return result


def extract_step_from_branch(branch: str) -> int:
    """Extract step number from branch name.

    Args:
        branch: Branch name

    Returns:
        Step number, or 0 if not found
    """
    parsed = parse_branch_name(branch)
    return parsed["step"] or 0


def extract_seed_from_branch(branch: str) -> str:
    """Extract seed configuration from branch name.

    Args:
        branch: Branch name

    Returns:
        Seed configuration string, or "unknown" if not found
    """
    parsed = parse_branch_name(branch)
    return parsed["seed"] or "unknown"


def sort_branches_by_step(branches: List[str]) -> List[str]:
    """Sort branches by their step number.

    Args:
        branches: List of branch names

    Returns:
        Sorted list of branch names
    """
    return sorted(branches, key=extract_step_from_branch)


def group_branches_by_seed(branches: List[str]) -> Dict[str, List[str]]:
    """Group branches by their seed configuration.

    Args:
        branches: List of branch names

    Returns:
        Dictionary mapping seed configs to branch lists
    """
    groups = {}
    for branch in branches:
        seed = extract_seed_from_branch(branch)
        if seed not in groups:
            groups[seed] = []
        groups[seed].append(branch)

    # Sort branches within each group by step
    for seed in groups:
        groups[seed] = sort_branches_by_step(groups[seed])

    return groups


def get_step_range_for_seed(branches: List[str]) -> Tuple[int, int]:
    """Get the step range for a list of branches.

    Args:
        branches: List of branch names (should be same seed)

    Returns:
        Tuple of (min_step, max_step)
    """
    if not branches:
        return (0, 0)

    steps = [extract_step_from_branch(b) for b in branches]
    return (min(steps), max(steps))


def create_branch_metadata(repo_id: str) -> Dict[str, Any]:
    """Create comprehensive branch metadata for a repository.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        Structured metadata dictionary
    """
    from datetime import datetime, timezone

    # Get all branch data
    all_branches = get_all_repo_branches(repo_id)
    checkpoint_branches = [b for b in all_branches if is_checkpoint_branch(b)]
    other_branches = [b for b in all_branches if not is_checkpoint_branch(b)]

    # Group checkpoint branches by seed
    seed_groups = group_branches_by_seed(checkpoint_branches)

    # Build seed configurations
    seed_configurations = {}
    for seed, branches in seed_groups.items():
        step_range = get_step_range_for_seed(branches)
        seed_configurations[seed] = {
            "count": len(branches),
            "step_range": list(step_range),
            "branches": [
                {"step": extract_step_from_branch(b), "branch": b} for b in branches
            ],
        }

    return {
        "repo_id": repo_id,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_branches": len(all_branches),
        "checkpoint_branches": len(checkpoint_branches),
        "seed_configurations": seed_configurations,
        "other_branches": sorted(other_branches),
        "all_checkpoint_branches": sorted(
            checkpoint_branches, key=extract_step_from_branch
        ),
    }
