#!/usr/bin/env python3
"""
Build demonstration pool from successful trajectories
Fixed for SWE-agent's actual directory structure
"""
# python build_demonstration_pool.py --results_dir ../results

import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
import argparse


def load_trajectory_file(traj_file: Path) -> Optional[str]:
    """Load trajectory file content"""
    try:
        with open(traj_file, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading {traj_file}: {e}")
        return None


def load_instance_for_demo(instance_dir: Path) -> Optional[Dict]:
    """Load instance data for demonstration pool"""
    
    instance_id = instance_dir.name
    
    # Check for required files
    patch_file = instance_dir / f"{instance_id}.patch"
    traj_file = instance_dir / f"{instance_id}.traj"
    
    if not patch_file.exists():
        return None
    
    # Check if resolved (patch exists and is non-empty)
    patch_size = patch_file.stat().st_size
    if patch_size == 0:
        return None
    
    # Extract repository
    parts = instance_id.split("__")
    repo = parts[0] if len(parts) >= 2 else "unknown"
    
    # Load trajectory if available
    trajectory_content = None
    if traj_file.exists():
        trajectory_content = load_trajectory_file(traj_file)
    
    demonstration = {
        "instance_id": instance_id,
        "repository": repo,
        "resolved": True,
        "patch_file": str(patch_file.name),
        "patch_size": patch_size,
        "trajectory_file": str(traj_file.name) if traj_file.exists() else None,
        "source_directory": str(instance_dir.name)
    }
    
    return demonstration


def build_demonstration_pool(
    results_dir: Path,
    output_dir: Path,
    min_success_rate: float = 1.0
) -> List[Dict]:
    """Build demonstration pool from successful instances"""
    
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all instance directories
    instance_dirs = [d for d in results_dir.iterdir() if d.is_dir() and "__" in d.name]
    
    demonstrations = []
    skipped = 0
    
    print(f"Processing {len(instance_dirs)} instance directories...")
    
    for instance_dir in sorted(instance_dirs):
        demo = load_instance_for_demo(instance_dir)
        if demo is None:
            skipped += 1
            continue
        
        demonstrations.append(demo)
        
        # Copy instance directory to demonstration pool
        dest_dir = output_dir / instance_dir.name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(instance_dir, dest_dir)
    
    print(f"\n✅ Created demonstration pool:")
    print(f"   Total instances processed: {len(instance_dirs)}")
    print(f"   Valid demonstrations:      {len(demonstrations)}")
    print(f"   Skipped:                   {skipped}")
    
    # Save demonstration index
    index_file = output_dir / "demonstration_index.json"
    with open(index_file, 'w') as f:
        json.dump({
            "total_demonstrations": len(demonstrations),
            "demonstrations": demonstrations
        }, f, indent=2)
    
    print(f"\n💾 Demonstration pool saved to: {output_dir}")
    print(f"   Index file: {index_file}")
    
    return demonstrations


def print_pool_statistics(demonstrations: List[Dict]):
    """Print statistics about demonstration pool"""
    
    print("\n" + "="*80)
    print("DEMONSTRATION POOL STATISTICS")
    print("="*80)
    
    print(f"\n📊 OVERALL:")
    print(f"   Total demonstrations: {len(demonstrations)}")
    
    # Repository distribution
    repos = [d["repository"] for d in demonstrations]
    repo_counts = Counter(repos)
    
    print(f"\n📁 REPOSITORY DISTRIBUTION:")
    print(f"   Unique repositories:  {len(repo_counts)}")
    print(f"\n   {'Repository':<30} {'Count':>8} {'Percentage':>12}")
    print("   " + "-"*52)
    
    for repo, count in repo_counts.most_common():
        percentage = count / len(demonstrations) * 100
        print(f"   {repo:<30} {count:>8} {percentage:>11.1f}%")
    
    # Diversity assessment
    print(f"\n📊 DIVERSITY ASSESSMENT:")
    
    # Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    counts = sorted([count for _, count in repo_counts.items()])
    n = len(counts)
    if n > 0:
        cumsum = sum((i+1) * count for i, count in enumerate(counts))
        gini = (2 * cumsum) / (n * sum(counts)) - (n + 1) / n
        
        print(f"   Gini coefficient:     {gini:.3f} (0=equal, 1=unequal)")
        
        if gini < 0.3:
            diversity = "Excellent"
        elif gini < 0.5:
            diversity = "Good"
        elif gini < 0.7:
            diversity = "Fair"
        else:
            diversity = "Poor"
        
        print(f"   Diversity rating:     {diversity}")
    
    # Patch size statistics
    patch_sizes = [d["patch_size"] for d in demonstrations]
    if patch_sizes:
        avg_size = sum(patch_sizes) / len(patch_sizes)
        min_size = min(patch_sizes)
        max_size = max(patch_sizes)
        
        print(f"\n📝 PATCH STATISTICS:")
        print(f"   Average patch size:   {avg_size:.0f} bytes")
        print(f"   Smallest patch:       {min_size} bytes")
        print(f"   Largest patch:        {max_size} bytes")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if len(demonstrations) < 20:
        print(f"   ⚠️  Consider collecting more demonstrations (have {len(demonstrations)}, recommend 20+)")
    else:
        print(f"   ✅ Good number of demonstrations ({len(demonstrations)})")
    
    max_repo = repo_counts.most_common(1)[0]
    if max_repo[1] / len(demonstrations) > 0.5:
        print(f"   ⚠️  {max_repo[0]} is over-represented ({max_repo[1]}/{len(demonstrations)} = {max_repo[1]/len(demonstrations)*100:.1f}%)")
        print(f"       Consider collecting more from other repositories")
    else:
        print(f"   ✅ No single repository dominates (largest: {max_repo[0]} at {max_repo[1]/len(demonstrations)*100:.1f}%)")
    
    repos_with_one = sum(1 for count in repo_counts.values() if count == 1)
    if repos_with_one > 0:
        print(f"   ℹ️  {repos_with_one} repositories have only 1 demonstration")
        print(f"       These may not be representative")
    
    if len(repo_counts) >= 10 and gini < 0.5:
        print(f"   ✅ Excellent diversity! Ready for demonstration selection experiments")
    elif len(repo_counts) >= 5:
        print(f"   ✅ Good diversity! Ready for experiments")
    else:
        print(f"   ⚠️  Limited diversity ({len(repo_counts)} repositories)")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Build demonstration pool from successful trajectories")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory containing instance subdirectories"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demonstration_pool",
        help="Output directory for demonstration pool"
    )
    parser.add_argument(
        "--min_success_rate",
        type=float,
        default=1.0,
        help="Minimum success rate (0-1) for demonstrations"
    )
    
    args = parser.parse_args()
    
    print(f"Building demonstration pool from: {args.results_dir}")
    
    # Build demonstration pool
    demonstrations = build_demonstration_pool(
        Path(args.results_dir),
        Path(args.output_dir),
        args.min_success_rate
    )
    
    if not demonstrations:
        print("\n❌ No successful demonstrations found!")
        print("\nPossible reasons:")
        print("  - No patches were generated successfully")
        print("  - Check that SWE-agent completed successfully")
        print("  - Run analyze_collected_data_fixed.py first to see what was collected")
        return 1
    
    # Print statistics
    print_pool_statistics(demonstrations)
    
    print(f"\n✅ Demonstration pool ready!")
    print(f"   Location: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review the demonstration pool")
    print(f"  2. Run demonstration selection strategies")
    print(f"  3. Start evaluation experiments")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())