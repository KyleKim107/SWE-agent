#!/usr/bin/env python3
"""
Enrich demonstration pool with problem statements from SWE-bench
This is required for TF-IDF similarity to work properly
"""

import json
from pathlib import Path
from datasets import load_dataset
from typing import Dict, List
import sys


def load_demonstration_index(pool_dir: Path) -> Dict:
    """Load existing demonstration index"""
    index_file = pool_dir / "demonstration_index.json"
    
    if not index_file.exists():
        print(f"❌ Error: {index_file} not found")
        return None
    
    with open(index_file, 'r') as f:
        return json.load(f)


def enrich_with_problem_statements(
    demonstrations: List[Dict],
    swe_bench_subset: str = "test"
) -> List[Dict]:
    """
    Add problem statements from SWE-bench to demonstrations
    
    Args:
        demonstrations: List of demonstration dicts
        swe_bench_subset: Which SWE-bench split to use (train/test/dev)
    """
    
    print(f"\n📥 Loading SWE-bench dataset (subset: {swe_bench_subset})...")
    try:
        # Load SWE-bench Lite dataset
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split=swe_bench_subset)
        
        # Create lookup by instance_id
        swe_bench_lookup = {item["instance_id"]: item for item in dataset}
        
        print(f"✅ Loaded {len(swe_bench_lookup)} instances from SWE-bench")
    except Exception as e:
        print(f"❌ Error loading SWE-bench: {e}")
        print("\nTrying alternative: SWE-bench (not Lite)...")
        try:
            dataset = load_dataset("princeton-nlp/SWE-bench", split=swe_bench_subset)
            swe_bench_lookup = {item["instance_id"]: item for item in dataset}
            print(f"✅ Loaded {len(swe_bench_lookup)} instances from SWE-bench")
        except Exception as e2:
            print(f"❌ Error loading SWE-bench: {e2}")
            return None
    
    # Enrich demonstrations
    print(f"\n🔄 Enriching {len(demonstrations)} demonstrations with problem statements...")
    
    enriched = []
    found = 0
    missing = 0
    
    for demo in demonstrations:
        instance_id = demo["instance_id"]
        
        if instance_id in swe_bench_lookup:
            swe_item = swe_bench_lookup[instance_id]
            
            # Add text features
            demo["problem_statement"] = swe_item.get("problem_statement", "")
            demo["hints_text"] = swe_item.get("hints_text", "")
            demo["created_at"] = swe_item.get("created_at", "")
            
            # Add patch if not already present
            if "patch" not in demo:
                demo["patch"] = swe_item.get("patch", "")
            
            found += 1
            print(f"  ✅ {instance_id}: {len(demo['problem_statement'])} chars")
        else:
            print(f"  ⚠️  {instance_id}: NOT FOUND in SWE-bench {swe_bench_subset} split")
            missing += 1
            
            # Keep demo but mark as missing problem statement
            demo["problem_statement"] = ""
            demo["hints_text"] = ""
        
        enriched.append(demo)
    
    print(f"\n📊 Enrichment Summary:")
    print(f"   Found:   {found}/{len(demonstrations)}")
    print(f"   Missing: {missing}/{len(demonstrations)}")
    
    if missing > 0:
        print(f"\n⚠️  Note: {missing} demonstrations not found in {swe_bench_subset} split")
        print(f"   They may be in train or dev split. Try running with different --split")
    
    return enriched


def save_enriched_index(
    pool_dir: Path,
    original_data: Dict,
    enriched_demonstrations: List[Dict]
):
    """Save enriched demonstration index"""
    
    # Update demonstrations
    original_data["demonstrations"] = enriched_demonstrations
    original_data["enriched"] = True
    original_data["text_features"] = ["problem_statement", "hints_text", "patch"]
    
    # Backup original
    index_file = pool_dir / "demonstration_index.json"
    backup_file = pool_dir / "demonstration_index_backup.json"
    
    if not backup_file.exists():
        import shutil
        shutil.copy(index_file, backup_file)
        print(f"\n💾 Backed up original to: {backup_file}")
    
    # Save enriched version
    with open(index_file, 'w') as f:
        json.dump(original_data, f, indent=2)
    
    print(f"✅ Saved enriched index to: {index_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enrich demonstration pool with problem statements from SWE-bench"
    )
    parser.add_argument(
        "--demo_pool",
        type=str,
        default="./demonstration_pool",
        help="Directory containing demonstration pool"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="SWE-bench split to load (your demos are likely from 'test')"
    )
    
    args = parser.parse_args()
    
    pool_dir = Path(args.demo_pool)
    
    # Load existing index
    print(f"📂 Loading demonstration index from: {pool_dir}")
    original_data = load_demonstration_index(pool_dir)
    
    if not original_data:
        return 1
    
    demonstrations = original_data.get("demonstrations", [])
    print(f"   Found {len(demonstrations)} demonstrations")
    
    # Check if already enriched
    if original_data.get("enriched", False):
        print("\n⚠️  Demonstration pool already enriched!")
        print("   To re-enrich, delete the 'enriched' field from demonstration_index.json")
        
        # Show sample
        if demonstrations and "problem_statement" in demonstrations[0]:
            sample = demonstrations[0]
            print(f"\n📝 Sample problem statement ({sample['instance_id']}):")
            print(f"   {sample['problem_statement'][:200]}...")
        
        return 0
    
    # Enrich with problem statements
    enriched = enrich_with_problem_statements(demonstrations, args.split)
    
    if enriched is None:
        print("\n❌ Failed to enrich demonstrations")
        return 1
    
    # Save
    save_enriched_index(pool_dir, original_data, enriched)
    
    print("\n✅ Demonstration pool enriched successfully!")
    print("\nNext step: Test TF-IDF selection with real text similarity")
    print("  python select_demonstrations.py --strategy tfidf --k 5 --query \"your test query\"")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())