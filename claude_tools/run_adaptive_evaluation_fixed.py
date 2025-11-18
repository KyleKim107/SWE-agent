#!/usr/bin/env python3
r"""
Adaptive Few-Shot Evaluation for SWE-agent - FIXED VERSION
For each test instance, select k most similar demonstrations using TF-IDF

KEY FIXES:
1. Create temporary config file with demonstrations (SWE-agent requires this)
2. Use 'python -m sweagent.run.run_single' command (like data collection)
3. Create temporary JSONL file with instance data
4. Use correct CLI arguments: --config_file, --instance_filter, --data_path

python run_adaptive_evaluation_fixed.py \
  --demo_pool ./demonstration_pool \
  --instances django__django-11283 \
  --k 5 \
  --strategy semantic \
  --semantic_model all-MiniLM-L6-v2 \
  --output_dir ./test_results/semantic_k5 \
  --config ../config/tamu_config_improved.yaml \
  --swe_agent_dir ../ \
  --log_dir ./logs


python run_adaptive_evaluation_fixed.py \
  --demo_pool ./demonstration_pool \          # 📦 Demonstration pool directory
  --instances django__django-11283 \          # 🎯 Test instance(s) to evaluate
  --k 5 \                                     # 🔢 Number of demonstrations per instance
  --strategy tfidf \                          # 🧠 Selection strategy
  --output_dir ./test_results/adaptive_tfidf_k5 \  # 💾 Output directory
  --config ../config/tamu_config_improved.yaml \   # ⚙️ Base config file
  --swe_agent_dir ../ \                       # 📁 SWE-agent root directory
  --log_dir ./logs                            # 📋 Log directory (optional, collects .log files)
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datasets import load_dataset
import argparse
import sys
import tempfile
import yaml


def load_demonstration_pool(pool_dir: Path) -> List[Dict]:
    """Load demonstration pool"""
    index_file = pool_dir / "demonstration_index.json"
    
    with open(index_file, 'r') as f:
        data = json.load(f)
    
    demonstrations = data.get("demonstrations", [])
    
    if not data.get("enriched", False):
        print("\n❌ ERROR: Demonstration pool must be enriched first!")
        print("   Run: python enrich_demonstration_pool.py --demo_pool ./demonstration_pool")
        sys.exit(1)
    
    return demonstrations


def extract_text_features(demo: Dict) -> str:
    """Extract text features from demonstration"""
    text_parts = [
        demo.get("problem_statement", ""),
        demo.get("hints_text", "")
    ]
    return " ".join(text_parts)


def select_demonstrations_tfidf(
    demonstrations: List[Dict],
    query_text: str,
    k: int,
    mmr_lambda: float = 0.7
) -> List[Dict]:
    """Select k most similar demonstrations using TF-IDF"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Extract text features
    demo_texts = [extract_text_features(demo) for demo in demonstrations]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )
    
    tfidf_matrix = vectorizer.fit_transform(demo_texts)
    query_vector = vectorizer.transform([query_text])
    
    # Compute similarities
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    # MMR selection for diversity
    return _select_with_mmr(demonstrations, similarities, k, mmr_lambda, tfidf_matrix)


def select_demonstrations_bm25(
    demonstrations: List[Dict],
    query_text: str,
    k: int,
    mmr_lambda: float = 0.7
) -> List[Dict]:
    """Select k most similar demonstrations using BM25 (improved TF-IDF)"""
    try:
        from rank_bm25 import BM25Okapi
        import numpy as np
    except ImportError:
        print("\n⚠️  rank_bm25 not installed. Install with: pip install rank-bm25")
        print("   Falling back to TF-IDF...")
        return select_demonstrations_tfidf(demonstrations, query_text, k, mmr_lambda)
    
    # Extract text features and tokenize
    demo_texts = [extract_text_features(demo) for demo in demonstrations]
    tokenized_demos = [text.lower().split() for text in demo_texts]
    tokenized_query = query_text.lower().split()
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_demos)
    
    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)
    similarities = np.array(scores)
    
    # MMR selection (using TF-IDF matrix for diversity calculation)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(demo_texts)
    
    return _select_with_mmr(demonstrations, similarities, k, mmr_lambda, tfidf_matrix)


def select_demonstrations_semantic(
    demonstrations: List[Dict],
    query_text: str,
    k: int,
    mmr_lambda: float = 0.7,
    model_name: str = "all-MiniLM-L6-v2"
) -> List[Dict]:
    """
    Select k most similar demonstrations using Sentence Transformers (semantic similarity)
    
    Recommended models:
    - "all-MiniLM-L6-v2": Fast, good quality (default)
    - "all-mpnet-base-v2": Better quality, slower
    - "sentence-transformers/all-MiniLM-L6-v2": Explicit path
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        print("\n⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")
        print("   Falling back to TF-IDF...")
        return select_demonstrations_tfidf(demonstrations, query_text, k, mmr_lambda)
    
    print(f"   📦 Loading semantic model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Extract text features
    demo_texts = [extract_text_features(demo) for demo in demonstrations]
    
    # Encode all texts
    print("   🔄 Encoding texts...")
    demo_embeddings = model.encode(demo_texts, show_progress_bar=False)
    query_embedding = model.encode([query_text], show_progress_bar=False)
    
    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, demo_embeddings)[0]
    
    # MMR selection (using embeddings for diversity calculation)
    return _select_with_mmr(demonstrations, similarities, k, mmr_lambda, demo_embeddings)


def _select_with_mmr(
    demonstrations: List[Dict],
    similarities,
    k: int,
    mmr_lambda: float,
    feature_matrix
) -> List[Dict]:
    """MMR (Maximal Marginal Relevance) selection for diversity"""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    selected_indices = []
    remaining_indices = list(range(len(demonstrations)))
    
    # Start with most relevant
    first_idx = np.argmax(similarities)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Iteratively select with MMR
    while len(selected_indices) < k and remaining_indices:
        selected_matrix = feature_matrix[selected_indices]
        best_score = -float('inf')
        best_idx = None
        
        for idx in remaining_indices:
            rel_score = similarities[idx]
            sims_to_selected = cosine_similarity(
                feature_matrix[idx:idx+1],
                selected_matrix
            )[0]
            max_sim = sims_to_selected.max()
            mmr_score = mmr_lambda * rel_score - (1 - mmr_lambda) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        else:
            break
    
    return [demonstrations[i] for i in selected_indices]


def select_demonstrations_for_instance(
    demonstrations: List[Dict],
    query_text: str,
    k: int,
    strategy: str = "tfidf",
    mmr_lambda: float = 0.7,
    semantic_model: str = "all-MiniLM-L6-v2"
) -> List[Dict]:
    """
    Select k most similar demonstrations using various strategies
    
    Strategies:
    - "tfidf": TF-IDF (fast, baseline)
    - "bm25": BM25 (improved TF-IDF, better for short queries)
    - "semantic": Sentence Transformers (best quality, understands meaning)
    """
    if strategy == "tfidf":
        return select_demonstrations_tfidf(demonstrations, query_text, k, mmr_lambda)
    elif strategy == "bm25":
        return select_demonstrations_bm25(demonstrations, query_text, k, mmr_lambda)
    elif strategy == "semantic":
        return select_demonstrations_semantic(demonstrations, query_text, k, mmr_lambda, semantic_model)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_config_with_demonstrations(
    base_config_path: Path,
    demo_paths: List[Path],
    output_path: Path,
    test_instance: Dict = None
) -> None:
    """
    NEW FUNCTION: Create a temporary config file with demonstrations
    
    This is the KEY FIX - SWE-agent needs demonstrations in the config file,
    not as CLI arguments!
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add demonstrations to config
    if 'agent' not in config:
        config['agent'] = {}
    if 'templates' not in config['agent']:
        config['agent']['templates'] = {}
    
    # Enhance system template with completion instructions to prevent infinite loops
    if 'system_template' in config['agent']['templates']:
        original_template = config['agent']['templates']['system_template']
        completion_instructions = """

  IMPORTANT: When you have successfully:
  1. Fixed the issue
  2. Verified the fix works (tests pass, migrations run successfully)
  3. Committed the changes

  You MUST submit your solution by running: submit

  Do NOT keep repeating the same verification commands. If you see the same results 2-3 times, submit immediately."""
        config['agent']['templates']['system_template'] = original_template + completion_instructions
    else:
        # If no system template exists, create one with completion instructions
        config['agent']['templates']['system_template'] = """You are a helpful software engineering assistant.

  IMPORTANT: When you have successfully:
  1. Fixed the issue
  2. Verified the fix works (tests pass, migrations run successfully)
  3. Committed the changes

  You MUST submit your solution by running: submit

  Do NOT keep repeating the same verification commands. If you see the same results 2-3 times, submit immediately."""
    
    # Set demonstrations (convert to strings for YAML)
    # Limit demonstrations to prevent context window overflow
    # With put_demos_in_history=True, each demo adds many steps to history
    # Strategy: Start with fewer demos (1-2) for better context retention
    # If context allows, can increase to 3-5 demos
    max_demos = min(len(demo_paths), 2)  # Limit to 2 demos to preserve more context for agent's own steps
    config['agent']['templates']['demonstrations'] = [str(p) for p in demo_paths[:max_demos]]
    config['agent']['templates']['put_demos_in_history'] = True
    
    if len(demo_paths) > max_demos:
        # Log warning if we're limiting demonstrations
        import logging
        logging.warning(f"Limiting demonstrations from {len(demo_paths)} to {max_demos} to prevent context window overflow")
    
    # Add history processor to reduce context window usage
    # This keeps only the last N observations to prevent context window overflow
    if 'history_processors' not in config['agent']:
        config['agent']['history_processors'] = []
    
    # Check if last_n_observations processor already exists
    has_last_n = any(
        isinstance(hp, dict) and hp.get('type') == 'last_n_observations'
        for hp in config['agent']['history_processors']
    )
    
    if not has_last_n:
        # Add history processor to keep only last N observations
        # Balance: Too small (n=1) loses important context, too large causes context overflow
        # With 3 demonstrations, n=5-10 is a good balance
        # If still having issues, try: (1) reduce demonstrations to 1-2, or (2) increase n to 10
        config['agent']['history_processors'].append({
            'type': 'last_n_observations',
            'n': 4  # Keep last 5 observations (default from SWE-agent paper)
        })
    
    # Configure environment for SWE-bench instance if provided
    if test_instance:
        # Set repo configuration
        if 'env' not in config:
            config['env'] = {}
        if 'repo' not in config['env']:
            config['env']['repo'] = {}
        
        config['env']['repo']['type'] = 'preexisting'
        config['env']['repo']['repo_name'] = 'testbed'
        if 'base_commit' in test_instance:
            config['env']['repo']['base_commit'] = test_instance['base_commit']
        
        # Set deployment image if available
        if 'deployment' not in config['env']:
            config['env']['deployment'] = {}
        
        # Get image name from instance or generate it
        image_name = test_instance.get('image_name')
        if not image_name:
            # Generate image name from instance_id (same logic as SimpleBatchInstance.from_swe_bench)
            instance_id = test_instance.get('instance_id', '')
            id_docker_compatible = instance_id.replace("__", "_1776_")
            image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        
        config['env']['deployment']['image'] = image_name
        config['env']['deployment']['type'] = 'docker'
        config['env']['deployment']['platform'] = 'linux/amd64'
        # Important: Set python_standalone_dir like run-batch does
        # This ensures standalone Python is installed in the Docker container
        if 'python_standalone_dir' not in config['env']['deployment']:
            config['env']['deployment']['python_standalone_dir'] = '/root'
    
    # Save temporary config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def collect_log_files(instance_dir: Path, log_dir: Path, instance_id: str) -> None:
    """Collect log files from instance directory to log directory"""
    log_levels = ["trace", "debug", "info"]
    instance_log_dir = log_dir / instance_id
    instance_log_dir.mkdir(parents=True, exist_ok=True)
    
    for level in log_levels:
        log_file = instance_dir / f"{instance_id}.{level}.log"
        if log_file.exists():
            dest_file = instance_log_dir / f"{instance_id}.{level}.log"
            shutil.copy2(log_file, dest_file)


def run_adaptive_evaluation(
    demo_pool_dir: Path,
    test_instances: List[str],
    k: int,
    output_dir: Path,
    config_file: Path,
    swe_agent_dir: Path,
    strategy: str = "tfidf",
    mmr_lambda: float = 0.7,
    dry_run: bool = False,
    log_dir: Optional[Path] = None,
    semantic_model: str = "all-MiniLM-L6-v2"
):
    """
    Run adaptive few-shot evaluation
    
    For each test instance:
    1. Load its problem statement
    2. Select k most similar demonstrations
    3. Create temporary config with those demonstrations
    4. Run SWE-agent with that config
    """
    
    print("\n" + "="*80)
    print("ADAPTIVE FEW-SHOT EVALUATION")
    print("="*80)
    print(f"\nStrategy: {strategy}")
    print(f"Test instances: {len(test_instances)}")
    print(f"Demonstrations per instance: k={k}")
    print(f"Output directory: {output_dir}")
    print(f"SWE-agent directory: {swe_agent_dir}")
    if log_dir:
        print(f"Log directory: {log_dir}")
    print("\n" + "="*80)
    
    # Create log directory if specified
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load demonstration pool
    print("\n📂 Loading demonstration pool...")
    demonstrations = load_demonstration_pool(demo_pool_dir)
    print(f"   Loaded {len(demonstrations)} demonstrations")
    
    # Load test instances from SWE-bench
    print("\n📥 Loading SWE-bench test instances...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    test_lookup = {item["instance_id"]: item for item in dataset}
    print(f"   Loaded {len(test_lookup)} test instances")
    
    # Process each test instance
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    
    for i, instance_id in enumerate(test_instances, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(test_instances)}] Processing: {instance_id}")
        print(f"{'='*80}")
        
        if instance_id not in test_lookup:
            print(f"   ⚠️  Instance not found in SWE-bench, skipping...")
            continue
        
        test_instance = test_lookup[instance_id]
        query_text = test_instance["problem_statement"]
        
        print(f"\n📝 Problem: {query_text[:100]}...")
        
        # Select demonstrations for this instance
        if strategy == "random":
            import random
            selected = random.sample(demonstrations, min(k, len(demonstrations)))
            print(f"\n🎲 Random selection:")
        elif strategy == "diversity":
            # For diversity, use same fixed set for all instances
            from collections import defaultdict
            by_repo = defaultdict(list)
            for demo in demonstrations:
                by_repo[demo["repository"]].append(demo)
            
            selected = []
            for repo in sorted(by_repo.keys()):
                if len(selected) < k:
                    selected.append(max(by_repo[repo], key=lambda x: x.get("patch_size", 0)))
            
            print(f"\n🌈 Diversity selection (fixed):")
        else:  # tfidf, bm25, semantic
            selected = select_demonstrations_for_instance(
                demonstrations,
                query_text,
                k,
                strategy=strategy,
                mmr_lambda=mmr_lambda,
                semantic_model=semantic_model
            )
            strategy_names = {
                "tfidf": "TF-IDF",
                "bm25": "BM25",
                "semantic": "Semantic (Sentence Transformers)"
            }
            print(f"\n🔍 {strategy_names.get(strategy, strategy.upper())} selection (adaptive):")
        
        for j, demo in enumerate(selected, 1):
            print(f"   {j}. {demo['instance_id']:<40} ({demo['repository']})")
        
        # Build trajectory file paths for demonstrations
        demo_traj_paths = []
        for demo in selected:
            traj_path = demo_pool_dir / demo["source_directory"] / demo["trajectory_file"]
            if not traj_path.exists():
                print(f"   ⚠️  Warning: Trajectory file not found: {traj_path}")
                continue
            demo_traj_paths.append(traj_path.resolve())
        
        if not demo_traj_paths:
            print(f"   ❌ No valid trajectory files found, skipping...")
            continue
        
        # Ensure output_dir is absolute before creating files
        output_dir = Path(output_dir).absolute()
        
        # Save demonstration metadata for this instance (for reference)
        demo_metadata_file = output_dir / f"demos_{instance_id}.json"
        with open(demo_metadata_file, 'w') as f:
            json.dump({
                "strategy": strategy,
                "k": k,
                "test_instance": instance_id,
                "query": query_text[:200],
                "demonstrations": selected,
                "trajectory_files": [str(p) for p in demo_traj_paths]
            }, f, indent=2)
        
        # NEW: Create temporary config file with demonstrations and environment config
        temp_config = output_dir / f"temp_config_{instance_id}.yaml"
        create_config_with_demonstrations(
            config_file,
            demo_traj_paths,
            temp_config,
            test_instance=test_instance
        )
        print(f"\n📝 Created config with {len(demo_traj_paths)} demonstrations")
        
        # NEW: Create temporary JSONL file with instance data (for reference)
        temp_instance_file = output_dir / f"temp_instance_{instance_id}.jsonl"
        with open(temp_instance_file, 'w') as f:
            json.dump(test_instance, f)
            f.write('\n')
        
        # Use TextProblemStatement with the SWE-bench instance text and id
        # subprocess.run with a list handles escaping automatically, so no need to escape
        problem_text = test_instance["problem_statement"]
        
        cmd = [
            'sweagent', 'run',
            '--config', str(temp_config),
            '--problem_statement.type', 'text',
            '--problem_statement.text', problem_text,
            '--problem_statement.id', instance_id,
            '--output_dir', str(output_dir)
        ]
        
        if dry_run:
            print(f"\n🔨 Would run:")
            print(f"   Command: {' '.join(cmd)}")
            print(f"   Working dir: {swe_agent_dir}")
        else:
            print(f"\n🚀 Running SWE-agent...")
            try:
                # Create log file for this instance (like auto_runner.sh does)
                instance_log_file = None
                if log_dir:
                    instance_log_file = log_dir / f"output_{instance_id}.log"
                    instance_log_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Run command and capture output
                if instance_log_file:
                    # Write to both log file and capture for error checking
                    with open(instance_log_file, 'w') as log_f:
                        result = subprocess.run(
                            cmd,
                            cwd=swe_agent_dir,
                            stdout=log_f,
                            stderr=subprocess.STDOUT,  # Merge stderr into stdout
                            text=True,
                            timeout=600
                        )
                    # Read back the log for error checking
                    with open(instance_log_file, 'r') as log_f:
                        log_output = log_f.read()
                else:
                    # If no log_dir, just capture output
                    result = subprocess.run(
                        cmd,
                        cwd=swe_agent_dir,
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    log_output = result.stdout + result.stderr


                # Initialize success variable
                success = False
                
                # Show errors if command failed
                if result.returncode != 0:
                    print(f"   ❌ Command failed with return code {result.returncode}")
                    if instance_log_file:
                        print(f"   📋 Full log saved to: {instance_log_file}")
                        # Show last few lines of log for quick debugging
                        if log_output:
                            last_lines = log_output.split('\n')[-20:]
                            print(f"   📋 Last 20 lines:\n" + '\n'.join(last_lines))
                    else:
                        print(f"   📋 STDOUT:\n{log_output[:1000]}")  # Show first 1000 chars
                    # Command failed, so success remains False
                else:
                    # Verify execution - check if trajectory file was created
                    traj_file = output_dir / instance_id / f"{instance_id}.traj"
                    if traj_file.exists():
                        print(f"   ✅ Agent executed successfully (trajectory file created)")
                    
                    # Log file info
                    if instance_log_file:
                        print(f"   📋 Full execution log saved to: {instance_log_file}")
                    
                    # Collect detailed log files if log_dir is specified
                    if log_dir:
                        instance_dir = output_dir / instance_id
                        if instance_dir.exists():
                            collect_log_files(instance_dir, log_dir, instance_id)
                            print(f"   📋 Detailed log files collected to: {log_dir / instance_id}")

                # Check if prediction file was created
                pred_file = output_dir / instance_id / f"{instance_id}.pred"
                if pred_file.exists():
                    with open(pred_file, 'r') as f:
                        pred_data = json.load(f)
                    success = pred_data.get('model_patch') not in [None, ""]
                    
                    if success:
                        print(f"   ✅ SUCCESS: Generated patch")
                    else:
                        print(f"   ⚠️  COMPLETED: No patch generated")
                else:
                    success = False
                    print(f"   ❌ FAILED: No prediction file")
                
                results.append({
                    "instance_id": instance_id,
                    "success": success,
                    "output_dir": str(output_dir / instance_id),
                    "return_code": result.returncode
                })
                
            except subprocess.TimeoutExpired:
                print(f"   ⏱️  TIMEOUT (10 min)")
                results.append({
                    "instance_id": instance_id,
                    "success": False,
                    "timeout": True
                })
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                results.append({
                    "instance_id": instance_id,
                    "success": False,
                    "error": str(e)
                })
            finally:
                # Clean up temp files
                if temp_config.exists():
                    temp_config.unlink()
                if temp_instance_file.exists():
                    temp_instance_file.unlink()
    
    # Save summary
    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "strategy": strategy,
            "k": k,
            "total_instances": len(test_instances),
            "results": results,
            "success_rate": sum(1 for r in results if r.get("success", False)) / len(results) if results else 0
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\n📊 Results:")
    print(f"   Total: {len(results)}")
    print(f"   Success: {sum(1 for r in results if r.get('success', False))}")
    print(f"   Failed: {sum(1 for r in results if not r.get('success', False))}")
    print(f"\n💾 Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive few-shot evaluation for SWE-agent (FIXED VERSION)"
    )
    parser.add_argument(
        "--demo_pool",
        type=str,
        default="./demonstration_pool",
        help="Demonstration pool directory"
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=None,
        help="Test instance IDs to evaluate (optional if --all is used)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on all instances in SWE-bench Lite test split"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="lite",
        choices=["lite", "verified", "full"],
        help="SWE-bench subset to use (default: lite)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "dev"],
        help="Dataset split to use (default: test)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of instances to process (useful for testing)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of demonstrations per instance"
    )
    parser.add_argument(
        "--strategy",
        choices=["random", "tfidf", "bm25", "semantic", "diversity"],
        default="tfidf",
        help="Selection strategy: tfidf (fast baseline), bm25 (improved TF-IDF), semantic (best quality, uses Sentence Transformers), random, diversity"
    )
    parser.add_argument(
        "--semantic_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence Transformer model name (only for semantic strategy). Options: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (better quality)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../config/tamu_config_improved.yaml",
        help="SWE-agent config file"
    )
    parser.add_argument(
        "--swe_agent_dir",
        type=str,
        default="../",
        help="SWE-agent directory (to run commands from)"
    )
    parser.add_argument(
        "--mmr_lambda",
        type=float,
        default=0.7,
        help="MMR lambda for TF-IDF (1.0=relevance, 0.0=diversity)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory to collect log files (if not specified, logs stay in output_dir)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.instances:
        parser.error("Either --instances or --all must be specified")
    
    # Load all instances if --all is specified
    if args.all:
        print(f"\n📥 Loading all instances from SWE-bench {args.subset} ({args.split} split)...")
        dataset = load_dataset(f"princeton-nlp/SWE-bench_{args.subset}", split=args.split)
        test_instances = [item["instance_id"] for item in dataset]
        
        if args.limit:
            test_instances = test_instances[:args.limit]
            print(f"   Limited to first {args.limit} instances")
        
        print(f"   Loaded {len(test_instances)} instances")
    else:
        test_instances = args.instances
    
    run_adaptive_evaluation(
        demo_pool_dir=Path(args.demo_pool),
        test_instances=test_instances,
        k=args.k,
        output_dir=Path(args.output_dir),
        config_file=Path(args.config),
        swe_agent_dir=Path(args.swe_agent_dir),
        strategy=args.strategy,
        mmr_lambda=args.mmr_lambda,
        dry_run=args.dry_run,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        semantic_model=args.semantic_model
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
