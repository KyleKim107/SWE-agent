# SWE-bench Lite Evaluation with Semantic Features

This guide explains how to run SWE-bench Lite evaluation using semantic similarity for demonstration selection.

## Overview

The semantic evaluation uses **Sentence Transformers** to find the most similar demonstrations for each test instance based on semantic meaning, not just keyword matching. This typically provides better results than TF-IDF or BM25.

## Quick Start

### 1. Prerequisites

Make sure you have:
- ✅ Enriched demonstration pool (run `enrich_demonstration_pool.py` if needed)
- ✅ Sentence Transformers installed: `pip install sentence-transformers`
- ✅ SWE-agent configured with your API keys

### 2. Find Test Instances

```bash
cd SWE-agent/claude_tools

# List all available test instances
python3 get_test_instances.py

# Filter by repository
python3 get_test_instances.py --repo django

# Show only counts
python3 get_test_instances.py --count
```

### 3. Run Semantic Evaluation

```bash
# Single instance
./run_semantic_evaluation.sh --instances django__django-11283

# Multiple instances
./run_semantic_evaluation.sh --instances django__django-11283 django__django-10914 django__django-11179

# FULL SWE-bench Lite dataset (all instances)
./run_semantic_evaluation.sh --all

# Test with first 10 instances (recommended for initial testing)
./run_semantic_evaluation.sh --all --limit 10

# Custom number of demonstrations
./run_semantic_evaluation.sh --instances django__django-11283 --k 3

# Use a different model (better quality, slower)
./run_semantic_evaluation.sh --instances django__django-11283 --model all-mpnet-base-v2
```

### 4. Direct Python Usage

If you prefer to run the Python script directly:

```bash
# Single or multiple instances
python3 run_adaptive_evaluation_fixed.py \
    --demo_pool ./demonstration_pool \
    --instances django__django-11283 \
    --k 5 \
    --strategy semantic \
    --semantic_model all-MiniLM-L6-v2 \
    --output_dir ./test_results/semantic_k5 \
    --config ../config/tamu_config_improved.yaml \
    --swe_agent_dir ../ \
    --log_dir ./logs

# FULL SWE-bench Lite dataset
python3 run_adaptive_evaluation_fixed.py \
    --demo_pool ./demonstration_pool \
    --all \
    --k 5 \
    --strategy semantic \
    --semantic_model all-MiniLM-L6-v2 \
    --output_dir ./test_results/semantic_k5_full \
    --config ../config/tamu_config_improved.yaml \
    --swe_agent_dir ../ \
    --log_dir ./logs

# Test with limited instances
python3 run_adaptive_evaluation_fixed.py \
    --demo_pool ./demonstration_pool \
    --all \
    --limit 10 \
    --k 5 \
    --strategy semantic \
    --output_dir ./test_results/semantic_k5_test \
    --config ../config/tamu_config_improved.yaml \
    --swe_agent_dir ../
```

## Semantic Models

### Recommended Models

1. **all-MiniLM-L6-v2** (default)
   - Fast, good quality
   - 384-dimensional embeddings
   - Best for most use cases

2. **all-mpnet-base-v2**
   - Better quality, slower
   - 768-dimensional embeddings
   - Use when quality is more important than speed

3. **all-MiniLM-L12-v2**
   - Balance between speed and quality
   - 384-dimensional embeddings
   - Slightly slower than L6, better quality

### Model Selection

```bash
# Fast (default)
--model all-MiniLM-L6-v2

# Better quality
--model all-mpnet-base-v2

# Balanced
--model all-MiniLM-L12-v2
```

## Parameters

### Key Parameters

- `--k`: Number of demonstrations per instance (default: 5)
  - More demos = better context but higher cost
  - Recommended: 3-5 for most cases

- `--semantic_model`: Sentence Transformer model name
  - See "Semantic Models" section above

- `--mmr_lambda`: MMR diversity parameter (default: 0.7)
  - 1.0 = pure relevance (may have duplicates)
  - 0.0 = pure diversity (may miss relevant)
  - 0.7 = good balance (default)

### Output Structure

```
test_results/semantic_k5/
├── demos_<instance_id>.json          # Selected demonstrations metadata
├── <instance_id>/
│   ├── <instance_id>.traj            # Full trajectory
│   ├── <instance_id>.pred             # Prediction file
│   ├── <instance_id>.patch            # Generated patch (if successful)
│   └── *.log                          # Log files
└── evaluation_summary.json            # Summary of all results
```

## Comparison with Other Strategies

| Strategy | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| **semantic** | Medium | ⭐⭐⭐⭐⭐ | Best overall quality |
| tfidf | Fast | ⭐⭐⭐ | Quick baseline |
| bm25 | Fast | ⭐⭐⭐⭐ | Better than TF-IDF for short queries |
| random | Fastest | ⭐⭐ | Baseline comparison |

## Troubleshooting

### "sentence-transformers not installed"
```bash
pip install sentence-transformers
```

### "Demonstration pool not enriched"
```bash
python3 enrich_demonstration_pool.py --demo_pool ./demonstration_pool
```

### "Out of memory" errors
- Reduce `--k` (fewer demonstrations)
- Use smaller model: `--model all-MiniLM-L6-v2`
- Process fewer instances at a time

### Slow performance
- Use `all-MiniLM-L6-v2` instead of `all-mpnet-base-v2`
- Reduce `--k` to 3
- Process instances sequentially (not in parallel)

## Example Workflow

### Running on Full Dataset

```bash
# 1. Check demonstration pool
ls demonstration_pool/demonstration_index.json

# 2. Test with a small subset first (recommended!)
./run_semantic_evaluation.sh --all --limit 10 --output_dir ./test_results/semantic_k5_test

# 3. Check test results
cat test_results/semantic_k5_test/evaluation_summary.json

# 4. If test looks good, run on FULL dataset
./run_semantic_evaluation.sh --all --output_dir ./test_results/semantic_k5_full

# 5. Check final results
cat test_results/semantic_k5_full/evaluation_summary.json
```

### Running on Specific Instances

```bash
# 1. Find test instances
python3 get_test_instances.py --repo django --limit 5

# 2. Run evaluation on selected instances
./run_semantic_evaluation.sh \
    --instances django__django-11283 django__django-10914 \
    --k 5 \
    --output_dir ./test_results/semantic_k5

# 3. Check results
cat test_results/semantic_k5/evaluation_summary.json
```

## Advanced Usage

### Custom MMR Lambda

```bash
python3 run_adaptive_evaluation_fixed.py \
    --demo_pool ./demonstration_pool \
    --instances django__django-11283 \
    --k 5 \
    --strategy semantic \
    --mmr_lambda 0.8 \
    --output_dir ./test_results/semantic_k5_mmr08
```

### Dry Run (Test Without Executing)

```bash
python3 run_adaptive_evaluation_fixed.py \
    --demo_pool ./demonstration_pool \
    --instances django__django-11283 \
    --k 5 \
    --strategy semantic \
    --dry_run
```

## Results Analysis

After running, check:
1. `evaluation_summary.json` - Overall success rate
2. `demos_<instance_id>.json` - Which demonstrations were selected
3. `<instance_id>/<instance_id>.pred` - Generated patch
4. Log files in `--log_dir` for detailed execution traces

## Next Steps

- Compare semantic vs TF-IDF vs BM25 strategies
- Experiment with different `k` values
- Try different semantic models
- Analyze which demonstrations are most helpful

