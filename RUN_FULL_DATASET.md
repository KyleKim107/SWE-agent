# Running Semantic Evaluation on Full SWE-bench Lite Dataset

This guide shows you how to run semantic evaluation on the **entire SWE-bench Lite test dataset**.

## Quick Start

### Option 1: Using the Helper Script (Recommended)

```bash
cd /Users/kylekim/Desktop/TAMU/CS689/course_project/SWE-agent/claude_tools

# Test with first 10 instances (RECOMMENDED - test before full run!)
./run_semantic_evaluation.sh --all --limit 10

# Run on FULL dataset (all instances)
./run_semantic_evaluation.sh --all
```

### Option 2: Direct Python Command

```bash
cd /Users/kylekim/Desktop/TAMU/CS689/course_project/SWE-agent/claude_tools

# Test with 10 instances
python3 run_adaptive_evaluation_fixed.py \
    --demo_pool ./demonstration_pool \
    --all \
    --limit 10 \
    --k 5 \
    --strategy semantic \
    --semantic_model all-MiniLM-L6-v2 \
    --output_dir ./test_results/semantic_k5_test \
    --config ../config/tamu_config_improved.yaml \
    --swe_agent_dir ../ \
    --log_dir ./logs

# Full dataset
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
```

## Important Notes

### ⚠️ Time and Cost Considerations

- **SWE-bench Lite test split** contains **~300 instances**
- Each instance takes **~5-10 minutes** to process
- **Full dataset will take ~25-50 hours** to complete
- Each instance uses API calls (costs apply)

### ✅ Recommended Workflow

1. **Test with small subset first:**
   ```bash
   ./run_semantic_evaluation.sh --all --limit 10
   ```

2. **Check results:**
   ```bash
   cat test_results/semantic_k5_test/evaluation_summary.json
   ```

3. **If results look good, run full dataset:**
   ```bash
   ./run_semantic_evaluation.sh --all
   ```

### 📊 Monitoring Progress

The script will show progress for each instance:
```
[1/300] Processing: django__django-11283
[2/300] Processing: django__django-10914
...
```

Results are saved incrementally, so you can check progress anytime:
```bash
cat test_results/semantic_k5_full/evaluation_summary.json
```

### 🔧 Customization Options

```bash
# Use fewer demonstrations (faster, less context)
./run_semantic_evaluation.sh --all --k 3

# Use better quality model (slower, better results)
./run_semantic_evaluation.sh --all --model all-mpnet-base-v2

# Custom output directory
./run_semantic_evaluation.sh --all --output_dir ./results/full_run_$(date +%Y%m%d)
```

### 📁 Output Structure

After completion, you'll have:
```
test_results/semantic_k5_full/
├── evaluation_summary.json          # Overall results
├── demos_<instance_id>.json        # Selected demos for each instance
├── <instance_id>/
│   ├── <instance_id>.traj          # Full trajectory
│   ├── <instance_id>.pred           # Prediction
│   └── <instance_id>.patch          # Generated patch (if successful)
└── ...
```

### 🛑 Resuming After Interruption

If the process is interrupted, you can:
1. Check which instances completed: `ls test_results/semantic_k5_full/`
2. Identify missing instances
3. Re-run with only missing instances using `--instances`

Or simply re-run `--all` - the script will overwrite existing results.

### 💡 Tips

- **Run in background** for long runs:
  ```bash
  nohup ./run_semantic_evaluation.sh --all > full_run.log 2>&1 &
  ```

- **Monitor disk space** - each instance generates ~1-5MB of data

- **Check logs** if issues occur:
  ```bash
  tail -f logs/output_<instance_id>.log
  ```

## Expected Results

After full run, check:
```bash
# Overall success rate
cat test_results/semantic_k5_full/evaluation_summary.json | grep success_rate

# Count successful patches
find test_results/semantic_k5_full -name "*.patch" | wc -l
```

