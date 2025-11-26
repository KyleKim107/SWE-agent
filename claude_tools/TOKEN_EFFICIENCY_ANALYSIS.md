# Token Efficiency Analysis: Optimizing Input Context Loading

## Executive Summary

Analysis of SWE-agent token usage reveals that **93-99% of input tokens come from demonstrations**, with demonstration observations (command outputs) accounting for 71% of those tokens. This represents a significant opportunity for optimization.

### Key Findings

| Metric | Current | Optimized (Est.) |
|--------|---------|------------------|
| Avg tokens per instance | 863,261 | ~150,000-200,000 |
| Demo token percentage | 93-99% | 50-70% |
| Estimated 300-instance cost | ~258M tokens | ~50-60M tokens |
| Token reduction potential | - | **70-80%** |

### Implemented Optimizations ✅

| Compression Level | Demo Pool Tokens | Reduction |
|------------------|------------------|-----------|
| Original | 2,178,689 | - |
| Moderate (5k chars) | 1,305,802 | **40.1%** |
| Aggressive (2k chars) | 956,675 | **56.1%** |

**New compressed demo pools created:**
- `demonstration_pool_compressed/` - Moderate compression (40% reduction)
- `demonstration_pool_aggressive/` - Aggressive compression (56% reduction)

## Token Breakdown Analysis

### Current Distribution (11 test instances)

```
┌─────────────────────────────────────────────────────────┐
│ TOKEN DISTRIBUTION (per instance average)               │
├──────────────────────────────────────────┬──────────────┤
│ Demonstrations                           │ 93-99%       │
│   ├── Demo Observations (command output) │ 71% of demo  │
│   └── Demo Actions (agent responses)     │ 29% of demo  │
│ Agent's own steps                        │ 1-7%         │
│   ├── System prompt                      │ ~240 tokens  │
│   └── Instance template + history        │ ~5-25k       │
└──────────────────────────────────────────┴──────────────┘
```

### Root Cause Analysis

1. **Verbose Command Outputs in Demos**
   - Single `ls -la` output: 100,000 characters (~25,000 tokens)
   - Large code file dumps: 10,000+ characters per view
   - Directory listings consuming majority of context

2. **Full Demonstration Inclusion**
   - 2 demonstrations × ~50k tokens each = 100k tokens baseline
   - `put_demos_in_history=True` adds all 100+ demo steps to history
   - No compression applied to demo observations

3. **Accumulated Context**
   - Each API call resends entire history
   - 6 API calls average × 100k+ tokens = 600k+ total tokens sent

## Optimization Strategies

### 1. Demonstration Compression (Highest Impact: 60-70% reduction)

**a) Observation Truncation in Demos**
```python
# Current: Full output preserved
observation_chars = 100,000  # ~25k tokens

# Optimized: Aggressive truncation
observation_chars = 5,000    # ~1.25k tokens (95% reduction)
```

**b) Essential Step Filtering**
```python
# Keep only:
# 1. Problem understanding (first 2-3 steps)
# 2. Actual code modifications (str_replace_editor)
# 3. Verification (test runs)
# 4. Submit step

# Current: 100+ steps per demo
# Optimized: 10-15 essential steps (85% reduction)
```

**c) Minimal Demo Format**
```yaml
# Instead of full trajectory, use condensed format:
demonstrations:
  - problem: "Brief issue description"
    solution: "Key code change"
    verification: "Test result summary"
```

### 2. History Processors (Medium Impact: 20-30% reduction)

**Current Configuration:**
```yaml
history_processors:
  - type: last_n_observations
    n: 5
```

**Optimized Configuration:**
```yaml
history_processors:
  # Truncate old observations
  - type: last_n_observations
    n: 5
    always_remove_output_for_tags: ["remove_output", "verbose"]
  
  # Remove duplicate file views
  - type: closed_window
  
  # Strip verbose patterns (like full diffs)
  - type: remove_regex
    remove:
      - "<diff>.*</diff>"
      - "^total \\d+\\n(drwx.*\\n)+"  # Directory listings
```

### 3. Observation Limits (Medium Impact: 15-25% reduction)

```yaml
# Current
max_observation_length: 100_000  # ~25k tokens allowed

# Optimized
max_observation_length: 20_000   # ~5k tokens max
# or aggressive:
max_observation_length: 5_000    # ~1.25k tokens max
```

### 4. Fewer, More Targeted Demonstrations (High Impact: 30-50% reduction)

```python
# Current: 2 demonstrations selected by semantic similarity
max_demos = 2

# Optimized: 1 highly relevant demo
max_demos = 1

# Alternative: No demos for simpler issues
if issue_complexity < threshold:
    max_demos = 0
```

### 5. Problem Statement Summarization (Low Impact: 5-10% reduction)

```python
def summarize_problem_statement(text: str, max_tokens: int = 500) -> str:
    """Summarize verbose GitHub issues to key points."""
    # Extract: title, error message, expected behavior
    # Remove: lengthy stack traces, repeated code blocks
    pass
```

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 hours)

1. **Reduce `max_observation_length`** from 100k to 20k
   - File: `config/tamu_config_improved.yaml`
   - Expected savings: 15-20%

2. **Add `closed_window` history processor**
   - Removes duplicate file views
   - Expected savings: 5-10%

3. **Limit to 1 demonstration**
   - File: `run_adaptive_evaluation_fixed.py` line 315
   - Expected savings: 30-40%

### Phase 2: Demo Compression (2-4 hours)

1. Create compressed demonstration pool using `optimize_context_loading.py`
2. Filter demo steps to essential only
3. Truncate demo observations to 5k chars

### Phase 3: Advanced Optimizations (4-8 hours)

1. Implement minimal demo format (summary-based, not full trajectory)
2. Add problem statement summarization
3. Dynamic demo selection based on issue complexity

## Recommended Configuration

```yaml
# config/tamu_config_token_optimized.yaml
agent:
  templates:
    max_observation_length: 20000  # Was 100000
    demonstrations: []  # Will be set dynamically
    put_demos_in_history: true
  
  history_processors:
    - type: last_n_observations
      n: 5
    - type: closed_window
    - type: remove_regex
      remove:
        - "^total \\d+[\\s\\S]*?(?=\\n[^d-]|$)"  # Directory listings
```

## Validation Metrics

Track these metrics to measure optimization effectiveness:

1. **Tokens per instance** (target: <200k)
2. **Tokens per API call** (target: <50k)  
3. **Demo tokens as % of total** (target: <70%)
4. **Resolve rate** (maintain or improve current)
5. **Steps to resolution** (should not increase significantly)

## Usage

### Quick Start: Use Compressed Demo Pool

```bash
# Option 1: Use the pre-created compressed pool (40% reduction)
python run_adaptive_evaluation_fixed.py \
    --demo_pool ./demonstration_pool_compressed \
    --config ../config/tamu_config_token_optimized.yaml \
    --k 1 \  # Use only 1 demo for maximum efficiency
    --output_dir ./test_results/optimized \
    --all --limit 10

# Option 2: Use aggressive compression (56% reduction)
python run_adaptive_evaluation_fixed.py \
    --demo_pool ./demonstration_pool_aggressive \
    --config ../config/tamu_config_token_optimized.yaml \
    --k 1 \
    --output_dir ./test_results/aggressive \
    --all --limit 10
```

### Create Your Own Compressed Pool

```bash
# Create moderately compressed pool
python create_compressed_demo_pool.py \
    --input ./demonstration_pool \
    --output ./demonstration_pool_compressed \
    --max_obs_chars 5000

# Create aggressively compressed pool
python create_compressed_demo_pool.py \
    --input ./demonstration_pool \
    --output ./demonstration_pool_aggressive \
    --max_obs_chars 2000
```

### Analysis Tools

```bash
# Analyze current token usage
python optimize_context_loading.py analyze \
    --results_dir ./test_results/semantic_k5

# Create optimized config
python optimize_context_loading.py optimize-config \
    --base_config ../config/tamu_config_improved.yaml \
    --output ../config/tamu_config_token_optimized.yaml \
    --level moderate
```

## Expected Impact

| Scenario | Current Tokens | Optimized Tokens | Reduction |
|----------|---------------|------------------|-----------|
| Single instance (avg) | 863k | 150-200k | 75-80% |
| 300 instances (est.) | 259M | 45-60M | 75-80% |
| API calls saved | - | ~50% fewer calls* | - |

*Due to faster convergence with less context noise

## Risks and Mitigations

1. **Risk**: Removing too much context hurts model understanding
   - **Mitigation**: Gradual reduction, A/B testing on subset

2. **Risk**: Compressed demos lose teaching value
   - **Mitigation**: Keep essential problem-solving steps, verify resolve rate

3. **Risk**: Aggressive truncation cuts off relevant info
   - **Mitigation**: Smart truncation (preserve code blocks, error messages)

---

*Analysis performed on 11 SWE-bench Lite instances using semantic K5 demonstration selection.*

