#!/bin/bash
# Helper script to run SWE-bench Lite evaluation with semantic features
# This script uses Sentence Transformers for demonstration selection

set -e

# Default values
DEMO_POOL="./demonstration_pool"
K=5
SEMANTIC_MODEL="all-MiniLM-L6-v2"
OUTPUT_DIR="../results"  # Changed to ../results to match original SWE-agent behavior
CONFIG="../config/tamu_config_improved.yaml"
SWE_AGENT_DIR="../"
LOG_DIR="./logs"
AUTO_EVALUATE=true  # Automatically evaluate results after completion

# Parse command line arguments
INSTANCES=()
ALL_INSTANCES=false
LIMIT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ALL_INSTANCES=true
            shift
            ;;
        --instances)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                INSTANCES+=("$1")
                shift
            done
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --model)
            SEMANTIC_MODEL="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --demo_pool)
            DEMO_POOL="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --swe_agent_dir)
            SWE_AGENT_DIR="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --no-auto-evaluate)
            AUTO_EVALUATE=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS] (--instances INSTANCE1 [INSTANCE2 ...] | --all)"
            echo ""
            echo "Options:"
            echo "  --instances INSTANCE1 [INSTANCE2 ...]  Test instance IDs (use with specific instances)"
            echo "  --all                                  Run on ALL instances from SWE-bench Lite (test split)"
            echo "  --limit N                             Limit to first N instances (only with --all)"
            echo "  --k N                                 Number of demonstrations (default: 5)"
            echo "  --model MODEL_NAME                     Sentence Transformer model (default: all-MiniLM-L6-v2)"
            echo "  --output_dir DIR                       Output directory (default: ../results)"
            echo "  --demo_pool DIR                        Demonstration pool directory (default: ./demonstration_pool)"
            echo "  --config FILE                          Config file (default: ../config/tamu_config_improved.yaml)"
            echo "  --swe_agent_dir DIR                    SWE-agent directory (default: ../)"
            echo "  --log_dir DIR                          Log directory (default: ./logs)"
            echo "  --no-auto-evaluate                     Don't automatically evaluate results after completion"
            echo ""
            echo "Examples:"
            echo "  # Run on a single instance"
            echo "  $0 --instances django__django-11283"
            echo ""
            echo "  # Run on multiple instances"
            echo "  $0 --instances django__django-11283 django__django-10914"
            echo ""
            echo "  # Run on FULL SWE-bench Lite dataset"
            echo "  $0 --all"
            echo ""
            echo "  # Run on first 10 instances (for testing)"
            echo "  $0 --all --limit 10"
            echo ""
            echo "  # Use a different model"
            echo "  $0 --all --model all-mpnet-base-v2"
            echo ""
            echo "  # Use fewer demonstrations"
            echo "  $0 --all --k 3"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if instances were provided or --all is used
if [ "$ALL_INSTANCES" = false ] && [ ${#INSTANCES[@]} -eq 0 ]; then
    echo "❌ Error: No instances provided"
    echo "Use --instances to specify test instance IDs, or --all to run on full dataset"
    echo "Example: $0 --instances django__django-11283"
    echo "Example: $0 --all"
    exit 1
fi

# Check if demonstration pool exists
if [ ! -d "$DEMO_POOL" ]; then
    echo "❌ Error: Demonstration pool not found at: $DEMO_POOL"
    echo "Please ensure the demonstration pool exists and is enriched"
    exit 1
fi

# Check if demonstration pool is enriched
if [ ! -f "$DEMO_POOL/demonstration_index.json" ]; then
    echo "❌ Error: demonstration_index.json not found in $DEMO_POOL"
    echo "Please run: python enrich_demonstration_pool.py --demo_pool $DEMO_POOL"
    exit 1
fi

# Check if sentence-transformers is installed
python3 -c "import sentence_transformers" 2>/dev/null || {
    echo "⚠️  Warning: sentence-transformers not installed"
    echo "Installing sentence-transformers..."
    pip install sentence-transformers
}

echo "="*80
echo "SWE-bench Lite Evaluation with Semantic Features"
echo "="*80
echo ""
echo "Configuration:"
echo "  Strategy:        semantic (Sentence Transformers)"
echo "  Model:           $SEMANTIC_MODEL"
echo "  Demonstrations: k=$K"
if [ "$ALL_INSTANCES" = true ]; then
    echo "  Instances:       ALL instances from SWE-bench Lite (test split)"
    if [ -n "$LIMIT" ]; then
        echo "  Limit:           First $LIMIT instances"
    fi
else
    echo "  Instances:       ${INSTANCES[*]}"
fi
echo "  Output:          $OUTPUT_DIR"
echo "  Demo Pool:       $DEMO_POOL"
echo "  Config:          $CONFIG"
echo "  SWE-agent Dir:   $SWE_AGENT_DIR"
if [ -n "$LOG_DIR" ]; then
    echo "  Log Dir:         $LOG_DIR"
fi
echo ""
echo "="*80
echo ""

# Build command
CMD_ARGS=(
    --demo_pool "$DEMO_POOL"
    --k "$K"
    --strategy semantic
    --semantic_model "$SEMANTIC_MODEL"
    --output_dir "$OUTPUT_DIR"
    --config "$CONFIG"
    --swe_agent_dir "$SWE_AGENT_DIR"
)

if [ "$ALL_INSTANCES" = true ]; then
    CMD_ARGS+=(--all)
    if [ -n "$LIMIT" ]; then
        CMD_ARGS+=(--limit "$LIMIT")
    fi
else
    CMD_ARGS+=(--instances "${INSTANCES[@]}")
fi

if [ -n "$LOG_DIR" ]; then
    CMD_ARGS+=(--log_dir "$LOG_DIR")
fi

# Run the evaluation
set +e  # Temporarily disable exit on error to handle evaluation result
python3 run_adaptive_evaluation_fixed.py "${CMD_ARGS[@]}"
EVAL_EXIT_CODE=$?
set -e  # Re-enable exit on error

echo ""
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation complete!"
    echo "   Results saved to: $OUTPUT_DIR"
    if [ -n "$LOG_DIR" ]; then
        echo "   Logs saved to: $LOG_DIR"
    fi
    
    # Automatically evaluate results if enabled
    if [ "$AUTO_EVALUATE" = true ]; then
        echo ""
        echo "="*80
        echo "Automatically evaluating results with swe-bench..."
        echo "="*80
        echo ""
        
        # Determine subset and split from the evaluation
        SUBSET="lite"
        SPLIT="test"
        
        python3 evaluate_swe_bench_results.py \
            --output_dir "$OUTPUT_DIR" \
            --subset "$SUBSET" \
            --split "$SPLIT" \
            --config "$CONFIG"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Results evaluation complete!"
            echo "   Check $OUTPUT_DIR/evaluation_stats.json for final statistics"
            echo "   Check $OUTPUT_DIR/results.json for detailed results"
        else
            echo ""
            echo "⚠️  Warning: Results evaluation failed"
            echo "   You can manually run:"
            echo "   python3 evaluate_swe_bench_results.py --output_dir $OUTPUT_DIR"
        fi
    fi
else
    echo "❌ Evaluation failed with exit code: $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

