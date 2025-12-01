#!/bin/bash

# macOS-compatible version (works with bash 3.x)
# Create log directory if it doesn't exist
mkdir -p ./log

# Define topics as parallel arrays
TOPIC_NAMES=(
    # "astropy"
    "django"
    # "marshmallow-code"
    # "matplotlib"
    # "mwaskom"
    # "pallets"
    # "psf"
    # "pvlib"
    # "pydata"
    # "pydicom"
    # "pylint-dev"
    # "pytest-dev"
    # "pyvista"
    # "scikit-learn"
    # "sphinx-doc"
    # "sqlfluff"
    # "sympy"
)

TOPIC_FILTERS=(
    # "astropy__astropy-.*"
    "django__django-.*"
    # "marshmallow-code__marshmallow-.*"
    # "matplotlib__matplotlib-.*"
    # "mwaskom__mwaskom-.*"
    # "pallets__pallets-.*"
    # "psf__psf-.*"
    # "pvlib__pvlib-.*"
    # "pydata__pydata-.*"
    # "pydicom__pydicom-.*"
    # "pylint-dev__pylint-.*"
    # "pytest-dev__pytest-.*"
    # "pyvista__pyvista-.*"
    # "scikit-learn__scikit-learn-.*"
    # "sphinx-doc__sphinx-.*"
    # "sqlfluff__sqlfluff-.*"
    # "sympy__sympy-.*"
)

# Configuration
CONFIG_FILE="config/tamu_config_improved.yaml"
OUTPUT_DIR="./results"
NUM_WORKERS=1
PROBLEMS_PER_TOPIC=5

# Function to run a single topic
run_topic() {
    local topic_name=$1
    local filter_pattern=$2
    
    echo "========================================"
    echo "Starting: $topic_name"
    echo "Filter: $filter_pattern"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    
    sweagent run-batch \
        --config "$CONFIG_FILE" \
        --instances.type swe_bench \
        --instances.subset lite \
        --instances.split test \
        --instances.filter "$filter_pattern" \
        --instances.slice ":$PROBLEMS_PER_TOPIC" \
        --output_dir "$OUTPUT_DIR" \
        --num_workers "$NUM_WORKERS" > "./log/output_${topic_name}.log" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed: $topic_name (exit code: $exit_code)"
    else
        echo "✗ Failed: $topic_name (exit code: $exit_code)"
    fi
    
    echo ""
}

# Main execution
echo "========================================"
echo "SWE-bench Batch Runner"
echo "========================================"
echo "Problems per topic: $PROBLEMS_PER_TOPIC"
echo "Total topics: ${#TOPIC_NAMES[@]}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Track start time
start_time=$(date +%s)

# Run each topic sequentially
for i in "${!TOPIC_NAMES[@]}"; do
    topic_name="${TOPIC_NAMES[$i]}"
    filter_pattern="${TOPIC_FILTERS[$i]}"
    
    echo "Progress: $((i+1))/${#TOPIC_NAMES[@]}"
    run_topic "$topic_name" "$filter_pattern"
done

# Track end time
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
hours=$((minutes / 60))
remaining_minutes=$((minutes % 60))

echo "========================================"
echo "All topics completed!"
echo "Total time: ${duration} seconds (${hours}h ${remaining_minutes}m)"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""
echo "Check individual logs in ./log/ directory"
echo "Check results in $OUTPUT_DIR directory"