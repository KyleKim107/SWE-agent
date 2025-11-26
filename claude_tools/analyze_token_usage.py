#!/usr/bin/env python3
"""
Analyze token usage from SWE-agent trajectory files.

This script extracts token usage statistics from completed runs to help estimate
the total tokens/cost needed for running the full SWE-bench Lite dataset (300 instances).

Usage:
    # Analyze a specific results directory
    python analyze_token_usage.py --results_dir ./test_results/semantic_k5_test

    # After running a few instances, estimate total for 300 instances
    python analyze_token_usage.py --results_dir ./test_results/semantic_k5_test --estimate 300
"""

import json
import argparse
from pathlib import Path
from typing import Optional


def extract_token_stats(traj_file: Path) -> Optional[dict]:
    """Extract token statistics from a trajectory file."""
    try:
        data = json.loads(traj_file.read_text())
        info = data.get("info", {})
        model_stats = info.get("model_stats", {})
        
        if not model_stats:
            return None
        
        return {
            "instance_id": traj_file.stem,
            "tokens_sent": model_stats.get("tokens_sent", 0),
            "tokens_received": model_stats.get("tokens_received", 0),
            "total_tokens": model_stats.get("tokens_sent", 0) + model_stats.get("tokens_received", 0),
            "api_calls": model_stats.get("api_calls", 0),
            "instance_cost": model_stats.get("instance_cost", 0.0),
            "exit_status": info.get("exit_status", "unknown")
        }
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"⚠️  Error reading {traj_file}: {e}")
        return None


def analyze_results_directory(results_dir: Path) -> list[dict]:
    """Analyze all trajectory files in a results directory."""
    results_dir = Path(results_dir)
    stats_list = []
    
    # Find all .traj files
    traj_files = list(results_dir.glob("**/*.traj"))
    
    if not traj_files:
        print(f"❌ No trajectory files found in {results_dir}")
        return []
    
    print(f"📂 Found {len(traj_files)} trajectory files")
    
    for traj_file in traj_files:
        stats = extract_token_stats(traj_file)
        if stats:
            stats_list.append(stats)
    
    return stats_list


def print_statistics(stats_list: list[dict], estimate_instances: Optional[int] = None):
    """Print token usage statistics."""
    if not stats_list:
        print("❌ No valid statistics to display")
        return
    
    print("\n" + "=" * 80)
    print("TOKEN USAGE STATISTICS")
    print("=" * 80)
    
    # Calculate aggregates
    total_sent = sum(s["tokens_sent"] for s in stats_list)
    total_received = sum(s["tokens_received"] for s in stats_list)
    total_tokens = sum(s["total_tokens"] for s in stats_list)
    total_api_calls = sum(s["api_calls"] for s in stats_list)
    total_cost = sum(s["instance_cost"] for s in stats_list)
    n_instances = len(stats_list)
    
    avg_sent = total_sent / n_instances
    avg_received = total_received / n_instances
    avg_total = total_tokens / n_instances
    avg_api_calls = total_api_calls / n_instances
    avg_cost = total_cost / n_instances
    
    # Find min/max
    min_tokens = min(s["total_tokens"] for s in stats_list)
    max_tokens = max(s["total_tokens"] for s in stats_list)
    min_cost = min(s["instance_cost"] for s in stats_list)
    max_cost = max(s["instance_cost"] for s in stats_list)
    
    print(f"\n📊 Analyzed Instances: {n_instances}")
    print(f"\n{'─' * 40}")
    print("PER INSTANCE AVERAGES:")
    print(f"{'─' * 40}")
    print(f"  Input Tokens (avg):   {avg_sent:,.0f}")
    print(f"  Output Tokens (avg):  {avg_received:,.0f}")
    print(f"  Total Tokens (avg):   {avg_total:,.0f}")
    print(f"  API Calls (avg):      {avg_api_calls:.1f}")
    print(f"  Cost (avg):           ${avg_cost:.4f}")
    
    print(f"\n{'─' * 40}")
    print("TOKEN RANGE:")
    print(f"{'─' * 40}")
    print(f"  Min Total Tokens:     {min_tokens:,}")
    print(f"  Max Total Tokens:     {max_tokens:,}")
    print(f"  Min Cost:             ${min_cost:.4f}")
    print(f"  Max Cost:             ${max_cost:.4f}")
    
    print(f"\n{'─' * 40}")
    print("TOTALS (current run):")
    print(f"{'─' * 40}")
    print(f"  Total Input Tokens:   {total_sent:,}")
    print(f"  Total Output Tokens:  {total_received:,}")
    print(f"  Total Tokens:         {total_tokens:,}")
    print(f"  Total API Calls:      {total_api_calls:,}")
    print(f"  Total Cost:           ${total_cost:.4f}")
    
    # Estimation for full dataset
    if estimate_instances:
        print(f"\n{'=' * 80}")
        print(f"ESTIMATION FOR {estimate_instances} INSTANCES (SWE-bench Lite)")
        print(f"{'=' * 80}")
        
        est_sent = avg_sent * estimate_instances
        est_received = avg_received * estimate_instances
        est_total = avg_total * estimate_instances
        est_cost = avg_cost * estimate_instances
        est_api_calls = avg_api_calls * estimate_instances
        
        # Conservative estimate (using max values)
        est_total_max = max_tokens * estimate_instances
        est_cost_max = max_cost * estimate_instances
        
        print(f"\n📈 Based on {n_instances} sample instances:")
        print(f"\n{'─' * 40}")
        print("AVERAGE ESTIMATE:")
        print(f"{'─' * 40}")
        print(f"  Est. Input Tokens:    {est_sent:,.0f}")
        print(f"  Est. Output Tokens:   {est_received:,.0f}")
        print(f"  Est. Total Tokens:    {est_total:,.0f}")
        print(f"  Est. API Calls:       {est_api_calls:,.0f}")
        print(f"  Est. Total Cost:      ${est_cost:.2f}")
        
        print(f"\n{'─' * 40}")
        print("CONSERVATIVE ESTIMATE (based on max values):")
        print(f"{'─' * 40}")
        print(f"  Est. Total Tokens:    {est_total_max:,}")
        print(f"  Est. Total Cost:      ${est_cost_max:.2f}")
        
        print(f"\n{'─' * 40}")
        print("💡 RECOMMENDATION FOR CREDIT REQUEST:")
        print(f"{'─' * 40}")
        # Add 20% buffer for safety
        buffer_tokens = int(est_total * 1.2)
        buffer_cost = est_cost * 1.2
        buffer_tokens_max = int(est_total_max * 1.2)
        buffer_cost_max = est_cost_max * 1.2
        
        print(f"  Average + 20% buffer:")
        print(f"    Tokens: {buffer_tokens:,}")
        print(f"    Cost:   ${buffer_cost:.2f}")
        print(f"\n  Conservative + 20% buffer:")
        print(f"    Tokens: {buffer_tokens_max:,}")
        print(f"    Cost:   ${buffer_cost_max:.2f}")
    
    # Per-instance breakdown
    print(f"\n{'=' * 80}")
    print("PER-INSTANCE BREAKDOWN")
    print(f"{'=' * 80}")
    print(f"\n{'Instance ID':<45} {'Sent':>12} {'Received':>12} {'Total':>12} {'Cost':>10}")
    print("-" * 95)
    
    for s in sorted(stats_list, key=lambda x: x["total_tokens"], reverse=True):
        print(f"{s['instance_id']:<45} {s['tokens_sent']:>12,} {s['tokens_received']:>12,} {s['total_tokens']:>12,} ${s['instance_cost']:>8.4f}")
    
    print("-" * 95)
    print(f"{'TOTAL':<45} {total_sent:>12,} {total_received:>12,} {total_tokens:>12,} ${total_cost:>8.4f}")


def save_report(stats_list: list[dict], output_file: Path, estimate_instances: Optional[int] = None):
    """Save a JSON report of the statistics."""
    if not stats_list:
        return
    
    total_sent = sum(s["tokens_sent"] for s in stats_list)
    total_received = sum(s["tokens_received"] for s in stats_list)
    total_tokens = sum(s["total_tokens"] for s in stats_list)
    total_api_calls = sum(s["api_calls"] for s in stats_list)
    total_cost = sum(s["instance_cost"] for s in stats_list)
    n_instances = len(stats_list)
    
    report = {
        "summary": {
            "instances_analyzed": n_instances,
            "total_tokens_sent": total_sent,
            "total_tokens_received": total_received,
            "total_tokens": total_tokens,
            "total_api_calls": total_api_calls,
            "total_cost": total_cost,
            "avg_tokens_sent": total_sent / n_instances,
            "avg_tokens_received": total_received / n_instances,
            "avg_total_tokens": total_tokens / n_instances,
            "avg_api_calls": total_api_calls / n_instances,
            "avg_cost": total_cost / n_instances,
            "min_tokens": min(s["total_tokens"] for s in stats_list),
            "max_tokens": max(s["total_tokens"] for s in stats_list),
            "min_cost": min(s["instance_cost"] for s in stats_list),
            "max_cost": max(s["instance_cost"] for s in stats_list),
        },
        "instances": stats_list
    }
    
    if estimate_instances:
        avg_total = total_tokens / n_instances
        avg_cost = total_cost / n_instances
        max_tokens = max(s["total_tokens"] for s in stats_list)
        max_cost = max(s["instance_cost"] for s in stats_list)
        
        report["estimation"] = {
            "target_instances": estimate_instances,
            "avg_estimate": {
                "total_tokens": avg_total * estimate_instances,
                "total_cost": avg_cost * estimate_instances,
            },
            "conservative_estimate": {
                "total_tokens": max_tokens * estimate_instances,
                "total_cost": max_cost * estimate_instances,
            },
            "recommended_with_buffer": {
                "total_tokens": int(avg_total * estimate_instances * 1.2),
                "total_cost": avg_cost * estimate_instances * 1.2,
            }
        }
    
    output_file.write_text(json.dumps(report, indent=2))
    print(f"\n💾 Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token usage from SWE-agent trajectory files"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing trajectory files (.traj)"
    )
    parser.add_argument(
        "--estimate",
        type=int,
        default=300,
        help="Number of instances to estimate for (default: 300 for SWE-bench Lite)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for the report (optional)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Directory not found: {results_dir}")
        return 1
    
    stats_list = analyze_results_directory(results_dir)
    
    if stats_list:
        print_statistics(stats_list, args.estimate)
        
        if args.output:
            save_report(stats_list, Path(args.output), args.estimate)
        else:
            # Auto-save report
            report_file = results_dir / "token_usage_report.json"
            save_report(stats_list, report_file, args.estimate)
    
    return 0


if __name__ == "__main__":
    exit(main())

