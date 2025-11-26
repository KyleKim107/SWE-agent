#!/usr/bin/env python3
"""
Optimize Input Context Loading for SWE-agent Token Efficiency

This module provides tools to reduce token usage by optimizing how input context
is loaded and presented to the LLM, with a focus on:

1. Demonstration Compression - 93%+ of tokens come from demos
2. Observation Truncation - Reduce verbose command outputs  
3. History Pruning - Smart context window management
4. Problem Statement Summarization - Condense issue descriptions

Analysis Results:
- Demo tokens: 93-99% of total context
- Demo observations: 71% of demo tokens (verbose command outputs)
- Single large observation: 100k chars (25k tokens) from `ls -la` output

Key Optimizations:
1. Compress demonstrations by removing verbose observations
2. Apply history processors to demos (currently only applied to agent history)
3. Summarize long code files shown in demos
4. Use fewer, more targeted demonstrations (1-2 vs 5)
"""

import json
import re
import argparse
from pathlib import Path
from typing import Any
from copy import deepcopy


def compress_demonstration(traj_path: Path, max_observation_chars: int = 5000) -> dict:
    """
    Compress a demonstration trajectory by:
    1. Truncating long observations
    2. Removing redundant directory listings
    3. Summarizing large code blocks
    
    Args:
        traj_path: Path to the trajectory file
        max_observation_chars: Maximum characters per observation
        
    Returns:
        Compressed trajectory dict
    """
    traj = json.loads(traj_path.read_text())
    history = traj.get("history", [])
    compressed_history = []
    
    for entry in history:
        entry = deepcopy(entry)
        content = entry.get("content", "")
        
        if entry.get("message_type") == "observation":
            # Truncate long observations
            if len(content) > max_observation_chars:
                # Check for directory listings - these are low value
                if _is_directory_listing(content):
                    content = _summarize_directory_listing(content)
                # Check for large code file dumps
                elif _is_code_dump(content):
                    content = _summarize_code_dump(content, max_observation_chars)
                else:
                    content = content[:max_observation_chars] + f"\n... [{len(content) - max_observation_chars} chars truncated]"
                entry["content"] = content
        
        compressed_history.append(entry)
    
    traj["history"] = compressed_history
    return traj


def _is_directory_listing(content: str) -> bool:
    """Check if content is a directory listing."""
    lines = content.split('\n')
    dir_patterns = [
        r'^d[rwx-]+\s+\d+\s+\w+\s+\w+',  # ls -l format
        r'^total \d+',  # ls output header
        r'^\.\./\s*$',  # .. directory
    ]
    dir_line_count = sum(1 for line in lines[:50] if any(re.match(p, line) for p in dir_patterns))
    return dir_line_count > 10


def _summarize_directory_listing(content: str) -> str:
    """Summarize a directory listing to key info."""
    lines = content.split('\n')
    
    # Extract important files (Python, config files)
    important_extensions = ['.py', '.yaml', '.yml', '.json', '.txt', '.md', '.rst']
    important_files = []
    
    for line in lines:
        for ext in important_extensions:
            if ext in line.lower():
                # Extract filename
                parts = line.split()
                if parts:
                    important_files.append(parts[-1])
                break
    
    summary = f"[Directory listing with {len(lines)} entries]\n"
    if important_files:
        summary += "Key files: " + ", ".join(important_files[:20])
        if len(important_files) > 20:
            summary += f" ... and {len(important_files) - 20} more"
    return summary


def _is_code_dump(content: str) -> bool:
    """Check if content is a code file dump."""
    # Look for line number patterns from cat -n
    line_num_pattern = r'^\s*\d+[\|\t]'
    lines = content.split('\n')[:100]
    numbered_lines = sum(1 for line in lines if re.match(line_num_pattern, line))
    return numbered_lines > 20


def _summarize_code_dump(content: str, max_chars: int) -> str:
    """Summarize a code dump to relevant portions."""
    lines = content.split('\n')
    
    # Keep first part (context) and look for key sections
    header = '\n'.join(lines[:30])
    
    # Find function/class definitions
    key_patterns = [
        r'^\s*def\s+\w+',
        r'^\s*class\s+\w+',
        r'^\s*async\s+def\s+\w+',
    ]
    
    key_lines = []
    for i, line in enumerate(lines):
        for pattern in key_patterns:
            if re.match(pattern, line):
                # Include context around the definition
                start = max(0, i - 1)
                end = min(len(lines), i + 5)
                key_lines.extend(lines[start:end])
                key_lines.append('...')
                break
    
    result = header + '\n...\n' + '\n'.join(key_lines[:50])
    
    if len(result) > max_chars:
        result = result[:max_chars] + f"\n... [code truncated]"
    
    return result


def compress_demo_for_efficiency(demo_path: Path, output_path: Path, 
                                   target_tokens: int = 10000) -> dict:
    """
    Aggressively compress a demonstration to fit in target token count.
    
    Strategy:
    1. Keep essential steps (problem understanding, solution, verification)
    2. Remove intermediate exploration steps
    3. Heavily truncate observations
    
    Args:
        demo_path: Path to original demonstration
        output_path: Path for compressed output
        target_tokens: Target token count (chars / 4 approx)
        
    Returns:
        Stats about compression
    """
    target_chars = target_tokens * 4
    
    traj = json.loads(demo_path.read_text())
    history = traj.get("history", [])
    
    original_chars = sum(len(h.get("content", "")) for h in history)
    
    # Phase 1: Identify essential vs exploratory steps
    essential_indices = _identify_essential_steps(history)
    
    # Phase 2: Filter to essential steps only
    filtered_history = []
    for i, entry in enumerate(history):
        if i in essential_indices or entry.get("role") == "system":
            filtered_history.append(deepcopy(entry))
    
    # Phase 3: Compress observations
    max_obs_chars = target_chars // len(filtered_history) if filtered_history else 2000
    
    for entry in filtered_history:
        if entry.get("message_type") == "observation":
            content = entry.get("content", "")
            if len(content) > max_obs_chars:
                entry["content"] = content[:max_obs_chars] + "\n... [truncated]"
    
    compressed_chars = sum(len(h.get("content", "")) for h in filtered_history)
    
    # Save compressed demo
    traj["history"] = filtered_history
    output_path.write_text(json.dumps(traj, indent=2))
    
    return {
        "original_tokens": original_chars // 4,
        "compressed_tokens": compressed_chars // 4,
        "reduction_pct": 100 * (1 - compressed_chars / original_chars),
        "original_steps": len(history),
        "compressed_steps": len(filtered_history)
    }


def _identify_essential_steps(history: list) -> set:
    """
    Identify essential steps in a demonstration.
    
    Essential steps:
    1. First step (problem understanding)
    2. Steps that modify files (actual solution)
    3. Verification steps (test runs)
    4. Submit step
    """
    essential = set()
    
    for i, entry in enumerate(history):
        content = str(entry.get("content", "")).lower()
        action = str(entry.get("action", "")).lower()
        
        # First few entries are usually important context
        if i < 4:
            essential.add(i)
            continue
        
        # File modification steps
        if any(kw in action for kw in ["str_replace", "edit", "create", "write"]):
            essential.add(i)
            essential.add(i + 1)  # Include observation
            continue
        
        # Test/verification steps
        if any(kw in action for kw in ["test", "pytest", "python", "verify"]):
            essential.add(i)
            essential.add(i + 1)
            continue
            
        # Submit step
        if "submit" in action:
            essential.add(i)
            continue
    
    # Always include last few steps
    for i in range(max(0, len(history) - 4), len(history)):
        essential.add(i)
    
    return essential


def create_minimal_demo(traj_path: Path) -> dict:
    """
    Create a minimal demonstration showing only:
    1. Problem statement
    2. Key code modification
    3. Verification
    4. Submit
    
    This is ~5-10x more token efficient than full demos.
    """
    traj = json.loads(traj_path.read_text())
    history = traj.get("history", [])
    
    # Extract key moments
    minimal_history = []
    
    # 1. Get system prompt
    for entry in history:
        if entry.get("role") == "system":
            minimal_history.append(entry)
            break
    
    # 2. Get first instance template (problem statement)
    for entry in history:
        if entry.get("message_type") == "observation" and not entry.get("is_demo"):
            minimal_history.append(entry)
            break
    
    # 3. Find the actual fix (str_replace_editor edit)
    for i, entry in enumerate(history):
        action = str(entry.get("action", ""))
        if "str_replace" in action and "old_str" in action:
            # Include the thought + action
            minimal_history.append({
                "role": "assistant",
                "content": entry.get("thought", "") or "Making the fix...",
                "action": action,
                "message_type": "action"
            })
            # Include the observation
            if i + 1 < len(history):
                minimal_history.append(history[i + 1])
            break
    
    # 4. Find submit
    for i, entry in enumerate(history):
        action = str(entry.get("action", ""))
        if "submit" in action.lower():
            minimal_history.append(entry)
            break
    
    return {"history": minimal_history}


def analyze_context_breakdown(results_dir: Path) -> dict:
    """
    Analyze token usage breakdown for optimization opportunities.
    """
    traj_files = list(results_dir.glob("**/*.traj"))
    
    breakdown = {
        "total_files": len(traj_files),
        "demo_tokens_pct": [],
        "observation_tokens_pct": [],
        "largest_observations": [],
        "potential_savings": {}
    }
    
    for traj_file in traj_files:
        traj = json.loads(traj_file.read_text())
        history = traj.get("history", [])
        
        demo_chars = sum(len(h.get("content", "")) for h in history if h.get("is_demo"))
        non_demo_chars = sum(len(h.get("content", "")) for h in history if not h.get("is_demo"))
        total_chars = demo_chars + non_demo_chars
        
        if total_chars > 0:
            breakdown["demo_tokens_pct"].append(100 * demo_chars / total_chars)
        
        # Find largest observations
        for i, h in enumerate(history):
            if h.get("message_type") == "observation":
                content = h.get("content", "")
                if len(content) > 20000:
                    breakdown["largest_observations"].append({
                        "file": str(traj_file),
                        "index": i,
                        "chars": len(content),
                        "preview": content[:200]
                    })
    
    # Calculate potential savings
    avg_demo_pct = sum(breakdown["demo_tokens_pct"]) / len(breakdown["demo_tokens_pct"]) if breakdown["demo_tokens_pct"] else 0
    breakdown["potential_savings"] = {
        "demo_compression_50pct": f"{avg_demo_pct * 0.5:.1f}% token reduction",
        "demo_compression_75pct": f"{avg_demo_pct * 0.75:.1f}% token reduction",
        "observation_truncation": f"~{len(breakdown['largest_observations']) * 20000 // 4} tokens saved"
    }
    
    return breakdown


def create_optimized_config(base_config_path: Path, output_path: Path,
                            demo_compression: str = "moderate") -> None:
    """
    Create an optimized config with context-saving settings.
    
    Args:
        base_config_path: Original config
        output_path: Output path for optimized config
        demo_compression: "light" (5k chars), "moderate" (2k), "aggressive" (1k)
    """
    import yaml
    
    with open(base_config_path) as f:
        config = yaml.safe_load(f)
    
    # Optimization 1: Reduce max_observation_length
    observation_limits = {"light": 50000, "moderate": 20000, "aggressive": 5000}
    if "agent" not in config:
        config["agent"] = {}
    if "templates" not in config["agent"]:
        config["agent"]["templates"] = {}
    
    config["agent"]["templates"]["max_observation_length"] = observation_limits.get(demo_compression, 20000)
    
    # Optimization 2: Add history processors
    if "history_processors" not in config["agent"]:
        config["agent"]["history_processors"] = []
    
    # Add last_n_observations if not present
    has_last_n = any(
        isinstance(hp, dict) and hp.get("type") == "last_n_observations"
        for hp in config["agent"]["history_processors"]
    )
    
    if not has_last_n:
        n_values = {"light": 7, "moderate": 5, "aggressive": 3}
        config["agent"]["history_processors"].append({
            "type": "last_n_observations",
            "n": n_values.get(demo_compression, 5)
        })
    
    # Optimization 3: Add closed_window processor to remove duplicate file views
    has_closed_window = any(
        isinstance(hp, dict) and hp.get("type") == "closed_window"
        for hp in config["agent"]["history_processors"]
    )
    
    if not has_closed_window:
        config["agent"]["history_processors"].append({
            "type": "closed_window"
        })
    
    # Optimization 4: Limit demonstrations
    demo_limits = {"light": 3, "moderate": 2, "aggressive": 1}
    # Note: This will be applied during runtime, just a comment in config
    config["_optimization_notes"] = {
        "max_demonstrations": demo_limits.get(demo_compression, 2),
        "compression_level": demo_compression
    }
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created optimized config at: {output_path}")
    print(f"  - Max observation: {config['agent']['templates']['max_observation_length']} chars")
    print(f"  - History processors: {len(config['agent']['history_processors'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize input context loading for SWE-agent token efficiency"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze token usage")
    analyze_parser.add_argument("--results_dir", type=str, required=True,
                               help="Directory with trajectory files")
    
    # Compress command
    compress_parser = subparsers.add_parser("compress", help="Compress demonstrations")
    compress_parser.add_argument("--demo_path", type=str, required=True,
                                help="Path to demonstration trajectory")
    compress_parser.add_argument("--output", type=str, required=True,
                                help="Output path for compressed demo")
    compress_parser.add_argument("--target_tokens", type=int, default=10000,
                                help="Target token count")
    
    # Optimize config command
    config_parser = subparsers.add_parser("optimize-config", help="Create optimized config")
    config_parser.add_argument("--base_config", type=str, required=True,
                              help="Base config file")
    config_parser.add_argument("--output", type=str, required=True,
                              help="Output path")
    config_parser.add_argument("--level", choices=["light", "moderate", "aggressive"],
                              default="moderate", help="Compression level")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        results = analyze_context_breakdown(Path(args.results_dir))
        print(json.dumps(results, indent=2, default=str))
    
    elif args.command == "compress":
        stats = compress_demo_for_efficiency(
            Path(args.demo_path),
            Path(args.output),
            args.target_tokens
        )
        print(f"Compression results:")
        print(f"  Original: {stats['original_tokens']:,} tokens ({stats['original_steps']} steps)")
        print(f"  Compressed: {stats['compressed_tokens']:,} tokens ({stats['compressed_steps']} steps)")
        print(f"  Reduction: {stats['reduction_pct']:.1f}%")
    
    elif args.command == "optimize-config":
        create_optimized_config(
            Path(args.base_config),
            Path(args.output),
            args.level
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

