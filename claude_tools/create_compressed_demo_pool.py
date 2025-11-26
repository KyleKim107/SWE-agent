#!/usr/bin/env python3
"""
Create a compressed demonstration pool for token-efficient evaluation.

This script processes the existing demonstration pool and creates compressed
versions that are 5-10x more token efficient.

Usage:
    python create_compressed_demo_pool.py \
        --input ./demonstration_pool \
        --output ./demonstration_pool_compressed \
        --target_tokens 10000

The compressed pool maintains the same structure and index file format,
making it a drop-in replacement for the original pool.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Any
from copy import deepcopy


def compress_observation(content: str, max_chars: int = 5000) -> str:
    """
    Intelligently compress an observation to save tokens.
    
    Strategies:
    1. Truncate directory listings to summary
    2. Keep only key lines from code dumps
    3. Preserve error messages and test output summaries
    """
    if len(content) <= max_chars:
        return content
    
    # Check for directory listings (very low value)
    if _is_directory_listing(content):
        return _summarize_directory_listing(content, max_chars)
    
    # Check for code file dumps
    if _is_code_dump(content):
        return _summarize_code_dump(content, max_chars)
    
    # Check for test output
    if _is_test_output(content):
        return _summarize_test_output(content, max_chars)
    
    # Default: truncate with indicator
    return content[:max_chars] + f"\n[...{len(content) - max_chars} chars truncated]"


def _is_directory_listing(content: str) -> bool:
    """Check if content is a directory listing."""
    lines = content.split('\n')[:50]
    dir_patterns = [
        r'^d[rwx-]+\s+\d+\s+\w+\s+\w+',
        r'^-[rwx-]+\s+\d+\s+\w+\s+\w+',
        r'^total \d+',
    ]
    dir_line_count = sum(1 for line in lines if any(re.match(p, line) for p in dir_patterns))
    return dir_line_count > 10


def _summarize_directory_listing(content: str, max_chars: int) -> str:
    """Summarize a directory listing."""
    lines = content.split('\n')
    
    # Count files by type
    py_files = sum(1 for line in lines if '.py' in line and not line.startswith('d'))
    dirs = sum(1 for line in lines if line.startswith('d'))
    total = len([l for l in lines if l.strip()])
    
    return f"[Directory listing: {total} items, {dirs} dirs, {py_files} Python files]"


def _is_code_dump(content: str) -> bool:
    """Check if content is a code file dump (from cat -n or str_replace_editor view)."""
    patterns = [
        r'^\s*\d+[\|\t]',  # Line numbers
        r"Here's the result of running `cat",
        r'\[File:.*\(\d+ lines total\)\]',
    ]
    return any(re.search(p, content[:1000]) for p in patterns)


def _summarize_code_dump(content: str, max_chars: int) -> str:
    """Summarize a code dump keeping key structures."""
    lines = content.split('\n')
    
    # Keep header (file info)
    header_lines = []
    for line in lines[:10]:
        if '[File:' in line or "Here's the result" in line:
            header_lines.append(line)
            break
    
    # Find key structures (class/function definitions)
    key_patterns = [
        (r'^\s*\d*\|?\s*class\s+\w+', 'class'),
        (r'^\s*\d*\|?\s*def\s+\w+', 'function'),
        (r'^\s*\d*\|?\s*async\s+def\s+\w+', 'async function'),
    ]
    
    structures_found = []
    for i, line in enumerate(lines):
        for pattern, stype in key_patterns:
            if re.match(pattern, line):
                # Extract the definition line and a bit of context
                structures_found.append(line.strip())
                break
    
    result = '\n'.join(header_lines)
    if structures_found:
        result += f"\n[Key structures: {len(structures_found)} found]\n"
        result += '\n'.join(structures_found[:10])
        if len(structures_found) > 10:
            result += f"\n[...and {len(structures_found) - 10} more]"
    
    # If still under max_chars, add some actual content
    remaining = max_chars - len(result) - 50
    if remaining > 500:
        result += f"\n\n[First {remaining} chars of content:]\n"
        content_start = content.find('\n') + 1 if '\n' in content else 0
        result += content[content_start:content_start + remaining]
    
    return result


def _is_test_output(content: str) -> bool:
    """Check if content is test output."""
    patterns = [
        r'PASSED|FAILED|ERROR',
        r'test.*::\w+',
        r'===.*===',
        r'\d+ passed',
    ]
    return any(re.search(p, content, re.IGNORECASE) for p in patterns)


def _summarize_test_output(content: str, max_chars: int) -> str:
    """Summarize test output keeping key results."""
    lines = content.split('\n')
    
    # Find summary lines
    summary_patterns = [
        r'\d+ passed',
        r'\d+ failed',
        r'\d+ error',
        r'PASSED',
        r'FAILED',
        r'===.*===',
    ]
    
    summary_lines = []
    for line in lines:
        if any(re.search(p, line, re.IGNORECASE) for p in summary_patterns):
            summary_lines.append(line)
    
    result = "[Test output summary]\n"
    result += '\n'.join(summary_lines[:20])
    
    # If failed, try to capture error message
    if 'FAILED' in content or 'Error' in content:
        # Find error context
        for i, line in enumerate(lines):
            if 'Error' in line or 'Exception' in line:
                start = max(0, i - 2)
                end = min(len(lines), i + 5)
                result += '\n\n[Error context:]\n'
                result += '\n'.join(lines[start:end])
                break
    
    return result[:max_chars]


def identify_essential_steps(history: list) -> set:
    """
    Identify essential steps in a demonstration.
    Keep: first steps, file edits, test runs, submit.
    Remove: exploration, repeated file views, debugging.
    """
    essential = set()
    
    for i, entry in enumerate(history):
        content = str(entry.get("content", "")).lower()
        action = str(entry.get("action", "")).lower()
        role = entry.get("role", "")
        msg_type = entry.get("message_type", "")
        
        # Always keep system prompt
        if role == "system":
            essential.add(i)
            continue
        
        # First few entries (problem understanding)
        if i < 4:
            essential.add(i)
            continue
        
        # File modification steps (the actual fix)
        if any(kw in action for kw in ["str_replace", "edit", "create", "write"]):
            essential.add(i)
            # Also include next entry (observation)
            if i + 1 < len(history):
                essential.add(i + 1)
            continue
        
        # Test/verification steps
        if any(kw in action for kw in ["test", "pytest", "python"]) and "view" not in action:
            essential.add(i)
            if i + 1 < len(history):
                essential.add(i + 1)
            continue
        
        # Submit step
        if "submit" in action:
            essential.add(i)
            continue
    
    # Always include last few steps (final state)
    for i in range(max(0, len(history) - 4), len(history)):
        essential.add(i)
    
    return essential


def compress_demonstration(traj_path: Path, max_obs_chars: int = 5000) -> dict:
    """
    Compress a demonstration trajectory.
    
    Steps:
    1. Filter to essential steps only
    2. Compress each observation
    3. Return compressed trajectory
    """
    traj = json.loads(traj_path.read_text())
    history = traj.get("history", [])
    
    # Identify essential steps
    essential_indices = identify_essential_steps(history)
    
    compressed_history = []
    for i, entry in enumerate(history):
        # Skip non-essential steps
        if i not in essential_indices:
            continue
        
        entry = deepcopy(entry)
        
        # Compress observations
        if entry.get("message_type") == "observation":
            content = entry.get("content", "")
            if len(content) > max_obs_chars:
                entry["content"] = compress_observation(content, max_obs_chars)
        
        compressed_history.append(entry)
    
    # Create compressed trajectory
    compressed = {
        "history": compressed_history,
        "info": traj.get("info", {}),
        "_compression": {
            "original_steps": len(history),
            "compressed_steps": len(compressed_history),
            "max_obs_chars": max_obs_chars
        }
    }
    
    return compressed


def compress_demo_pool(input_dir: Path, output_dir: Path, 
                       max_obs_chars: int = 5000) -> dict:
    """
    Compress an entire demonstration pool.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original index
    index_file = input_dir / "demonstration_index.json"
    with open(index_file) as f:
        index_data = json.load(f)
    
    demonstrations = index_data.get("demonstrations", [])
    
    stats = {
        "total": len(demonstrations),
        "compressed": 0,
        "original_tokens": 0,
        "compressed_tokens": 0
    }
    
    compressed_demos = []
    
    for demo in demonstrations:
        instance_id = demo["instance_id"]
        
        # Find trajectory file
        traj_path = input_dir / instance_id / f"{instance_id}.traj"
        if not traj_path.exists():
            print(f"⚠️  Skipping {instance_id}: trajectory not found")
            continue
        
        # Compress
        original_traj = json.loads(traj_path.read_text())
        original_chars = sum(len(h.get("content", "")) for h in original_traj.get("history", []))
        
        compressed = compress_demonstration(traj_path, max_obs_chars)
        compressed_chars = sum(len(h.get("content", "")) for h in compressed.get("history", []))
        
        stats["original_tokens"] += original_chars // 4
        stats["compressed_tokens"] += compressed_chars // 4
        stats["compressed"] += 1
        
        # Save compressed trajectory
        out_instance_dir = output_dir / instance_id
        out_instance_dir.mkdir(parents=True, exist_ok=True)
        
        out_traj_path = out_instance_dir / f"{instance_id}.traj"
        out_traj_path.write_text(json.dumps(compressed, indent=2))
        
        # Copy other files (patch, pred)
        for ext in [".patch", ".pred"]:
            src = input_dir / instance_id / f"{instance_id}{ext}"
            if src.exists():
                (out_instance_dir / f"{instance_id}{ext}").write_text(src.read_text())
        
        # Update demo info
        demo_copy = demo.copy()
        demo_copy["trajectory_path"] = str(out_traj_path)
        demo_copy["compressed"] = True
        compressed_demos.append(demo_copy)
        
        reduction = 100 * (1 - compressed_chars / max(1, original_chars))
        print(f"✅ {instance_id}: {original_chars//4:,} → {compressed_chars//4:,} tokens ({reduction:.0f}% reduction)")
    
    # Save compressed index
    compressed_index = {
        "demonstrations": compressed_demos,
        "enriched": index_data.get("enriched", True),
        "compressed": True,
        "compression_settings": {
            "max_obs_chars": max_obs_chars
        },
        "stats": stats
    }
    
    (output_dir / "demonstration_index.json").write_text(json.dumps(compressed_index, indent=2))
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Create compressed demonstration pool for token efficiency"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./demonstration_pool",
        help="Input demonstration pool directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./demonstration_pool_compressed",
        help="Output directory for compressed pool"
    )
    parser.add_argument(
        "--max_obs_chars",
        type=int,
        default=5000,
        help="Maximum characters per observation (default: 5000 = ~1.25k tokens)"
    )
    
    args = parser.parse_args()
    
    print(f"🔧 Creating compressed demonstration pool...")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Max observation chars: {args.max_obs_chars}")
    print()
    
    stats = compress_demo_pool(
        Path(args.input),
        Path(args.output),
        args.max_obs_chars
    )
    
    print()
    print("=" * 60)
    print("COMPRESSION COMPLETE")
    print("=" * 60)
    print(f"Total demonstrations: {stats['total']}")
    print(f"Successfully compressed: {stats['compressed']}")
    print(f"Original tokens: {stats['original_tokens']:,}")
    print(f"Compressed tokens: {stats['compressed_tokens']:,}")
    if stats['original_tokens'] > 0:
        reduction = 100 * (1 - stats['compressed_tokens'] / stats['original_tokens'])
        print(f"Overall reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()

