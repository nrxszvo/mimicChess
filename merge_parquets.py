#!/usr/bin/env python3
"""
Script to merge parquet files from a two-level directory structure based on minimum threshold.

Directory structure example:
Top level: 1000, 1500, 2000, 2500, 2500+
Second level: 1000/1000/data.parquet, 1000/1500/data.parquet, ..., 2500+/2500+/data.parquet

The script performs two types of merging:
1. Second-level merging: Within each first-level directory, files above the threshold 
   are merged into threshold+ second-level directories
2. First-level merging: First-level directories above the threshold are merged into 
   a new threshold+ first-level directory

For example, with threshold 2000:
- Files in 2000/ second-level directories remain unchanged
- Files in 2500/, 2500+/, etc. second-level directories are merged into 2000+/ second-level directories
- First-level directories 2500/, 2500+/, etc. are merged into a new 2000+/ first-level directory

Usage:
    python merge_parquets.py <root_dir> <min_threshold> [--dry-run]
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import pyarrow as pa
import pyarrow.parquet as pq


def parse_directory_name(dir_name: str) -> Tuple[int, bool]:
    """
    Parse directory name to extract numeric value and plus suffix.
    
    Args:
        dir_name: Directory name (e.g., "1000", "2500+")
        
    Returns:
        Tuple of (numeric_value, has_plus_suffix)
    """
    match = re.match(r'^(\d+)(\+?)$', dir_name)
    if not match:
        raise ValueError(f"Invalid directory name format: {dir_name}")
    
    numeric_value = int(match.group(1))
    has_plus = bool(match.group(2))
    
    return numeric_value, has_plus


def get_sorted_directories(root_dir: Path) -> List[Tuple[Path, int, bool]]:
    """
    Get all first-level directories sorted by their numeric values.
    
    Args:
        root_dir: Root directory path
        
    Returns:
        List of tuples (path, numeric_value, has_plus_suffix)
    """
    directories = []
    
    for item in root_dir.iterdir():
        if item.is_dir():
            try:
                numeric_value, has_plus = parse_directory_name(item.name)
                directories.append((item, numeric_value, has_plus))
            except ValueError:
                print(f"Skipping directory with invalid name: {item.name}")
                continue
    
    # Sort by numeric value, then by plus suffix (non-plus first)
    directories.sort(key=lambda x: (x[1], x[2]))
    
    return directories


def find_parquet_files_to_merge(
    first_level_dir: Path, 
    min_threshold: int
) -> Tuple[List[Path], List[Path]]:
    """
    Find parquet files to merge based on threshold, separating threshold and above-threshold files.
    
    Args:
        first_level_dir: First level directory path
        min_threshold: Minimum threshold value
        
    Returns:
        Tuple of (threshold_files, above_threshold_files)
        - threshold_files: Files from directories exactly matching the threshold
        - above_threshold_files: Files from directories above the threshold
    """
    threshold_files = []
    above_threshold_files = []
    
    for second_level in first_level_dir.iterdir():
        if not second_level.is_dir():
            continue
            
        try:
            numeric_value, has_plus = parse_directory_name(second_level.name)
            if numeric_value >= min_threshold:
                parquet_file = second_level / "data.parquet"
                if parquet_file.exists():
                    if numeric_value == min_threshold and not has_plus:
                        threshold_files.append(parquet_file)
                    else:
                        above_threshold_files.append(parquet_file)
                else:
                    print(f"Warning: data.parquet not found in {second_level}")
        except ValueError:
            print(f"Skipping second-level directory with invalid name: {second_level.name}")
            continue
    
    return threshold_files, above_threshold_files


def merge_parquet_files(parquet_files: List[Path]) -> Optional[pa.Table]:
    """
    Merge multiple parquet files into a single PyArrow table.
    
    Args:
        parquet_files: List of parquet file paths to merge
        
    Returns:
        Merged PyArrow table or None if no files to merge
    """
    if not parquet_files:
        return None
    
    tables = []
    total_rows = 0
    
    print(f"Merging {len(parquet_files)} parquet files...")
    
    for file_path in parquet_files:
        try:
            table = pq.read_table(file_path)
            tables.append(table)
            total_rows += len(table)
            print(f"  - {file_path}: {len(table)} rows")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not tables:
        print("No valid parquet files found to merge")
        return None
    
    # Concatenate all tables
    merged_table = pa.concat_tables(tables)
    print(f"Total merged rows: {total_rows}")
    
    return merged_table



def main():
    parser = argparse.ArgumentParser(
        description="Merge parquet files based on directory structure and minimum threshold"
    )
    parser.add_argument(
        "root_dir", 
        type=str, 
        help="Root directory containing the two-level directory structure"
    )
    parser.add_argument(
        "min_threshold", 
        type=int, 
        help="Minimum threshold for second-level directories to include"
    )
    parser.add_argument(
        "--output-suffix", 
        type=str, 
        default="merged",
        help="Suffix for the output parquet file (default: merged)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without actually creating files"
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        print(f"Error: Root directory {root_dir} does not exist")
        return 1
    
    # Get sorted first-level directories
    directories = get_sorted_directories(root_dir)
    if not directories:
        print("No valid directories found in root directory")
        return 1
    
    print(f"Found {len(directories)} first-level directories:")
    for dir_path, numeric_value, has_plus in directories:
        suffix = "+" if has_plus else ""
        print(f"  - {dir_path.name} ({numeric_value}{suffix})")
    
    print(f"\nProcessing with minimum threshold: {args.min_threshold}")
    
    # Separate first-level directories by threshold
    threshold_first_level = []
    above_threshold_first_level = []
    
    for first_level_path, first_level_value, first_level_plus in directories:
        if first_level_value <= args.min_threshold and not first_level_plus:
            threshold_first_level.append((first_level_path, first_level_value, first_level_plus))
        elif first_level_value >= args.min_threshold:
            above_threshold_first_level.append((first_level_path, first_level_value, first_level_plus))
    
    print(f"\nFirst-level directories at threshold ({args.min_threshold}): {len(threshold_first_level)}")
    print(f"First-level directories above threshold: {len(above_threshold_first_level)}")
    
    # Process threshold first-level directories (keep as-is, but process second-level merging)
    for first_level_path, first_level_value, first_level_plus in threshold_first_level:
        print(f"\n--- Processing threshold first-level directory: {first_level_path.name} ---")
        
        # Find parquet files to merge in this first-level directory
        threshold_files, above_threshold_files = find_parquet_files_to_merge(first_level_path, args.min_threshold)
        
        if not threshold_files and not above_threshold_files:
            print(f"  No parquet files found meeting threshold {args.min_threshold}")
            continue
        
        print(f"  Found {len(threshold_files)} threshold files and {len(above_threshold_files)} above-threshold files")
        
        # Handle threshold directory (keep existing files)
        if threshold_files:
            print(f"  Threshold files ({args.min_threshold}):")
            for pf in threshold_files:
                print(f"    - {pf}")
            
            if args.dry_run:
                print(f"  [DRY RUN] Threshold files already exist in: {first_level_path}/{args.min_threshold}/")
            else:
                print(f"  Threshold files already exist in: {first_level_path}/{args.min_threshold}/")
        
        # Handle above-threshold files (merge into threshold+ directory)
        if above_threshold_files:
            print("  Above-threshold files:")
            for pf in above_threshold_files:
                print(f"    - {pf}")
            
            if args.dry_run:
                print(f"  [DRY RUN] Would merge above-threshold files into: {first_level_path}/{args.min_threshold}+/data.parquet")
            else:
                # Merge the above-threshold parquet files
                merged_table = merge_parquet_files(above_threshold_files)
                if merged_table is not None:
                    # Create output directory for merged above-threshold files
                    output_dir = first_level_path / f"{args.min_threshold}+"
                    output_dir.mkdir(exist_ok=True)
                    
                    # Write merged parquet file
                    output_file = output_dir / "data.parquet"
                    pq.write_table(merged_table, output_file)
                    
                    print(f"  Above-threshold files merged and saved to: {output_file}")
                    print(f"  Total rows in merged file: {len(merged_table)}")
    
    # Process above-threshold first-level directories (merge into threshold+ first-level directory)
    if above_threshold_first_level:
        print("\n--- Processing above-threshold first-level directories ---")
        print("Above-threshold first-level directories:")
        for first_level_path, first_level_value, first_level_plus in above_threshold_first_level:
            suffix = "+" if first_level_plus else ""
            print(f"  - {first_level_path.name} ({first_level_value}{suffix})")
        
        # Create merged first-level directory
        merged_first_level_dir = root_dir / f"{args.min_threshold}+"
        
        if args.dry_run:
            print(f"  [DRY RUN] Would merge above-threshold first-level directories into: {merged_first_level_dir}/")
        else:
            merged_first_level_dir.mkdir(exist_ok=True)
            print(f"  Created merged first-level directory: {merged_first_level_dir}/")
        
        # Collect all second-level directories from above-threshold first-level directories
        all_second_level_dirs = set()
        for first_level_path, _, _ in above_threshold_first_level:
            for second_level in first_level_path.iterdir():
                if second_level.is_dir():
                    all_second_level_dirs.add(second_level.name)
        
        # Separate second-level directories by threshold (same logic as before)
        threshold_second_level_files = {}  # second_level_name -> list of parquet files
        above_threshold_second_level_files = []  # list of parquet files
        
        for second_level_name in sorted(all_second_level_dirs):
            try:
                numeric_value, has_plus = parse_directory_name(second_level_name)
                
                # Collect all parquet files from this second-level across all above-threshold first-level dirs
                parquet_files = []
                for first_level_path, _, _ in above_threshold_first_level:
                    second_level_path = first_level_path / second_level_name
                    if second_level_path.is_dir():
                        parquet_file = second_level_path / "data.parquet"
                        if parquet_file.exists():
                            parquet_files.append(parquet_file)
                
                if parquet_files:
                    if numeric_value <= args.min_threshold and not has_plus:
                        # This is the threshold second-level directory
                        threshold_second_level_files[second_level_name] = parquet_files
                    elif numeric_value >= args.min_threshold:
                        # This is above threshold, add to merge list
                        above_threshold_second_level_files.extend(parquet_files)
                        
            except ValueError:
                print(f"    Skipping second-level directory with invalid name: {second_level_name}")
                continue
        
        # Process threshold second-level directories (create individual directories)
        for second_level_name, parquet_files in threshold_second_level_files.items():
            print(f"\n  Processing threshold second-level directory: {second_level_name}")
            for pf in parquet_files:
                print(f"    - {pf}")
            
            if args.dry_run:
                print(f"    [DRY RUN] Would merge {len(parquet_files)} files into: {merged_first_level_dir}/{second_level_name}/data.parquet")
            else:
                # Merge the parquet files
                merged_table = merge_parquet_files(parquet_files)
                if merged_table is not None:
                    # Create output directory
                    output_dir = merged_first_level_dir / second_level_name
                    output_dir.mkdir(exist_ok=True)
                    
                    # Write merged parquet file
                    output_file = output_dir / "data.parquet"
                    pq.write_table(merged_table, output_file)
                    
                    print(f"    Merged {len(parquet_files)} files into: {output_file}")
                    print(f"    Total rows in merged file: {len(merged_table)}")
        
        # Process above-threshold second-level directories (merge into threshold+ directory)
        if above_threshold_second_level_files:
            print("\n  Processing above-threshold second-level directories")
            print("  Above-threshold second-level files:")
            for pf in above_threshold_second_level_files:
                print(f"    - {pf}")
            
            if args.dry_run:
                print(f"    [DRY RUN] Would merge {len(above_threshold_second_level_files)} files into: {merged_first_level_dir}/{args.min_threshold}+/data.parquet")
            else:
                # Merge the above-threshold parquet files
                merged_table = merge_parquet_files(above_threshold_second_level_files)
                if merged_table is not None:
                    # Create output directory for merged above-threshold files
                    output_dir = merged_first_level_dir / f"{args.min_threshold}+"
                    output_dir.mkdir(exist_ok=True)
                    
                    # Write merged parquet file
                    output_file = output_dir / "data.parquet"
                    pq.write_table(merged_table, output_file)
                    
                    print(f"    Above-threshold files merged and saved to: {output_file}")
                    print(f"    Total rows in merged file: {len(merged_table)}")
    
    print("\nMerging complete!")
    return 0


if __name__ == "__main__":
    exit(main())
