#!/usr/bin/env python3
"""
Script to create a new dataset of chess games from existing parquet files.
Randomly samples rows from existing parquet files and splits them into train/validation/test sets.
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_parquet_files(root_dir: Path) -> List[Path]:
    """
    Find all data.parquet files in the two-level directory structure.
    
    Args:
        root_dir: Top-level directory containing Elo-based subdirectories
        
    Returns:
        List of paths to parquet files
    """
    parquet_files = []
    
    for elo_w_dir in sorted(root_dir.iterdir()):
        if not elo_w_dir.is_dir():
            continue
        for elo_b_dir in sorted(elo_w_dir.iterdir()):
            if not elo_b_dir.is_dir():
                continue
            pfile = elo_b_dir / "data.parquet"
            if pfile.exists():
                parquet_files.append(pfile)
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    return parquet_files


def filter_and_sample_parquet(
    file_path: Path, 
    min_timectl: int, 
    min_games: int,
    random_seed: int = None
) -> pd.DataFrame:
    """
    Filter parquet file by timeCtl and sample rows.
    
    Args:
        file_path: Path to parquet file
        min_timectl: Minimum timeCtl value to include
        min_games: Minimum number of games to sample
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with sampled rows
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Read the parquet file
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return pd.DataFrame()
    
    # Filter by timeCtl
    if 'timeCtl' in df.columns:
        filtered_df = df[df['timeCtl'] >= min_timectl].copy()
    else:
        logger.warning(f"timeCtl column not found in {file_path}")
        filtered_df = df.copy()
    
    if len(filtered_df) == 0:
        logger.warning(f"No rows meet timeCtl >= {min_timectl} in {file_path}")
        return pd.DataFrame()
    
    # Sample rows (with replacement if necessary)
    if len(filtered_df) >= min_games:
        # Sample without replacement
        sampled_df = filtered_df.sample(n=min_games, random_state=random_seed)
    else:
        # Sample with replacement to reach minimum
        sampled_df = filtered_df.sample(n=min_games, replace=True, random_state=random_seed)
        logger.info(f"Sampled with replacement from {file_path}: {len(filtered_df)} -> {min_games} rows")
    
    return sampled_df


def assign_split(num_rows: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> List[str]:
    """
    Assign each row to train, validation, or test split.
    
    Args:
        num_rows: Number of rows to assign
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        
    Returns:
        List of split assignments
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    
    splits = []
    splits.extend(['train'] * int(num_rows * train_ratio))
    splits.extend(['val'] * int(num_rows * val_ratio))
    splits.extend(['test'] * int(num_rows * test_ratio))
    
    # Handle any remaining rows due to rounding
    while len(splits) < num_rows:
        splits.append('train')
    
    # Shuffle the assignments
    random.shuffle(splits)
    return splits[:num_rows]


def create_dataset(
    input_dir: str,
    output_dir: str,
    min_timectl: int,
    min_games: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42
):
    """
    Create new dataset by sampling from existing parquet files.
    
    Args:
        input_dir: Directory containing input parquet files
        output_dir: Directory to save output parquet files
        min_timectl: Minimum timeCtl value to include
        min_games: Minimum number of games per input file to sample
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        random_seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(random_seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    parquet_files = find_parquet_files(input_path)
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    
    # Initialize data containers for each split
    train_data = []
    val_data = []
    test_data = []
    
    # Process each parquet file
    for i, file_path in enumerate(tqdm(parquet_files, desc="Processing files")):
        logger.info(f"Processing {i} of {len(parquet_files)}")
        # Sample data from this file
        sampled_df = filter_and_sample_parquet(
            file_path, min_timectl, min_games, random_seed + i
        )
        
        if len(sampled_df) == 0:
            continue
        
        # Assign splits randomly
        splits = assign_split(len(sampled_df), train_ratio, val_ratio)
        sampled_df['split'] = splits
        
        # Separate into splits
        train_subset = sampled_df[sampled_df['split'] == 'train'].drop('split', axis=1)
        val_subset = sampled_df[sampled_df['split'] == 'val'].drop('split', axis=1)
        test_subset = sampled_df[sampled_df['split'] == 'test'].drop('split', axis=1)
        
        if len(train_subset) > 0:
            train_data.append(train_subset)
        if len(val_subset) > 0:
            val_data.append(val_subset)
        if len(test_subset) > 0:
            test_data.append(test_subset)
    
    # Combine and shuffle data for each split
    logger.info("Combining and shuffling data...")
    
    splits_to_save = []
    if train_data:
        train_df = pd.concat(train_data, ignore_index=True)
        train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        splits_to_save.append(('train', train_df))
        logger.info(f"Training set: {len(train_df)} rows")
    
    if val_data:
        val_df = pd.concat(val_data, ignore_index=True)
        val_df = val_df.sample(frac=1, random_state=random_seed + 1).reset_index(drop=True)
        splits_to_save.append(('val', val_df))
        logger.info(f"Validation set: {len(val_df)} rows")
    
    if test_data:
        test_df = pd.concat(test_data, ignore_index=True)
        test_df = test_df.sample(frac=1, random_state=random_seed + 2).reset_index(drop=True)
        splits_to_save.append(('test', test_df))
        logger.info(f"Test set: {len(test_df)} rows")
    
    # Save each split as a parquet file
    logger.info("Saving parquet files...")
    for split_name, split_df in splits_to_save:
        output_file = output_path / f"{split_name}.parquet"
        split_df.to_parquet(output_file, index=False)
        logger.info(f"Saved {split_name} set to {output_file}")
    
    logger.info("Dataset creation completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Create a new chess dataset by sampling from existing parquet files"
    )
    parser.add_argument(
        "input_dir",
        help="Top-level directory containing parquet files in Elo-based subdirectories"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory to store the new parquet files"
    )
    parser.add_argument(
        "--min-timectl",
        type=int,
        default=60,
        help="Minimum timeCtl value to include (default: 60)"
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=1000,
        help="Minimum number of games per parquet file to sample (default: 1000)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training set (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for validation set (default: 0.1)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")
    
    create_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_timectl=args.min_timectl,
        min_games=args.min_games,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()