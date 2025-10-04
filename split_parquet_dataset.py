#!/usr/bin/env python3
"""
Script to split parquet files into train/val/test sets while preserving directory structure.

Usage:
    python split_parquet_dataset.py --input_dir /path/to/input --output_dir /path/to/output
    
Example:
    python split_parquet_dataset.py --input_dir ./data --output_dir ./split_data --train_pct 0.7 --val_pct 0.15 --test_pct 0.15
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
from typing import Tuple, List
import multiprocessing as mp
from functools import partial


def get_split_indices(n_total: int, train_pct: float, val_pct: float, test_pct: float, 
                     random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate shuffled indices for train, validation, and test sets.
    
    Args:
        n_total: Total number of rows
        train_pct: Percentage for training set (0.0 to 1.0)
        val_pct: Percentage for validation set (0.0 to 1.0)
        test_pct: Percentage for test set (0.0 to 1.0)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    # Validate percentages
    total_pct = train_pct + val_pct + test_pct
    if not np.isclose(total_pct, 1.0, rtol=1e-5):
        raise ValueError(f"Percentages must sum to 1.0, got {total_pct}")
    
    # Generate shuffled indices
    np.random.seed(random_seed)
    indices = np.random.permutation(n_total)
    
    n_train = int(n_total * train_pct)
    n_val = int(n_total * val_pct)
    
    # Split the indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return train_indices, val_indices, test_indices


def find_parquet_files(input_dir: Path) -> list:
    """
    Recursively find all parquet files in the input directory.
    
    Args:
        input_dir: Path to the input directory
        
    Returns:
        List of Path objects for parquet files
    """
    parquet_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(Path(root) / file)
    return parquet_files


def create_output_path(input_file: Path, input_dir: Path, output_dir: Path, split_name: str) -> Path:
    """
    Create the output path for a split file, preserving directory structure.
    
    Args:
        input_file: Path to the input parquet file
        input_dir: Root input directory
        output_dir: Root output directory
        split_name: Name of the split (train, val, test)
        
    Returns:
        Path for the output file
    """
    # Get relative path from input_dir to input_file
    relative_path = input_file.relative_to(input_dir)
    
    # Create output path with split directory
    output_path = output_dir / split_name / relative_path
    
    return output_path


class ProgressiveParquetWriter:
    """
    A wrapper for ParquetWriter that handles progressive writing to parquet files.
    """
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.writer = None
        self.is_closed = False
    
    def write_batch(self, batch: pa.RecordBatch):
        """Write a batch of data to the parquet file."""
        if batch.num_rows == 0:
            return
        
        if self.writer is None:
            # Create new writer on first batch
            self.writer = pq.ParquetWriter(self.output_path, batch.schema)
        
        self.writer.write_batch(batch)
    
    def close(self):
        """Close the writer."""
        if self.writer is not None and not self.is_closed:
            self.writer.close()
            self.is_closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def process_parquet_file(input_dir: Path, output_dir: Path,
                        train_pct: float, val_pct: float, test_pct: float,
                        input_file: Path):
    """
    Process a single parquet file and split it into train/val/test using chunked processing.
    
    Args:
        input_file: Path to input parquet file
        input_dir: Root input directory
        output_dir: Root output directory
        train_pct: Training set percentage
        val_pct: Validation set percentage
        test_pct: Test set percentage
    """
    
    # Create output paths
    train_path = create_output_path(input_file, input_dir, output_dir, 'train')
    val_path = create_output_path(input_file, input_dir, output_dir, 'val')
    test_path = create_output_path(input_file, input_dir, output_dir, 'test')
    
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get total number of rows first
        parquet_file = pq.ParquetFile(input_file)
        n_total = parquet_file.metadata.num_rows
        
        if n_total == 0:
            print(f"Warning: {input_file} is empty, skipping...")
            return

        chunk_size = n_total // 500
        
        # Initialize progressive writers only for files we need to create
        train_writer = None
        val_writer = None
        test_writer = None
        
        try:
            train_writer = ProgressiveParquetWriter(train_path)
            val_writer = ProgressiveParquetWriter(val_path)
            test_writer = ProgressiveParquetWriter(test_path)
            
            # Process file in batches to keep memory usage low
            batch_reader = parquet_file.iter_batches(batch_size=chunk_size)
            
            for batch in batch_reader:
                r = np.random.random()
                if r < train_pct:
                    train_writer.write_batch(batch)
                elif r < train_pct + val_pct:
                    val_writer.write_batch(batch)
                else:
                    test_writer.write_batch(batch)
           
        finally:
            # Ensure all writers are properly closed
            if train_writer:
                train_writer.close()
            if val_writer:
                val_writer.close()
            if test_writer:
                test_writer.close()
        
        # Report final counts only for files we actually created
        final_counts = []
        final_train_count = pq.ParquetFile(train_path).metadata.num_rows
        final_counts.append(f"train={final_train_count}")
        final_val_count = pq.ParquetFile(val_path).metadata.num_rows
        final_counts.append(f"val={final_val_count}")
        final_test_count = pq.ParquetFile(test_path).metadata.num_rows
        final_counts.append(f"test={final_test_count}")
        
        if final_counts:
            print(f"  {input_file}: {', '.join(final_counts)}")
            
    except Exception as e:
        print(f"Error processing {input_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Split parquet files into train/val/test sets')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing parquet files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for split datasets')
    parser.add_argument('--train_pct', type=float, default=0.7,
                       help='Training set percentage (default: 0.7)')
    parser.add_argument('--val_pct', type=float, default=0.15,
                       help='Validation set percentage (default: 0.15)')
    parser.add_argument('--test_pct', type=float, default=0.15,
                       help='Test set percentage (default: 0.15)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate arguments
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return
    
    # Validate percentages
    total_pct = args.train_pct + args.val_pct + args.test_pct
    if not np.isclose(total_pct, 1.0, rtol=1e-5):
        print(f"Error: Percentages must sum to 1.0, got {total_pct}")
        return
    
    if any(pct < 0 or pct > 1 for pct in [args.train_pct, args.val_pct, args.test_pct]):
        print("Error: All percentages must be between 0 and 1")
        return
    
    # Find all parquet files
    print(f"Searching for parquet files in {input_dir}...")
    parquet_files = find_parquet_files(input_dir)
    
    if not parquet_files:
        print("No parquet files found in the input directory")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    print(f"Split percentages: train={args.train_pct}, val={args.val_pct}, test={args.test_pct}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each parquet file using mp.Pool
    with mp.Pool(mp.cpu_count()) as pool:
        iter = pool.imap_unordered(
            partial(process_parquet_file,
            input_dir, output_dir,
            args.train_pct, args.val_pct, args.test_pct),
            parquet_files
        )
        for _ in iter:
            pass

    print("Dataset splitting completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Directory structure:")
    print(f"  {output_dir}/train/")
    print(f"  {output_dir}/val/")
    print(f"  {output_dir}/test/")


if __name__ == "__main__":
    main()
