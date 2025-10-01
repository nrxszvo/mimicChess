#!/usr/bin/env python3
"""
Script to create train/val/test dataset splits from chess parquet storage files.

This script processes a two-level directory structure of parquet files and creates
balanced train/val/test splits by round-robin sampling from all storage files.
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def discover_storage_files(root_dir: Path) -> List[Path]:
    """
    Discover all parquet files in the two-level directory structure.
    
    Args:
        root_dir: Root directory containing first-level folders (1000, 1500, etc.)
        
    Returns:
        List of paths to data.parquet files
    """
    storage_files = []
    
    for first_level in sorted(root_dir.iterdir()):
        if not first_level.is_dir():
            continue
            
        for second_level in sorted(first_level.iterdir()):
            if not second_level.is_dir():
                continue
                
            parquet_file = second_level / "data.parquet"
            if parquet_file.exists():
                storage_files.append(parquet_file)
    
    if not storage_files:
        raise FileNotFoundError(f"No data.parquet files found under {root_dir}")
    
    print(f"Found {len(storage_files)} storage files")
    return storage_files


def count_filtered_rows(file_path: Path, min_timectl: int) -> int:
    """
    Count rows in a parquet file that meet the timeCtl filter requirement.
    
    Args:
        file_path: Path to the parquet file
        min_timectl: Minimum timeCtl value required
        
    Returns:
        Number of rows meeting the filter criteria
    """
    try:
        dataset = ds.dataset(str(file_path), format="parquet")
        filter_expr = ds.field("timeCtl") >= pa.scalar(min_timectl, pa.int16())
        
        try:
            count = dataset.count_rows(filter=filter_expr)
        except Exception:
            # Fallback if driver lacks count_rows filter support
            table = dataset.to_table(columns=["timeCtl"])
            timectl_col = table.column("timeCtl")
            count = sum(
                1 for v in timectl_col.to_pylist() 
                if v is not None and v >= min_timectl
            )
        
        return count
    except Exception as e:
        print(f"Error counting rows in {file_path}: {e}")
        return 0


def find_max_filtered_rows(storage_files: List[Path], min_timectl: int, min_rows: int, max_repeats: int) -> int:
    """
    Find the maximum number of filtered rows across all storage files.
    
    Args:
        storage_files: List of storage file paths
        min_timectl: Minimum timeCtl value required
        
    Returns:
        Maximum number of filtered rows in any single file
    """
    max_rows = 0
    file_counts = {}
    
    print("Counting filtered rows in each storage file...")
    
    for file_path in storage_files:
        count = count_filtered_rows(file_path, min_timectl)
        if count < min_rows:
            print(f"WARNING: Not enough filtered rows in {file_path}: {count} < {min_rows}")
        else:
            file_counts[file_path] = count 
            max_rows = max(max_rows, count)
            print(f"{file_path}: {count} filtered rows")

    for fp in file_counts:
        file_counts[fp] = min(max_rows, max_repeats*file_counts[fp])
        
    print(f"Maximum filtered rows in any file: {max_rows}")
    return max_rows, file_counts


class FilteredRowIterator:
    """Memory-efficient iterator that streams filtered rows from a parquet file, cycling when exhausted."""
    
    def __init__(self, file_path: Path, min_timectl: int, max_rows: int, batch_size: int = 1000):
        self.file_path = file_path
        self.min_timectl = min_timectl
        self.filter_expr = ds.field("timeCtl") >= pa.scalar(min_timectl, pa.int16())
        self.batch_size = batch_size
        self.current_batch = None
        self.current_batch_index = 0
        self.global_index = 0
        self.total_filtered_rows = 0
        self.max_rows = max_rows
        self.total_rows = 0
        self.dataset = ds.dataset(str(self.file_path), format="parquet")
        # create schema that is subset of dataset schema
        self._columns = ["moves", "clk", "result", "welo", "belo", "timeCtl", "increment"]
        self._schema = pa.schema({field.name: field.type for field in self.dataset.schema if field.name in self._columns})
        self._count_total_rows()
        self._load_next_batch()
    
    def _count_total_rows(self):
        """Count total filtered rows for cycling logic."""
        try:
            self.total_filtered_rows = self.dataset.count_rows(filter=self.filter_expr)
        except Exception:
            # Fallback counting method
            scanner = self.dataset.scanner(filter=self.filter_expr, columns=["timeCtl"])
            self.total_filtered_rows = len(scanner.to_table())

        print(f"Found {self.total_filtered_rows} filtered rows in {self.file_path}")
    
    def _load_next_batch(self):
        """Load the next batch of filtered rows."""
        if self.total_filtered_rows == 0 or self.total_rows >= self.max_rows:
            self.current_batch = None
            return

        # Calculate which batch to load based on global index
        batch_start = (self.global_index // self.batch_size) * self.batch_size

        # Create scanner with filter and limit
        scanner = self.dataset.scanner(
            filter=self.filter_expr,
            columns=self._columns,
            #batch_size=self.batch_size,
            #offset=batch_start % self.total_filtered_rows
        )
        # create empty batch with number of arrays equal to schema
        arrays = [pa.array([], type=field.type) for field in self._schema]
        self.current_batch = pa.Table.from_arrays(arrays, schema=self._schema)
        while len(self.current_batch) < self.batch_size:
            indices = pa.array(range(batch_start, min(batch_start + self.batch_size, self.total_filtered_rows)))
            # concatenate batch
            self.current_batch = pa.concat_tables([self.current_batch, scanner.take(indices)])
            batch_start = (batch_start + len(indices)) % self.total_filtered_rows
            self.total_rows += len(indices)
            if self.total_rows >= self.max_rows:
                break
            
        self.current_batch_index = 0
    
    def get_next_row(self) -> Optional[pa.Table]:
        """Get the next filtered row, cycling back to start if needed."""
        if self.total_filtered_rows == 0:
            return None
        
        # Check if we need to load a new batch
        if (self.current_batch is None or 
            self.current_batch_index >= len(self.current_batch)):
            self._load_next_batch()
            
        if self.current_batch is None or len(self.current_batch) == 0:
            return None
        
        # Get current row from batch
        row = self.current_batch.slice(self.current_batch_index, 1)
        
        # Advance indices
        self.current_batch_index += 1
        self.global_index = (self.global_index + 1) % self.total_filtered_rows
        
        return row


class ProgressiveParquetWriter:
    """Memory-efficient parquet writer that writes rows in batches."""
    
    def __init__(self, output_path: Path, batch_size: int = 1000):
        self.output_path = output_path
        self.batch_size = batch_size
        self.batch_tables = []
        self.total_rows = 0
        self.writer = None
        self.schema = None
    
    def add_row(self, row: pa.Table):
        """Add a row to the current batch."""
        if self.schema is None:
            self.schema = row.schema
        
        self.batch_tables.append(row)
        self.total_rows += 1
        
        # Write batch if it's full
        if len(self.batch_tables) >= self.batch_size:
            self._write_batch()
    
    def _write_batch(self):
        """Write the current batch to the parquet file."""
        if not self.batch_tables:
            return
        
        batch_table = pa.concat_tables(self.batch_tables)
        
        if self.writer is None:
            # Initialize writer on first batch
            self.writer = pq.ParquetWriter(self.output_path, batch_table.schema)
        
        self.writer.write_table(batch_table)
        self.batch_tables.clear()
    
    def finalize(self):
        """Write any remaining rows and close the writer."""
        if self.batch_tables:
            self._write_batch()
        
        if self.writer is not None:
            self.writer.close()
            print(f"Wrote {self.total_rows} rows to {self.output_path}")
        else:
            print(f"WARNING: No data to write for {self.output_path}")


def create_dataset_splits(
    storage_files: List[Path],
    output_dir: Path,
    min_timectl: int,
    min_rows_per_file: int,
    file_counts: dict[Path, int],
    val_percentage: float,
    test_percentage: float,
    batch_size: int = 1000
):
    """
    Create train/val/test dataset splits using memory-efficient round-robin sampling.
    
    Args:
        storage_files: List of storage file paths
        output_dir: Directory to save the new parquet files
        min_timectl: Minimum timeCtl value for filtering
        file_counts: Dictionary of file paths to maximum rows per file (from max analysis)
        val_percentage: Percentage of rows for validation set
        test_percentage: Percentage of rows for test set
        batch_size: Number of rows to batch before writing to disk
    """
    # Calculate total rows and split sizes
    total_rows = sum(file_counts.values())
    val_rows = int(total_rows * val_percentage / 100)
    test_rows = int(total_rows * test_percentage / 100)
    train_rows = total_rows - val_rows - test_rows
    
    print(f"Creating dataset with {total_rows} total rows:")
    print(f"  Train: {train_rows} rows ({100 - val_percentage - test_percentage:.1f}%)")
    print(f"  Val: {val_rows} rows ({val_percentage:.1f}%)")
    print(f"  Test: {test_rows} rows ({test_percentage:.1f}%)")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize memory-efficient iterators for each storage file and handle ValueError
    iterators = []
    for file_path, count in file_counts.items():
        iterators.append(FilteredRowIterator(file_path, min_timectl, count, batch_size))
    
    # Initialize progressive writers for each split
    train_writer = ProgressiveParquetWriter(output_dir / "train.parquet", batch_size)
    val_writer = ProgressiveParquetWriter(output_dir / "val.parquet", batch_size)
    test_writer = ProgressiveParquetWriter(output_dir / "test.parquet", batch_size)
    
    # Create probability thresholds for split assignment
    train_prob = train_rows / total_rows
    val_prob = val_rows / total_rows
    
    rows_added = 0
    train_count = 0
    val_count = 0
    test_count = 0
    print("Starting memory-efficient round-robin sampling...")
    
    try:
        # Round-robin sampling
        while rows_added < total_rows:
            for iterator in iterators:
                if rows_added >= total_rows:
                    break
                    
                # Get next row from this storage file
                row = iterator.get_next_row()
                if row is None:
                    continue
                
                # Randomly assign to train/val/test based on remaining quotas
                remaining_train = train_rows - train_count
                remaining_val = val_rows - val_count
                remaining_test = test_rows - test_count
                
                # Adjust probabilities based on remaining quotas
                if remaining_train <= 0:
                    # No more train slots, split between val and test
                    if remaining_val <= 0:
                        split = "test"
                    elif remaining_test <= 0:
                        split = "val"
                    else:
                        split = "val" if random.random() < remaining_val / (remaining_val + remaining_test) else "test"
                elif remaining_val <= 0:
                    # No more val slots, split between train and test
                    if remaining_test <= 0:
                        split = "train"
                    else:
                        split = "train" if random.random() < remaining_train / (remaining_train + remaining_test) else "test"
                elif remaining_test <= 0:
                    # No more test slots, split between train and val
                    split = "train" if random.random() < remaining_train / (remaining_train + remaining_val) else "val"
                else:
                    # All splits have remaining slots, use original probabilities
                    rand_val = random.random()
                    if rand_val < train_prob:
                        split = "train"
                    elif rand_val < train_prob + val_prob:
                        split = "val"
                    else:
                        split = "test"
                
                # Add row to appropriate writer
                if split == "train":
                    train_writer.add_row(row)
                    train_count += 1
                elif split == "val":
                    val_writer.add_row(row)
                    val_count += 1
                else:  # test
                    test_writer.add_row(row)
                    test_count += 1
                
                rows_added += 1
                
                if rows_added % 10000 == 0:
                    print(f"Processed {rows_added}/{total_rows} ({rows_added/total_rows*100:.2f}%) rows", end="\r")
        
        print(f"Final counts - Train: {train_count}, Val: {val_count}, Test: {test_count}")
        
    finally:
        # Ensure all writers are properly finalized
        train_writer.finalize()
        val_writer.finalize()
        test_writer.finalize()


def main():
    """Main function to parse arguments and run the dataset creation."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create train/val/test splits from chess parquet storage files"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Top-level directory containing storage files"
    )
    parser.add_argument(
        "output_dir", 
        type=str,
        help="Output directory for train/val/test parquet files"
    )
    parser.add_argument(
        "--min-timectl",
        type=int,
        required=True,
        help="Minimum timeCtl value for filtering rows"
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        required=True,
        help="Minimum number of rows per file"
    )
    parser.add_argument(
        '--max-repeats',
        type=int,
        required=True,
        help="Maximum number of times a row can be repeated"
    )
    parser.add_argument(
        "--valp",
        type=float,
        required=True,
        help="Percentage of rows for validation set (0-100)"
    )
    parser.add_argument(
        "--testp",
        type=float,
        required=True,
        help="Percentage of rows for test set (0-100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for memory-efficient processing (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.valp < 0 or args.valp > 100:
        raise ValueError("valp must be between 0 and 100")
    if args.testp < 0 or args.testp > 100:
        raise ValueError("testp must be between 0 and 100")
    if args.valp + args.testp >= 100:
        raise ValueError("valp + testp must be less than 100")
    
    # Set random seed
    random.seed(args.seed)
    
    # Convert paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Min timeCtl: {args.min_timectl}")
    print(f"Val percentage: {args.valp}%")
    print(f"Test percentage: {args.testp}%")
    print(f"Random seed: {args.seed}")
    
    try:
        # Discover storage files
        storage_files = discover_storage_files(input_dir)
        
        # Find maximum filtered rows across all files
        max_rows, file_counts = find_max_filtered_rows(storage_files, args.min_timectl, args.min_rows, args.max_repeats)
        storage_files = list(file_counts.keys())
        
        if max_rows == 0:
            print("No rows meet the filter criteria in any file")
            return 1
        
        # Create the dataset splits
        create_dataset_splits(
            storage_files,
            output_dir,
            args.min_timectl,
            args.min_rows,
            file_counts,
            args.valp,
            args.testp,
            args.batch_size
        )
        
        print("Dataset creation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
