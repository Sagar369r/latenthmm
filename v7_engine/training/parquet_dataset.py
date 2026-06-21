import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq
import numpy as np
from v7_engine.config import EMBEDDING_DIM

class ParquetStreamingDataset(IterableDataset):
    """
    Lazy PyTorch Dataloader that streams sequences directly from Parquet files
    using PyArrow. Keeps RAM usage permanently low while providing continuous
    zero-copy I/O streaming.
    """
    def __init__(self, parquet_path: str, batch_size: int = 32):
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        
        # Fast metadata scan for length
        self.total_rows = pq.read_metadata(parquet_path).num_rows
        
    def __iter__(self):
        # Open file exactly ONCE and stream batches directly into RAM
        parquet_file = pq.ParquetFile(self.parquet_path)
        feature_cols = [f"f_{i}" for i in range(EMBEDDING_DIM)]
        
        for batch in parquet_file.iter_batches(batch_size=self.batch_size):
            # Convert PyArrow RecordBatch directly to dictionary/numpy
            df = batch.to_pydict()
            
            # Stack feature columns into a (batch_size, EMBEDDING_DIM) matrix
            seqs = np.column_stack([df[col] for col in feature_cols])
            vars_ = np.array(df["var"])
            labels = np.array(df["barrier_label"])
            
            yield (
                torch.tensor(seqs, dtype=torch.float32),
                torch.tensor(vars_, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32)
            )
