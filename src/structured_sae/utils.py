import os
import json
from datetime import datetime
import h5py
import torch

def cycle(iterable):
    """Cycle through an iterable indefinitely."""
    while True:
        for item in iterable:
            yield item

class HDF5ActivationBuffer:
    """An iterator that yields batches of activations from an HDF5 file.

    Instead of sampling randomly from the buffer, we load up buffers of 
    activations at a time. Within a buffer, we return a batch of activations.
    This batch is a non-contiguous slice of the buffer. For instance, the first
    batch might be activations 0, 32, 64... while the second batch might be
    activations 1, 33, 65... and so on.

    NOTE: on networked storage (om2), this is only about 4x faster than
    computing activations on the fly, via a naive comparison with how
    long it took my `save_activations.py` script to run vs. how
    quickly we can load activations from disk and into GPU memory. It
    is of course possible that the NNSightActivationBuffer, if it 
    uses batching or something, would totally close this gap, but
    I haven't tested it yet.

    NOTE: I have tested it now, and it is about 2-3x faster than computing
    activations on the fly with the ActivationBuffer class. So worth using!
    """
    def __init__(self, hdf5_path, 
            batch_size=8_192,    # 2^13
            buffer_size=262_144, # 2^18
            device='cpu',
            dtype=torch.float32):
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        assert self.batch_size <= self.buffer_size
        self.device = device

        if self.buffer_size % self.batch_size != 0:
            print("Warning: batch size not a multiple of buffer size, so some activations will be skipped.")

        with h5py.File(self.hdf5_path, 'r') as f:
            self.n_activations = f['activations'].shape[0]
            self.buffer = torch.from_numpy(
                f['activations'][:self.buffer_size], 
            ).to(self.device, dtype=dtype)
            self.activation_dim = self.buffer.shape[1]
            # print("Loaded activations from HDF5 file, shape:", self.buffer.shape)
            self.buffer_size = min(self.buffer_size, self.n_activations)
            self.skip = self.buffer_size // self.batch_size
            # print("Buffer size:", self.buffer_size)
            # print("Skip size:", self.skip)
            # print("n_activations:", self.n_activations)
        
        self.buffer_idx = 0
        self.file_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """
        If the buffer is exhausted, load a new buffer.
        If the buffer is exhausted and the file is exhausted, raise StopIteration
        The buffer is exhausted when buffer_idx >= skip.
        Skip is equal to buffer_size // batch_size.
        The buffer size is not necessarily constant, since the last buffer
            may be smaller than buffer_size.
        """
        # print("NEXT HAS BEEN CALLED---------------------") 
        # print("> buffer_idx:", self.buffer_idx)
        # print("> file_idx:", self.file_idx)
        # print("> skip:", self.skip)
        if self.buffer_idx >= self.skip:
            self.file_idx += self.buffer_size
            if self.file_idx >= self.n_activations:
                raise StopIteration
            else:
                with h5py.File(self.hdf5_path, 'r') as f:
                    self.buffer = torch.from_numpy(
                        f['activations'][self.file_idx:self.file_idx+self.buffer_size], 
                    ).to(self.device, dtype=self.buffer.dtype)
                    if len(self.buffer) < self.buffer_size:
                        print("Warning: buffer is smaller than set buffer size, due to end of file.")
                    self.buffer_size = len(self.buffer)
                    self.skip = self.buffer_size // self.batch_size
                self.buffer_idx = 0
        # print("After possible buffer reload:")  
        # print("> buffer_idx:", self.buffer_idx)
        # print("> file_idx:", self.file_idx)
        # print("> skip:", self.skip)

        batch_idxs = torch.arange(self.buffer_idx, self.buffer_size, self.skip)[:self.batch_size]
        batch = self.buffer[batch_idxs]

        self.buffer_idx += 1
        # print("Returning batch: ", batch)        
        return batch


class LocalLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"logs.jsonl")

    def log(self, data, step=None):
        with open(self.log_file, "a") as f:
            log_entry = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
            }
            for k, v in data.items():
                log_entry[k] = v
            json.dump(log_entry, f)
            f.write("\n")  # Add a newline for JSONL format
