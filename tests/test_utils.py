import pytest
import h5py
import numpy as np
import torch
import sys
import os

from structured_sae.utils import HDF5ActivationBuffer, cycle

def test_HDF5ActivationBuffer0():
    hdf5_path = 'test.h5'
    activations = np.array([
        [0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.],
    ])
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('activations', data=activations)
    
    buffer = HDF5ActivationBuffer(hdf5_path, batch_size=2, buffer_size=4, 
        device='cpu', dtype=torch.float32)

    assert torch.allclose(next(buffer), torch.tensor([[0.], [2.]]))
    assert torch.allclose(next(buffer), torch.tensor([[1.], [3.]]))
    assert torch.allclose(next(buffer), torch.tensor([[4.], [6.]]))
    assert torch.allclose(next(buffer), torch.tensor([[5.], [7.]]))
    with pytest.raises(StopIteration):
        next(buffer)
    
    os.remove(hdf5_path)

def test_HDF5ActivationBuffer1():
    hdf5_path = 'test.h5'
    activations = np.array([
        [0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.],
    ])
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('activations', data=activations)
    
    buffer = HDF5ActivationBuffer(hdf5_path, batch_size=2, buffer_size=5,
        device='cpu', dtype=torch.float32)
    
    assert torch.allclose(next(buffer), torch.tensor([[0.], [2.]]))
    assert torch.allclose(next(buffer), torch.tensor([[1.], [3.]]))
    assert torch.allclose(next(buffer), torch.tensor([[5.], [6.]]))

    with pytest.raises(StopIteration):
        next(buffer)
    
    os.remove(hdf5_path)

def test_HDF5ActivationBuffer2():
    hdf5_path = 'test.h5'
    activations = np.arange(1000).reshape(-1, 1)
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('activations', data=activations)
    
    buffer = HDF5ActivationBuffer(hdf5_path, batch_size=256, buffer_size=512,
        device='cpu', dtype=torch.float32)

    assert len(next(buffer)) == 256
    assert len(next(buffer)) == 256
    assert len(next(buffer)) == 256

    with pytest.raises(StopIteration):
        next(buffer)
    
    os.remove(hdf5_path)

def test_HDF5ActivationBuffer3():
    hdf5_path = 'test.h5'
    activations = np.arange(1000).reshape(-1, 1)
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('activations', data=activations)
    
    buffer = HDF5ActivationBuffer(hdf5_path, batch_size=999, buffer_size=1000,
        device='cpu', dtype=torch.float32)

    assert len(next(buffer)) == 999

    with pytest.raises(StopIteration):
        next(buffer)
    
    os.remove(hdf5_path)

def test_HDF5ActivationBuffer4():
    hdf5_path = 'test.h5'
    activations = np.arange(1000).reshape(-1, 1)
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('activations', data=activations)
    
    buffer = HDF5ActivationBuffer(hdf5_path, batch_size=1000, buffer_size=1000,
        device='cpu', dtype=torch.float32)

    assert len(next(buffer)) == 1000

    with pytest.raises(StopIteration):
        next(buffer)
    
    os.remove(hdf5_path)

def test_HDF5ActivationBuffer5():
    """Check that `cycle` works correctly
    with it now."""
    hdf5_path = 'test.h5'
    activations = np.array([
        [0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.],
    ])
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('activations', data=activations)
    
    buffer = HDF5ActivationBuffer(hdf5_path, batch_size=2, buffer_size=4, 
        device='cpu', dtype=torch.float32)
    buffer = cycle(buffer)

    assert torch.allclose(next(buffer), torch.tensor([[0.], [2.]]))
    assert torch.allclose(next(buffer), torch.tensor([[1.], [3.]]))
    assert torch.allclose(next(buffer), torch.tensor([[4.], [6.]]))
    assert torch.allclose(next(buffer), torch.tensor([[5.], [7.]]))
    assert torch.allclose(next(buffer), torch.tensor([[0.], [2.]]))
    assert torch.allclose(next(buffer), torch.tensor([[1.], [3.]]))
    assert torch.allclose(next(buffer), torch.tensor([[4.], [6.]]))
    assert torch.allclose(next(buffer), torch.tensor([[5.], [7.]]))
    
    os.remove(hdf5_path)

def test_HDF5ActivationBuffer6():
    hdf5_path = 'test.h5'
    activations = np.array([
        [0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.],
    ])
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('activations', data=activations)
    
    buffer = HDF5ActivationBuffer(hdf5_path, batch_size=2, buffer_size=5,
        device='cpu', dtype=torch.float32)
    buffer = cycle(buffer)
    
    assert torch.allclose(next(buffer), torch.tensor([[0.], [2.]]))
    assert torch.allclose(next(buffer), torch.tensor([[1.], [3.]]))
    assert torch.allclose(next(buffer), torch.tensor([[5.], [6.]]))
    assert torch.allclose(next(buffer), torch.tensor([[0.], [2.]]))
    
    os.remove(hdf5_path)
