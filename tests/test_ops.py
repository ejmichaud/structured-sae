import pytest
import torch
import sys
import os

from structured_sae.ops import (
    kronecker_mvm,
    kronecker_mmm,
    sum_kronecker_mvm,
    block_diagonal_mmm,
    block_diagonal_mvm,
)


def test_kronecker_mvm_shape():
    L = torch.randn(3, 4)
    R = torch.randn(2, 5)
    x = torch.randn(10, 20)  # 4 * 5 = 20
    result = kronecker_mvm(L, R, x)
    assert result.shape == (10, 6)  # 3 * 2 = 6

def test_kronecker_mvm0():
    L = torch.eye(2)
    R = torch.eye(3)
    x = torch.randn(5, 6)
    result = kronecker_mvm(L, R, x) # tensor product of identities is identity
    assert result.shape == (5, 6)
    assert torch.allclose(result, x, atol=1e-6)

def test_kronecker_mvm1():
    c = 2
    L = torch.eye(100)
    R = c * torch.eye(100)
    x = torch.randn(5, 100*100)
    result = kronecker_mvm(L, R, x)
    assert result.shape == (5, 100*100)
    assert torch.allclose(result, c*x, atol=1e-6)

def test_kronecker_mvm2():
    """Check that the multiplication works when the input doesn't
    have a batch dimension.
    """
    L = torch.eye(2)
    R = torch.eye(3)
    x = torch.randn(6)
    result = kronecker_mvm(L, R, x)
    assert result.shape == (6,)
    assert torch.allclose(result, x, atol=1e-6)

def test_kronecker_mmm_shape():
    L = torch.randn(3, 4)
    R = torch.randn(2, 5)
    result = kronecker_mmm(L, R)
    assert result.shape == (3 * 2, 4 * 5)

def test_kronecker_mmm0():
    L = torch.eye(2)
    R = torch.eye(3)
    result = kronecker_mmm(L, R) # tensor product of identities is identity
    assert result.shape == (2 * 3, 2 * 3)
    assert torch.allclose(result, torch.eye(6), atol=1e-6)

def test_sum_kronecker_mvm_shape():
    L = torch.randn(2, 3, 4)
    R = torch.randn(2, 2, 5)
    x = torch.randn(10, 20)  # 4 * 5 = 20
    result = sum_kronecker_mvm(L, R, x)
    assert result.shape == (10, 6)  # 3 * 2 = 6

def test_sum_kronecker_mvm0():
    L = torch.randn(2, 3, 4)
    R = torch.randn(2, 2, 5)
    x = torch.randn(10, 20)  # 4 * 5 = 20
    result = sum_kronecker_mvm(L, R, x)
    assert result.shape == (10, 6)  # 3 * 2 = 6
    assert torch.allclose(result, kronecker_mvm(L[0], R[0], x) + kronecker_mvm(L[1], R[1], x), atol=1e-5) # reduced precision on this one needed for some reason

def test_sum_kronecker_mvm1():
    L = torch.randn(2, 3, 4)
    R = torch.randn(2, 2, 5)
    x = torch.randn(10, 10, 20)  # 4 * 5 = 20
    result = sum_kronecker_mvm(L, R, x)
    assert result.shape == (10, 10, 6)  # 3 * 2 = 6
    assert torch.allclose(result, kronecker_mvm(L[0], R[0], x) + kronecker_mvm(L[1], R[1], x), atol=1e-5) # reduced precision on this one needed for some reason

def test_sum_kronecker_mvm2():
    """Test that the function works when the input doesn't have a batch dimension."""
    L = torch.randn(2, 3, 4)
    R = torch.randn(2, 2, 5)
    x = torch.randn(20,)  # 4 * 5 = 20
    result = sum_kronecker_mvm(L, R, x)
    assert result.shape == (6,)  # 3 * 2 = 6
    assert torch.allclose(result, kronecker_mvm(L[0], R[0], x) + kronecker_mvm(L[1], R[1], x), atol=1e-6)

def test_block_diagonal_mmm_shape0():
    blocks = torch.randn(2, 2, 2)
    result = block_diagonal_mmm(blocks)
    assert result.shape == (4, 4)

def test_block_diagonal_mmm_shape1():
    blocks = torch.randn(3, 4, 5)
    result = block_diagonal_mmm(blocks)
    assert result.shape == (12, 15)

def test_block_diagonal_mmm_shape2():
    blocks = torch.randn(10, 1, 10)
    result = block_diagonal_mmm(blocks)
    assert result.shape == (10, 100)

def test_block_diagonal_mmm0():
    blocks = torch.tensor([
        [[1., 0.],
         [0., 1.]],
        [[1., 0.],
         [0., 1.]],
    ])
    result = block_diagonal_mmm(blocks)
    assert result.shape == (4, 4)
    assert torch.allclose(result, torch.eye(4), atol=1e-6)

def test_block_diagonal_mmm1():
    blocks = torch.tensor([
        [[1., 2.],
         [3., 4.]],
        [[5., 6.],
         [7., 8.]],
    ])
    result = block_diagonal_mmm(blocks)
    assert result.shape == (4, 4)
    assert torch.allclose(result, torch.tensor([
        [1., 2., 0., 0.],
        [3., 4., 0., 0.],
        [0., 0., 5., 6.],
        [0., 0., 7., 8.],
    ]), atol=1e-6)

def test_block_diagonal_mvm0():
    blocks = torch.tensor([
        [[1., 0.],
         [0., 1.]],
        [[1., 0.],
         [0., 1.]],
    ])
    for _ in range(10):
        x = torch.randn(2, 4)
        result = block_diagonal_mvm(blocks, x)
        assert result.shape == (2, 4)
        assert torch.allclose(result, x, atol=1e-6)

def test_block_diagonal_mvm1():
    blocks = torch.tensor([
        [[1., 0.],
         [0., 1.]],
        [[1., 0.],
         [0., 1.]],
    ])
    for _ in range(10):
        x = torch.randn(4,)
        result = block_diagonal_mvm(blocks, x)
        assert result.shape == (4,)
        assert torch.allclose(result, x, atol=1e-6)

def test_block_diagonal_mvm2():
    blocks = torch.tensor([
        [[1., 0.],
         [0., 1.]],
        [[1., 0.],
         [0., 1.]],
    ])
    for _ in range(10):
        x = torch.randn(2, 2, 4)
        result = block_diagonal_mvm(blocks, x)
        assert result.shape == (2, 2, 4)
        assert torch.allclose(result, x, atol=1e-6)

def test_block_diagonal_mvm3():
    blocks = torch.tensor([
        [[1., 1.],
         [1., 1.]],
        [[1., 2.],
         [3., 4.]],
    ])
    x = torch.tensor([
        [1., 2., 3., 4.],
        [1., 1., 1., 1.],
    ])
    result = block_diagonal_mvm(blocks, x)
    assert result.shape == (2, 4)
    assert torch.allclose(result, torch.tensor([
        [3., 3., 11., 25.],
        [2., 2., 3., 7.],
    ]), atol=1e-6)

def test_block_diagonal_mvm4():
    blocks = torch.tensor([
        [[0., 1.],
         [1., 0.]],
        [[1., 2.],
         [2., 1.]],
    ])
    x = torch.tensor([
        [1., 2., 3., 4.],
        [1., 1., 1., 1.],
        [0., 1., 0., 1.]
    ])
    result = block_diagonal_mvm(blocks, x)
    assert result.shape == (3, 4)
    assert torch.allclose(result, torch.tensor([
        [2., 1., 11., 10.],
        [1., 1., 3., 3.],
        [1., 0., 2., 1.]
    ]), atol=1e-6)


# def test_block_diagonal_mvm

# def test_block_diagonal_

# def test_kronecker_known_values():
#     L = torch.tensor([[1., 2.], [3., 4.]])
#     R = torch.tensor([[0.5, 1.], [1.5, 2.]])
#     x = torch.tensor([[1., 2., 3., 4.]])
#     expected = torch.tensor([[11., 16., 21., 31.]])
#     result = kronecker(L, R, x)
#     assert torch.allclose(result, expected)

# def test_kronecker_batch():
#     L = torch.randn(3, 4)
#     R = torch.randn(2, 5)
#     x = torch.randn(7, 4, 5)
#     result = kronecker(L, R, x)
#     assert result.shape == (7, 6)

# def test_kronecker_invalid_input():
#     L = torch.randn(3, 4)
#     R = torch.randn(2, 5)
#     x = torch.randn(10, 19)  # Invalid shape
#     with pytest.raises(AssertionError):
#         kronecker(L, R, x)
