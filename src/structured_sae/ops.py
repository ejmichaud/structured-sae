"""
Defines various operations, including matrix-vector multiplies (MVM),
for different structured matrices.
"""

import torch

def low_rank_mvm(U: torch.Tensor, V: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix-vector product for a low-rank matrix UV^T.
    Args:
        U: Left matrix of shape (m, r)
        V: Right matrix of shape (n, r)
        x: Input vector of shape (..., n)
    Returns:
        Result vector of shape (..., m)
    """
    t = torch.einsum('ir,...i->...r', V, x)
    y = torch.einsum('or,...r->...o', U, t)
    return y

def low_rank_mmm(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix-matrix product for a low-rank matrix UV^T.
    Args:
        U: Left matrix of shape (m, r)
        V: Right matrix of shape (n, r)
    Returns:
        Result matrix of shape (m, n)
    """
    return torch.mm(U, V.t())

def kronecker_mvm(L: torch.Tensor, R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix-vector product (L ⊗ R)x.

    Args:
        L: Left matrix of shape (d1, d2)
        R: Right matrix of shape (d3, d4)
        x: Input tensor of shape (..., d2*d4)

    Returns:
        Result tensor of shape (..., d1*d3)
    """
    d1, d2 = L.shape
    d3, d4 = R.shape

    xm = x.reshape(*x.shape[:-1], d2, d4)
    ym = torch.einsum('ij,kl,...jl->...ik', L, R, xm)
    y = ym.reshape(*x.shape[:-1], d1 * d3)
    return y

def kronecker_mmm(L: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix-matrix product (L ⊗ R).

    Args:
        L: Left matrix of shape (d1, d2)
        R: Right matrix of shape (d3, d4)

    Returns:
        Result tensor of shape (d1*d3, d2*d4)
    
    TODO: should this just be torch.kron instead?
    """
    d1, d2 = L.shape
    d3, d4 = R.shape

    return torch.einsum('ij,kl->ikjl', L, R).reshape(d1 * d3, d2 * d4)

def sum_kronecker_mvm(L: torch.Tensor, R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the sum of matrix-vector products Σ_s (L[s] ⊗ R[s])x.
    
    Args:
        L: Left tensor of shape (s, d1, d2)
        R: Right tensor of shape (s, d3, d4)
        x: Input tensor of shape (..., d2*d4)
    
    Returns:
        Result tensor of shape (..., d1*d3)
    """
    s, d1, d2 = L.shape
    _, d3, d4 = R.shape
    
    xm = x.reshape(*x.shape[:-1], d2, d4)
    ym = torch.einsum('sij,skl,...jl->...ik', L, R, xm)
    y = ym.reshape(*x.shape[:-1], d1 * d3)
    return y

def sum_kronecker_mmm(L: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Compute the sum of matrix-matrix products Σ_s (L[s] ⊗ R[s]).
    
    Args:
        L: Left tensor of shape (s, d1, d2)
        R: Right tensor of shape (s, d3, d4)

    Returns:
        Result tensor of shape (d1*d3, d2*d4)
    """
    s, d1, d2 = L.shape
    _, d3, d4 = R.shape
    
    return torch.einsum('sij,skl->ikjl', L, R).reshape(d1 * d3, d2 * d4)

def block_diagonal_mmm(blocks: torch.Tensor) -> torch.Tensor:
    """
    Expands `blocks` into a dense matrix.
    """
    return torch.block_diag(*blocks)

def block_diagonal_mvm(blocks: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix-vector product for a block-diagonal matrix.
    
    Args:
        blocks: Tensor of blocks of shape (b, n, m)
        x: Input vector of shape (..., b*m)
    
    Returns:
        Result vector of shape (..., b*n)
    """
    b, n, m = blocks.shape
    lds, bm = x.shape[:-1], x.shape[-1]
    assert bm % m == 0
    x = x.reshape(*lds, b, m)
    ym = torch.einsum('bnm,...bm->...bn', blocks, x)
    return ym.reshape(*lds, b*n)

# def sum_kronecker_rightmult_mmm(L: torch.Tensor, R: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
#     """
#     Compute the sum of matrix-matrix products Σ_s (L[s] ⊗ R[s])U.
    
#     Args:
#         L: Left tensor of shape (s, d1, d2)
#         R: Right tensor of shape (s, d3, d4)
#         U: Input tensor of shape (..., d2*d4)
    
#     Returns:
#         Result tensor of shape (..., d1*d3)
#     """
#     return sum_kronecker_mmm(L, R) @ U

# def sum_kronecker_rightmult_mvm
    

# def btt_2core_mvm(L: torch.Tensor, R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#     """
#     Compute the matrix-vector product for a BTT matrix Wx.
    
#     Args:
#         L: Left matrix of shape (d1, d2)
#         R: Right matrix of shape (d3, d4)
#         x: Input vector of shape (..., d2*d4)
    
#     Returns:
#         Result vector of shape (..., d1*d3)
#     """
#     d1, d2 = L.shape
#     d3, d4 = R.shape

#     xm = x.reshape(*x.shape[:-1], d2, d4)
#     ym = torch.einsum('ij,kl,...jl->...ik', L, R, xm)
#     y = ym.reshape(*x.shape[:-1], d1 * d3)
#     return y



# def monarch_mvm(L: torch.Tensor, R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#     """
#     Compute the matrix-vector product for a Monarch matrix Wx.
    
#     Args:
#         L: Left matrix of shape (b, d1, d2)
#         R: Right matrix of shape (b, d2, d3)
#         x: Input vector of shape (..., d2*d3)
    
#     Returns:
#         Result vector of shape (..., d1*d2)
#     """
#     b, d1, d2 = L.shape
#     _, d2, d3 = R.shape
#     xr = x.reshape(*x.shape[:-1], d2, d3)
    


# def monarch_mvm(L: torch.Tensor, R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#     """
#     Compute the Monarch matrix mvm Wx.

#     Args:
#         L: Left matrix of shape (d1, d2)
#         R: Right matrix of shape (d3, d4)
#         x: Input tensor of shape (..., d2*d4)

#     Returns:
#         Result tensor of shape (..., d1*d3)
#     """
#     d1, d2 = L.shape
#     d3, d4 = R.shape

#     xm = x.reshape(*x.shape[:-1], d2, d4)
#     ym = torch.einsum('ij,kl,...jl->...ik', L, R, xm)
#     y = ym.reshape(*x.shape[:-1], d1 * d3)
#     return y


