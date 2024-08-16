
import pytest
import torch
import sys
import os

from structured_sae.trainers.low_rank_topk import LowRankAutoEncoderTopK
from structured_sae.ops import (
    low_rank_mvm,
    low_rank_mmm,
    kronecker_mvm,
    kronecker_mmm,
    sum_kronecker_mvm,
    block_diagonal_mmm,
    block_diagonal_mvm,
)


def test_LowRankAutoEncoderTopK_features0():
    ae = LowRankAutoEncoderTopK(
        activation_dim=10,
        dict_size=30,
        k=5,
        rank=3,
    )
    W_e_dense = low_rank_mmm(ae.enc_W1.data, ae.enc_W2.data)
    W_d_dense = low_rank_mmm(ae.dec_W1.data, ae.dec_W2.data)
    for i in range(30):
        assert torch.allclose(ae.encoder_feature(i), W_e_dense[i, :])
    for i in range(30):
        assert torch.allclose(ae.decoder_feature(i), W_d_dense[:, i])
