
import pytest
import torch
import sys
import os

from structured_sae.trainers.low_rank_topk import LowRankAutoEncoderTopK
from structured_sae.trainers.sum_kronecker_topk import SumKroneckerAutoEncoderTopK
from structured_sae.trainers.block_diagonal_topk import BlockDiagonalAutoEncoderTopK

from structured_sae.ops import (
    low_rank_mvm,
    low_rank_mmm,
    kronecker_mvm,
    kronecker_mmm,
    sum_kronecker_mvm,
    sum_kronecker_mmm,
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

def test_SumKroneckerAutoEncoderTopK_features0():
    ae = SumKroneckerAutoEncoderTopK(
        activation_dim=8,
        dict_size=30,
        k=5,
        r=3,
        d1=10,
        d2=4,
        d3=3,
        d4=2,
        prepost=True,
    )
    W_e_dense = sum_kronecker_mmm(ae.enc_L.data, ae.enc_R.data) @ ae.enc_V
    W_d_dense = ae.dec_V @ sum_kronecker_mmm(ae.dec_L.data, ae.dec_R.data)
    for i in range(30):
        assert torch.allclose(ae.encoder_feature(i), W_e_dense[i, :], atol=1e-5)
    for i in range(30):
        print(i)
        assert torch.allclose(ae.decoder_feature(i), W_d_dense[:, i], atol=1e-5)

def testSumKroneckerAutoEncoderTopK_features1():
    ae = SumKroneckerAutoEncoderTopK(
        activation_dim=8,
        dict_size=30,
        k=5,
        r=3,
        d1=10,
        d2=4,
        d3=3,
        d4=2,
        prepost=True,
    )
    ae.enc_V.data.normal_(0, 1) # init to non-identity for this test
    ae.dec_V.data.normal_(0, 1) # init to non-identity for this test
    W_e_dense = sum_kronecker_mmm(ae.enc_L.data, ae.enc_R.data) @ ae.enc_V
    W_d_dense = ae.dec_V @ sum_kronecker_mmm(ae.dec_L.data, ae.dec_R.data)
    for i in range(30):
        assert torch.allclose(ae.encoder_feature(i), W_e_dense[i, :], atol=1e-5)
    for i in range(30):
        assert torch.allclose(ae.decoder_feature(i), W_d_dense[:, i], atol=1e-5)

def test_SumKroneckerAutoEncoderTopK_features2():
    ae = SumKroneckerAutoEncoderTopK(
        activation_dim=16,
        dict_size=64,
        k=5,
        r=1,
        d1=8,
        d2=8,
        d3=8,
        d4=2,
        prepost=True,
    )
    ae.enc_V.data.normal_(0, 1) # init to non-identity for this test
    ae.dec_V.data.normal_(0, 1) # init to non-identity for this test
    W_e_dense = sum_kronecker_mmm(ae.enc_L.data, ae.enc_R.data) @ ae.enc_V
    W_d_dense = ae.dec_V @ sum_kronecker_mmm(ae.dec_L.data, ae.dec_R.data)
    for i in range(64):
        assert torch.allclose(ae.encoder_feature(i), W_e_dense[i, :], atol=1e-5)
    for i in range(64):
        assert torch.allclose(ae.decoder_feature(i), W_d_dense[:, i], atol=1e-5)

def test_SumKroneckerAutoEncoderTopK_features3():
    ae = SumKroneckerAutoEncoderTopK(
        activation_dim=16,
        dict_size=64,
        k=5,
        r=1,
        d1=8,
        d2=8,
        d3=8,
        d4=2,
        prepost=False,
    )
    W_e_dense = sum_kronecker_mmm(ae.enc_L.data, ae.enc_R.data)
    W_d_dense = sum_kronecker_mmm(ae.dec_L.data, ae.dec_R.data)
    for i in range(64):
        assert torch.allclose(ae.encoder_feature(i), W_e_dense[i, :], atol=1e-5)
    for i in range(64):
        assert torch.allclose(ae.decoder_feature(i), W_d_dense[:, i], atol=1e-5)

def test_SumKroneckerAutoEncoderTopK_decode0():
    ae = SumKroneckerAutoEncoderTopK(
        activation_dim=16,
        dict_size=64,
        k=5,
        r=1,
        d1=8,
        d2=8,
        d3=8,
        d4=2,
        prepost=True,
    )
    f = torch.randn(64).relu()
    decoded_acts = ae.decode(f)
    W_d = ae.dec_V @ sum_kronecker_mmm(ae.dec_L.data, ae.dec_R.data)
    decoded_acts_manual = W_d @ f + ae.b_dec
    assert torch.allclose(decoded_acts, decoded_acts_manual, atol=1e-5)

def test_BlockDiagonalAutoEncoderTopK_features0():
    ae = BlockDiagonalAutoEncoderTopK(
        activation_dim=16,
        dict_size=64,
        blocks=2,
        proj_dim=32,
        k=5,
    )
    W_e_dense = block_diagonal_mmm(ae.enc_B) @ ae.enc_V
    W_d_dense = ae.dec_V @ block_diagonal_mmm(ae.dec_B)
    for i in range(64):
        assert torch.allclose(ae.encoder_feature(i), W_e_dense[i, :], atol=1e-5)
    for i in range(64):
        assert torch.allclose(ae.decoder_feature(i), W_d_dense[:, i], atol=1e-5)

def test_BlockDiagonalAutoEncoderTopK_features1():
    ae = BlockDiagonalAutoEncoderTopK(
        activation_dim=32,
        dict_size=256,
        blocks=8,
        proj_dim=64,
        k=5,
    )
    W_e_dense = block_diagonal_mmm(ae.enc_B) @ ae.enc_V
    W_d_dense = ae.dec_V @ block_diagonal_mmm(ae.dec_B)
    for i in range(256):
        assert torch.allclose(ae.encoder_feature(i), W_e_dense[i, :], atol=1e-5)
    for i in range(256):
        assert torch.allclose(ae.decoder_feature(i), W_d_dense[:, i], atol=1e-5)
