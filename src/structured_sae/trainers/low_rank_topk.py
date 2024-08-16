
import torch as t
import torch.nn as nn

from ..dictionary import StructuredAutoEncoderTopK
from ..ops import low_rank_mvm

class LowRankAutoEncoderTopK(StructuredAutoEncoderTopK):
    """
    Structured autoencoder with low-rank encoder and decoder.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int, rank: int):
        super().__init__(activation_dim, dict_size, k)
        self.enc_W1 = nn.Parameter(t.empty(dict_size, rank))
        self.enc_W2 = nn.Parameter(t.empty(activation_dim, rank))
        self.dec_W1 = nn.Parameter(t.empty(activation_dim, rank))
        self.dec_W2 = nn.Parameter(t.empty(dict_size, rank))

        # in dense TopK, encoder W_e is the matrix in
        # nn.Linear(activation_dim, dict_size)
        # which has weights initialized in U(-1/sqrt(activation_dim), 1/sqrt(activation_dim))
        # which has variance (2/sqrt(activation_dim))^2 / 12 = 1 / (3 * activation_dim)
        # we want our encoder to have the same variance
        enc_var = 1 / (3 * activation_dim)
        es = (enc_var / rank) ** 0.25
        self.enc_W1.data.normal_(0, es)
        self.enc_W2.data.normal_(0, es)

        # in dense TopK, encoder W_d is initialized
        # self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        # which is an (activation_dim, dict_size) matrix. its columns are
        # then normalized to have unit norm. the columns have activation_dim
        # elements in them. What variance produces columns with an 
        # expected norm of 1? If we have variance dv, and
        # assume mean zero, then E[||x||^2] = E[sum_i x_i^2] = sum_i E[x_i^2]
        # = sum_i (dv) = activation_dim * (dv) = 1. Thus, dv = 1 / activation_dim
        # now, each element of the decoder matrix is a sum of rank elements
        # each of which is a product of two independent random variables
        # with variance (ds)^2. thus dec_var = rank * (ds)^2 * (ds)^2, and
        # ds = (dec_var / rank) ^ 0.25
        dec_var = 1 / activation_dim
        ds = (dec_var / rank) ** 0.25 
        self.dec_W1.data.normal_(0, ds)
        self.dec_W2.data.normal_(0, ds)

    def encoder_mvm(self, x: t.Tensor) -> t.Tensor:
        return low_rank_mvm(self.enc_W1, self.enc_W2, x)
    
    def decoder_mvm(self, x: t.Tensor) -> t.Tensor:
        return low_rank_mvm(self.dec_W1, self.dec_W2, x)
    
    def encoder_feature(self, i: int) -> t.Tensor:
        return self.enc_W2 @ self.enc_W1[i, :]
        # (activation_dim, rank) @ (dict_size, rank)[i, :]
 
    def decoder_feature(self, i: int) -> t.Tensor:
        return self.dec_W1 @ self.dec_W2[i, :] 
        # (activation_dim, rank) @ (dict_size, rank)[i, :]
    
    def from_pretrained(path, k: int, device=None):
        state_dict = t.load(path)
        rank = state_dict['enc_W1'].shape[1]
        dict_size = len(state_dict['encoder_bias'])
        activation_dim = len(state_dict['b_dec'])
        autoencoder = LowRankAutoEncoderTopK(activation_dim, dict_size, k, rank)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder
