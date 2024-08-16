
import torch as t
import torch.nn as nn

from .dictionary_learning.dictionary import Dictionary

class StructuredAutoEncoderTopK(Dictionary, nn.Module):
    """
    Base class for structured variants of the top-k autoencoder.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k
        self.encoder_bias = nn.Parameter(t.zeros(dict_size))
        self.b_dec = nn.Parameter(t.zeros(activation_dim))
    
    def encoder_mvm(self, x: t.Tensor) -> t.Tensor:
        raise NotImplementedError
    
    def decoder_mvm(self, x: t.Tensor) -> t.Tensor:
        raise NotImplementedError
    
    def encoder_feature(self, i: int) -> t.Tensor:
        raise NotImplementedError

    def decoder_feature(self, i: int) -> t.Tensor:
        raise NotImplementedError

    def encode(self, x: t.Tensor, return_topk: bool = False):
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder_mvm(x - self.b_dec) + self.encoder_bias)
        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = t.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return self.decoder_mvm(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def from_pretrained(path, k: int, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        raise NotImplementedError
        # state_dict = t.load(path)
        # dict_size = len(state_dict['encoder_bias'])
        # activation_dim = len(state_dict['b_dec'])
        # autoencoder = AutoEncoderTopK(activation_dim, dict_size, k)
        # autoencoder.load_state_dict(state_dict)
        # if device is not None:
        #     autoencoder.to(device)
        # return autoencoder

