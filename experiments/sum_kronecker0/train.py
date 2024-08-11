
import os
import sys
from collections import defaultdict
import pickle

import numpy as np
from tqdm import tqdm

import torch
# import blobfile as bf
# import transformer_lens
# from huggingface_hub import hf_hub_download
# from sae_lens import SparseAutoencoderDictionary
# import sparse_autoencoder

from structured_sae.operations import (
    low_rank_mmm,
    kronecker_mmm,
    sum_kronecker_mmm,
)

# compute prime factorization
def prime_factors(n):
    """Returns list of prime factors of n.
    >>> prime_factors(10)
    [2, 5]
    >>> prime_factors(32)
    [2, 2, 2, 2, 2]
    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
    
def main(idx):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    layer_index = 7
    location = "resid_post_mlp"
    # with bf.BlobFile(sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb") as f:
    #     state_dict = torch.load(f)
    #     autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
    #     autoencoder.to(device)

    # W_e = autoencoder.encoder.weight.detach()
    # save W_e to disk for easy access next time
    # torch.save(W_e, os.path.join(os.path.abspath(os.path.dirname(__file__)), "W_e.pt"))
    W_e = torch.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), "W_e.pt"))

    n, m = W_e.shape
    n_factors = prime_factors(n)
    m_factors = prime_factors(m)
    total = (len(n_factors) - 1) * (len(m_factors) - 1)
    print("total shapes:", total)
    if idx >= total:
        raise ValueError("idx out of bounds")
    n_idx = idx // (len(m_factors) - 1)
    m_idx = idx % (len(m_factors) - 1)
    d1 = np.prod(n_factors[:n_idx+1])
    d3 = np.prod(n_factors[n_idx+1:])
    d2 = np.prod(m_factors[:m_idx+1])
    d4 = np.prod(m_factors[m_idx+1:])
    # import code; code.interact(local=locals())

    assert d1 * d3 == W_e.shape[0]
    assert d2 * d4 == W_e.shape[1]
    print(f"shapes: {d1}x{d2} and {d3}x{d4}")

    single_term = (d1 * d2) + (d3 * d4)
    upper_r = (n * m) // single_term
    # import code; code.interact(local=locals())

    # define the rs
    # round upper_r to the nearest power of 2
    upper_r = 2 ** int(np.ceil(np.log2(upper_r)) + 0.1)
    # let the rs be powers of 2 from 8 up to upper_r * 4
    rs = [2 ** i for i in range(2, int(np.log2(upper_r)) + 4)]
    r3s = [r * 3 for r in rs][:-1]
    rs += r3s
    rs = sorted(rs)
    print("rs:", rs)

    var_W_e = W_e.var().item()

    sum_kronecker_encoder_dynamics = defaultdict(list)
    sum_kronecker_encoder_parameters = dict()
    steps = 10_000

    for r in tqdm(rs):
        W1 = torch.randn(r, d1, d2, device=device, dtype=dtype)
        W2 = torch.randn(r, d3, d4, device=device, dtype=dtype)

        # we want the variance of sum_kronecker_mmm(W1, W2) to be the same as the variance of W_e
        # this matrix will be a sum of r products. what is the variance of each element in 
        # each product? since each element in a Kronecker product is a product of two elements,
        # it will just be the product of the variances of the two elements. Let s
        # be the standard deviation of element in W1 and W2. Then the variance of the product
        # will be s^2 * s^2 = s^4. So the variance of the sum of r products will be r * s^4.
        # so s = (var_W_e / r)^(1/4)
        s = (var_W_e / r) ** 0.25
        W1 *= s
        W2 *= s
        W1.requires_grad = True
        W2.requires_grad = True

        sum_kronecker_encoder_parameters[r] = W1.numel() + W2.numel()

        optimizer = torch.optim.Adam([W1, W2], lr=1e-3)
        for i in range(steps):
            W_e_approx = sum_kronecker_mmm(W1, W2)
            # loss = 0.5 * torch.log(1.0 + (W_e - W_e_approx).square()).mean()
            loss = 0.5 * torch.log1p((W_e - W_e_approx).square()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_kronecker_encoder_dynamics[r].append(loss.item())
    
    results = {
        "losses": sum_kronecker_encoder_dynamics,
        "parameters": sum_kronecker_encoder_parameters,
        "rs": rs,
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "d4": d4,
        "steps": steps,
    }
    
    base_dir = "/om2/user/ericjm/structured-sae/"
    with open(os.path.join(base_dir, f"saved_data/layer{layer_index}_{location}_v5_32k_encoder_sum_kronecker_{d1}x{d2}_{d3}x{d4}_results.pkl"), "wb") as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    idx = int(sys.argv[1])
    main(idx)
