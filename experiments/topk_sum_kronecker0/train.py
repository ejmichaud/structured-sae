""" 
Trains a suite of TopK SAEs with activations from an HDF5 file.

Modified from a script by Anish Mudide.
"""

import sys
import wandb

import torch

from structured_sae.utils import HDF5ActivationBuffer, cycle
from structured_sae.dictionary_learning.training import trainSAE
from structured_sae.trainers.sum_kronecker_topk import SumKroneckerAutoEncoderTopK, TrainerSumKroneckerTopK

dss = [
    ((128, 32), (128, 24)), # done
    ((512, 32), (32, 24)),  # done
    ((32, 32), (512, 24)),  # done
    ((2_048, 32), (8, 24)),  
    ((8, 32), (2_048, 24)),
    ((128, 128), (128, 6)), # done
    ((128, 6), (128, 128)), # done
    ((512, 128), (32, 6)),
    ((32, 6), (512, 128)),
]

W_e_params = 16_384 * 768 # = 12_582_912

# for (d1, d2), (d3, d4) in dss:
#     k_params = d1 * d2 + d3 * d4
#     print(W_e_params / k_params)

rss = [
    [128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384],    # 7168
    [256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768],   # 17152
    [128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384],    # 13312
    [512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536],    # 65728
    [512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536],    # 49408
    [256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768],    # 17152
    [256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768],    # 17152
    [512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536],    # 65728
    [512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536],    # 65728
]

assert len(dss) == len(rss)

if __name__ == '__main__':
    exit()
    # get argv
    i = int(sys.argv[1])
    ds = dss[i]
    rs = rss[i]
    (d1, d2), (d3, d4) = ds

    hdf5_file = "/om2/user/ericjm/structured-sae/activations/gpt2-layer7.h5"
    steps = 64_000
    batch_size = 8_192
    buffer_size = 262_144
    seed = 0
    use_wandb = False
    log_steps = 5
    save_dir = f"/om2/user/ericjm/structured-sae/experiments/topk_sum_kronecker0/{d1}x{d2}-{d3}x{d4}/"

    k = 32
    n_latents = 16_384

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    buffer = HDF5ActivationBuffer(hdf5_file, 
            batch_size=batch_size, 
            buffer_size=buffer_size, 
            device=device
    )

    configs = [
        { 
            'trainer' : TrainerSumKroneckerTopK,
            'dict_class' : SumKroneckerAutoEncoderTopK,
            'activation_dim' : buffer.activation_dim,
            'dict_size' : n_latents,
            'k' : k,
            'r' : r,
            'd1' : d1,
            'd2' : d2,
            'd3' : d3,
            'd4' : d4,
            'auxk_alpha' : 0.0, # NO AUXILIARY LOSS
            'decay_start' : int(steps * 0.8),
            'steps' : steps,
            'seed' : seed,
            'device' : device,
            # 'wandb_name' : 'train_topk_v0',
            'lm_name' :  "lm_nameX",
            'layer' :  "layerX",
        }
        for r in rs
    ]
    
    trainSAE(cycle(buffer), 
        trainer_configs=configs, 
        use_wandb=use_wandb,
        # wandb_entity="ericjmichaud_",
        # wandb_project="structured-saes",
        steps=steps,
        log_steps=log_steps, 
        # save_steps=8_000, # no checkpoints, since there'll be a lot of data
        save_dir=save_dir, 
    )

