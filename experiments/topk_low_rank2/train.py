""" 
Trains a suite of TopK SAEs with activations from an HDF5 file.

Modified from a script by Anish Mudide.
"""

import argparse
import wandb

import torch

from structured_sae.utils import HDF5ActivationBuffer, cycle
from structured_sae.dictionary_learning.training import trainSAE
from structured_sae.trainers.low_rank_topk import LowRankAutoEncoderTopK, TrainerLowRankTopK

if __name__ == '__main__':

    hdf5_file = "/om2/user/ericjm/structured-sae/activations/gpt2-layer7.h5"
    steps = 64_000
    batch_size = 8_192
    buffer_size = 262_144
    seed = 0
    use_wandb = False
    log_steps = 5
    save_dir = "/om2/user/ericjm/structured-sae/experiments/topk_low_rank2/dictionaries/"

    # we'll define a grid search over ks and n_latents
    k = 32
    n_latents = 16_384
    ranks = [48, 96, 192, 384, 768]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    buffer = HDF5ActivationBuffer(hdf5_file, 
            batch_size=batch_size, 
            buffer_size=buffer_size, 
            device=device
    )

    configs = [
        { 
            'trainer' : TrainerLowRankTopK,
            'dict_class' : LowRankAutoEncoderTopK,
            'activation_dim' : buffer.activation_dim,
            'dict_size' : n_latents,
            'k' : k,
            'rank': rank,
            'auxk_alpha' : 0.0, # NO AUXILIARY LOSS
            'decay_start' : int(steps * 0.8),
            'steps' : steps,
            'seed' : seed,
            'device' : device,
            # 'wandb_name' : 'train_topk_v0',
            'lm_name' :  "lm_nameX",
            'layer' :  "layerX",
        }
        for rank in ranks
    ]
    
    trainSAE(cycle(buffer), 
        trainer_configs=configs, 
        use_wandb=use_wandb,
        # wandb_entity="ericjmichaud_",
        # wandb_project="structured-saes",
        steps=steps,
        log_steps=log_steps, 
        save_steps=8_000,
        save_dir=save_dir, 
    )

