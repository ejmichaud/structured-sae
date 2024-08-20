""" 
Trains a suite of TopK SAEs with activations from an HDF5 file.

Modified from a script by Anish Mudide.
"""

import argparse
import wandb

import torch

from structured_sae.utils import HDF5ActivationBuffer, cycle
from structured_sae.dictionary_learning.training import trainSAE
from structured_sae.dictionary_learning.trainers.top_k import AutoEncoderTopK, TrainerTopK
from structured_sae.dictionary_learning.evaluation import evaluate

if __name__ == '__main__':

    hdf5_file = "/om2/user/ericjm/structured-sae/activations/gpt2-layer7.h5"
    steps = 64_000 # increased this
    batch_size = 8_192
    buffer_size = 262_144
    seed = 0
    use_wandb = False
    log_steps = 5
    save_dir = "/om2/user/ericjm/structured-sae/experiments/topk_baselines4/dictionaries/"

    # we'll define a grid search over ks and n_latents
    ks = [32]
    n_latentss = [4_096, 8_192, 16_384, 32_768, 65_536]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    buffer = HDF5ActivationBuffer(hdf5_file, 
            batch_size=batch_size, 
            buffer_size=buffer_size, 
            device=device
    )

    configs = [
        { 
            'trainer' : TrainerTopK,
            'dict_class' : AutoEncoderTopK,
            'activation_dim' : buffer.activation_dim,
            'dict_size' : n_latents,
            'k' : k,
            'auxk_alpha' : 1/32,
            'decay_start' : int(steps * 0.8),
            'steps' : steps,
            'seed' : seed,
            'device' : device,
            'wandb_name' : 'train_topk_v0',
            'lm_name' :  "lm_nameX",
            'layer' :  "layerX",
        }
        for k in ks
        for n_latents in n_latentss
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

