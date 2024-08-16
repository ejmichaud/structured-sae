""" 
Trains a suite of TopK SAEs with activations from an HDF5 file.

Modified from a script by Anish Mudide.
"""

import argparse
import wandb

import torch

from structured_sae.utils import HDF5ActivationBuffer, cycle
from structured_sae.dictionary_learning.training import trainSAE
from structured_sae.dictionary_learning.trainers.top_k import AutoEncoderTopK
from structured_sae.dictionary_learning.trainers.top_k_unconstrained import TrainerTopKUnconstrained
from structured_sae.dictionary_learning.evaluation import evaluate

if __name__ == '__main__':

    hdf5_file = "/om2/user/ericjm/structured-sae/activations/gpt2-layer7.h5"
    steps = 32_000
    batch_size = 8_192
    buffer_size = 262_144
    seed = 0
    use_wandb = False
    log_steps = 5
    save_dir = "/om2/user/ericjm/structured-sae/experiments/topk_baselines1/dictionaries/"

    # we'll define a grid search over ks and n_latents
    ks = [32]
    n_latentss = [4_096, 16_384, 32_768, 65_536]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    buffer = HDF5ActivationBuffer(hdf5_file, 
            batch_size=batch_size, 
            buffer_size=buffer_size, 
            device=device
    )

    configs = [
        { 
            'trainer' : TrainerTopKUnconstrained,
            'dict_class' : AutoEncoderTopK,
            'activation_dim' : buffer.activation_dim,
            'dict_size' : n_latents,
            'k' : k,
            'auxk_alpha' : 0.0, # NO AUXILIARY LOSS
            'decay_start' : int(steps * 0.8),
            'steps' : steps,
            'seed' : seed,
            'device' : device,
            # 'wandb_name' : 'train_topk_v0',
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
        save_dir=save_dir, 
    )


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--hdf5_file", type=str, default="activations/gpt2-layer7.h5")
#     parser.add_argument("--n_latents", type=int, default=16_384)
#     parser.add_argument("--k", type=int, default=32)
#     parser.add_argument("--steps", type=int, default=100_000)
#     parser.add_argument("--batch_size", type=int, default=8_192)
#     parser.add_argument("--buffer_size", type=int, default=262_144)
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--use_wandb", action="store_true")
#     parser.add_argument("--log_steps", type=int, default=10)
#     parser.add_argument("--save_dir", type=str, default="dictionaries")
#     args = parser.parse_args()

#     main(args)



# device = f'cuda:{args.gpu}'
# model = LanguageModel(lm, dispatch=True, device_map=device)
# submodule = model.transformer.h[layer]
# data = hf_dataset_to_generator(hf)
# buffer = ActivationBuffer(data, model, submodule, d_submodule=activation_dim, n_ctxs=n_ctxs, device=device)

# base_trainer_config = {
#     'trainer' : TrainerTopK,
#     'dict_class' : AutoEncoderTopK,
#     'activation_dim' : activation_dim,
#     'dict_size' : args.dict_ratio * activation_dim,
#     'auxk_alpha' : 1/32,
#     'decay_start' : int(steps * 0.8),
#     'steps' : steps,
#     'seed' : 0,
#     'device' : device,
#     'layer' : layer,
#     'lm_name' : lm,
#     'wandb_name' : 'AutoEncoderTopK'
# }

# trainer_configs = [(base_trainer_config | {'k': k}) for k in args.ks]

# wandb.init(entity="amudide", project="TopK (Frequent Log)", config={f'{trainer_config["wandb_name"]}-{i}' : trainer_config for i, trainer_config in enumerate(trainer_configs)})

# trainSAE(buffer, trainer_configs=trainer_configs, save_dir='dictionaries', log_steps=1, steps=steps)

# print("Training finished. Evaluating SAE...", flush=True)
# for i, trainer_config in enumerate(trainer_configs):
#     ae = AutoEncoderTopK.from_pretrained(f'dictionaries/{cfg_filename(trainer_config)}/ae.pt', k = trainer_config['k'], device=device)
#     metrics = evaluate(ae, buffer, device=device)
#     log = {}
#     log.update({f'{trainer_config["wandb_name"]}-{i}/{k}' : v for k, v in metrics.items()})
#     wandb.log(log, step=steps+1)
# wandb.finish()
