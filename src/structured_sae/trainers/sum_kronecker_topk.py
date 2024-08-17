
from collections import namedtuple

import einops
import torch as t
import torch.nn as nn

from ..dictionary import StructuredAutoEncoderTopK
from ..ops import sum_kronecker_mvm
from ..dictionary_learning.trainers.trainer import SAETrainer
from ..dictionary_learning.trainers.top_k import geometric_median

class SumKroneckerAutoEncoderTopK(StructuredAutoEncoderTopK):
    """
    Structured autoencoder with low-rank encoder and decoder.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int, 
            r: int, d1: int, d2: int, d3: int, d4: int, prepost=True):
        super().__init__(activation_dim, dict_size, k)
        assert d1 * d3 == dict_size 
        assert d2 * d4 == activation_dim
        self.r = r
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.prepost = prepost
        
        self.enc_L = nn.Parameter(t.empty(r, d1, d2))
        self.enc_R = nn.Parameter(t.empty(r, d3, d4))
        if prepost:
            self.enc_V = nn.Parameter(t.eye(activation_dim))
        self.dec_L = nn.Parameter(t.empty(r, d2, d1))
        self.dec_R = nn.Parameter(t.empty(r, d4, d3))
        if prepost:
            self.dec_V = nn.Parameter(t.eye(activation_dim))

        # first initialize as standard normal
        self.enc_L.data.normal_(0., 1.)
        self.enc_R.data.normal_(0., 1.)
        self.dec_L.data = self.enc_R.data.clone().transpose(1, 2) # tie weights at initialization
        self.dec_R.data = self.enc_L.data.clone().transpose(1, 2) # tie weights at initialization

        # set encoder scale to match dense encoder variance
        enc_var = 1 / (3 * activation_dim) # desired variance of dense encoder
        es = (enc_var / r) ** 0.25         # std of of enc_L and enc_R
        self.enc_L.data *= es
        self.enc_R.data *= es

        # set decoder scale to match dense decoder variance
        dec_var = 1 / activation_dim    # desired variance of dense decoder
        ds = (dec_var / r) ** 0.25      # std of dec_L
        self.dec_L.data *= ds
        self.dec_R.data *= ds

    def encoder_mvm(self, x: t.Tensor) -> t.Tensor:
        if self.prepost:
            x = self.enc_V @ x
        return sum_kronecker_mvm(self.enc_L, self.enc_R, x)
    
    def decoder_mvm(self, x: t.Tensor) -> t.Tensor:
        x = sum_kronecker_mvm(self.dec_L, self.dec_R, x)
        if self.prepost:
            x = self.dec_V @ x
        return x
    
    def encoder_feature(self, i: int) -> t.Tensor:
        raise NotImplementedError
        # li = i // self.d1 
        # ri = i % self.d1
        # f = t.einsum('si,sj->ij', self.enc_L[:, li, :], self.enc_R[:, ri, :]).reshape(self.r, self.d2 * self.d4)
        # if self.prepost:
        #     f = self.enc_V @ f
        # return f
 
    def decoder_feature(self, i: int) -> t.Tensor:
        raise NotImplementedError
        # li = i // self.d1
        # ri = i % self.d1
        # # f = torch.kron(self.dec_L[:, li], self.dec_R[:, ri])
        # f = t.einsum('si,sj->ij', self.dec_L[:, :, li], self.dec_R[:, :, ri]).reshape(self.r, self.d2 * self.d4)
        # if self.prepost:
        #     f = f @ self.dec_V
        # return f
    
    def from_pretrained(path, k: int, device=None):
        state_dict = t.load(path)
        r = state_dict['enc_L'].shape[0]
        d1 = state_dict['enc_L'].shape[1]
        d2 = state_dict['enc_L'].shape[2]
        d3 = state_dict['enc_R'].shape[1]
        d4 = state_dict['enc_R'].shape[2]
        dict_size = d1 * d3
        activation_dim = d2 * d4
        prepost = bool('enc_V' in state_dict)
        autoencoder = SumKroneckerAutoEncoderTopK(activation_dim, dict_size, k, r, d1, d2, d3, d4, prepost)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class TrainerSumKroneckerTopK(SAETrainer):
    """
    Top-K SAE training scheme.
    """

    def __init__(
        self,
        dict_class=SumKroneckerAutoEncoderTopK,
        activation_dim=512,
        dict_size=64 * 512,
        k=100,
        r=32, d1=256, d2=32, d3=128, d4=16,
        prepost=True, 
        auxk_alpha=0.0, # NO AUXK
        decay_start=24000,  # when does the lr decay start
        steps=30000,  # when when does training end
        lr=None,
        seed=None,
        device=None,
        layer=None,
        lm_name=None,
        wandb_name="SumKroneckerAutoEncoderTopK",
        submodule_name=None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        self.wandb_name = wandb_name
        self.steps = steps
        self.k = k
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # Initialise autoencoder
        self.ae = dict_class(activation_dim, dict_size, k=k, r=r, d1=d1, d2=d2, d3=d3, d4=d4, prepost=prepost)
        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        # If not specified, auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
        scale = dict_size / (2**14)
        if lr is not None:
            self.lr = lr
        else:
            self.lr = 2e-4 / scale**0.5
        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000

        # Optimizer and scheduler
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))

        def lr_fn(step):
            if step < decay_start:
                return 1.0
            else:
                return (steps - step) / (steps - decay_start)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # Training parameters
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)

        # Log the effective L0, i.e. number of features actually used, which should a constant value (K)
        # Note: The standard L0 is essentially a measure of dead features for Top-K SAEs)
        self.logging_parameters = ["effective_l0", "dead_features"]
        self.effective_l0 = -1
        self.dead_features = -1

    def loss(self, x, step=None, logging=False):
        # Run the SAE
        f, top_acts, top_indices = self.ae.encode(x, return_topk=True)
        x_hat = self.ae.decode(f)

        # Measure goodness of reconstruction
        e = x_hat - x
        total_variance = (x - x.mean(0)).pow(2).sum(0)

        # Update the effective L0 (again, should just be K)
        self.effective_l0 = top_acts.size(1)

        # Update "number of tokens since fired" for each features
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        # Compute dead feature mask based on "number of tokens since fired"
        dead_mask = (
            self.num_tokens_since_fired > self.dead_feature_threshold
            # if self.auxk_alpha > 0
            # else None
        ).to(f.device)
        self.dead_features = int(dead_mask.sum())

        # If dead features: Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = x.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = t.where(dead_mask[None], f, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(f)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.ae.decode(auxk_acts_BF)
            auxk_loss = (e_hat - e).pow(2)  # .sum(0)
            auxk_loss = scale * t.mean(auxk_loss / total_variance)
        else:
            auxk_loss = x_hat.new_tensor(0.0)

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = auxk_loss.sum(dim=-1).mean()
        loss = l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {"l2_loss": l2_loss.item(), "auxk_loss": auxk_loss.item(), "loss": loss.item()},
            )

    def update(self, step, x):
        # Initialise the decoder bias
        if step == 0:
            median = geometric_median(x)
            self.ae.b_dec.data = median

        # Make sure the decoder is still unit-norm
        # self.ae.set_decoder_norm_to_unit_norm()

        # compute the loss
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # clip grad norm and remove grads parallel to decoder directions
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        # self.ae.remove_gradient_parallel_to_decoder_directions()

        # do a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        # return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "TrainerLowRankTopK",
            "dict_class": "LowRankAutoEncoderTopK",
            "lr": self.lr,
            "steps": self.steps,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k,
            "rank": self.ae.rank,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }

