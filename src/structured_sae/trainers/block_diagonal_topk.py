
from collections import namedtuple

import einops
import torch as t
import torch.nn as nn

from ..dictionary import StructuredAutoEncoderTopK
from ..ops import block_diagonal_mvm
from ..dictionary_learning.trainers.trainer import SAETrainer
from ..dictionary_learning.trainers.top_k import geometric_median

class BlockDiagonalAutoEncoderTopK(StructuredAutoEncoderTopK):
    """
    Structured autoencoder with block diagonal encoder and decoder
    matrices. Includes leading (trailing) dense matrix multiplication
    into `proj_dim` dimensions in the encoder (decoder).

    TODO: think about where to include biases.
        we just have to consider whether to put a bias in between
        the first up-projection and the block diagonal matrix.
        If we do this, then the transformation becomes:
        Wx = B(Vx + b) = BVx + Bb
        So ultimately we just get a new bias term Bb, which can
        be absorbed into the encoder bias.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int, 
            proj_dim: int, blocks: int):
        super().__init__(activation_dim, dict_size, k)
        self.proj_dim = proj_dim
        self.blocks = blocks
        assert dict_size % blocks == 0
        assert proj_dim % blocks == 0

        self.enc_V = nn.Parameter(t.empty(proj_dim, activation_dim))
        self.enc_B = nn.Parameter(t.empty(blocks, dict_size // blocks, proj_dim // blocks))
        self.dec_V = nn.Parameter(t.empty(activation_dim, proj_dim))
        self.dec_B = nn.Parameter(t.empty(blocks, proj_dim // blocks, dict_size // blocks))

        # first initialize as standard normal
        self.enc_V.data.normal_(0., 1.)
        self.enc_B.data.normal_(0., 1.)
        self.dec_V.data = self.enc_V.data.clone().T                 # tie weights at init
        self.dec_B.data = self.enc_B.data.clone().transpose(1, 2)   # tie weights at init

        # elements of the encoder matrix are sums of proj_dim // blocks elements
        # let vv be the variance of enc_V and vb be the variance of enc_B
        # then the variance of the encoder matrix is blocks * vv * vb
        # if we let vv = vb = v, then the variance of the encoder matrix is 
        # blocks * v^2. If we want the var_enc = blocks * v^2, then
        # v = sqrt(var_enc / blocks), so the standard deviation
        # of enc_V and enc_B should be (var_enc / blocks) ** 0.25

        # set encoder scale to match dense encoder variance
        enc_var = 1 / (3 * activation_dim) # desired variance of dense encoder
        es = (enc_var / (self.proj_dim // blocks)) ** 0.25
        self.enc_V.data *= es
        self.enc_B.data *= es

        # set decoder scale to match dense decoder variance
        dec_var = 1 / activation_dim    # desired variance of dense decoder
        ds = (dec_var / (self.proj_dim // blocks)) ** 0.25      # std of dec_L
        self.dec_V.data *= ds
        self.dec_B.data *= ds

    def encoder_mvm(self, x: t.Tensor) -> t.Tensor:
        x = t.matmul(x, self.enc_V.t())
        return block_diagonal_mvm(self.enc_B, x)

    def decoder_mvm(self, x: t.Tensor) -> t.Tensor:
        x = block_diagonal_mvm(self.dec_B, x)
        x = t.matmul(x, self.dec_V.t())
        return x
    
    def encoder_feature(self, i: int) -> t.Tensor:
        """ 
        if W = L @ R, then W_ij = sum_k L_ik R_kj
        So W_i* = sum_k L_ik R_k* = L_i @ R
        if L has shape (a, b)
        if R has shape (b, c)
        then W_i* has shape (b,) (b, c) = (c,), as desired
        """
        assert i < self.dict_size
        bi = i // (self.dict_size // self.blocks)
        ri = i % (self.dict_size // self.blocks)
        br = self.enc_B[bi, ri, :] # has length proj_dim // blocks
        v_start = bi * (self.proj_dim // self.blocks)
        v_end = v_start + self.proj_dim // self.blocks
        return br @ self.enc_V[v_start:v_end, :]
 
    def decoder_feature(self, i: int) -> t.Tensor:
        """ 
        W = V B
        has shape (activation_dim, dict_size)
        we want the ith column, which has length activation_dim
        W_*j = sum_k V_*k B_kj
        """
        assert i < self.dict_size
        # get the right column of dec_B
        bi = i // (self.dict_size // self.blocks)
        ri = i % (self.dict_size // self.blocks)
        bc = self.dec_B[bi, :, ri] # has length proj_dim // blocks
        v_start = bi * (self.proj_dim // self.blocks)
        v_end = v_start + self.proj_dim // self.blocks
        return self.dec_V[:, v_start:v_end] @ bc
        # li = i // self.d3
        # ri = i % self.d3
        # f = t.einsum('si,sj->ij', self.dec_L[:, :, li], self.dec_R[:, :, ri]).reshape(self.d2 * self.d4)
        # if self.prepost:
        #     f = self.dec_V @ f
        # return f
    
    def from_pretrained(path, k: int, device=None):
        raise NotImplementedError
        # state_dict = t.load(path)
        # r = state_dict['enc_L'].shape[0]
        # d1 = state_dict['enc_L'].shape[1]
        # d2 = state_dict['enc_L'].shape[2]
        # d3 = state_dict['enc_R'].shape[1]
        # d4 = state_dict['enc_R'].shape[2]
        # dict_size = d1 * d3
        # activation_dim = d2 * d4
        # prepost = bool('enc_V' in state_dict)
        # autoencoder = BlockDiagonalAutoEncoderTopK(activation_dim, dict_size, k, r, d1, d2, d3, d4, prepost)
        # autoencoder.load_state_dict(state_dict)
        # if device is not None:
        #     autoencoder.to(device)
        # return autoencoder


class TrainerBlockDiagonalTopK(SAETrainer):
    """
    Top-K SAE training scheme.
    """

    def __init__(
        self,
        dict_class=BlockDiagonalAutoEncoderTopK,
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
        wandb_name="BlockDiagonalAutoEncoderTopK",
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
            "trainer_class": "TrainerBlockDiagonalTopK",
            "dict_class": "BlockDiagonalAutoEncoderTopK",
            "lr": self.lr,
            "steps": self.steps,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k,
            "blocks": self.ae.blocks,
            "proj_dim": self.ae.proj_dim,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }
