from pathlib import Path
from typing import Any

import hydra
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from CatalystGen.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc, sanitize_batch_dtype)
from CatalystGen.modules.embeddings import MAX_ATOMIC_NUM
from CatalystGen.modules.embeddings import KHOT_EMBEDDINGS


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


class CatalystGen(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = hydra.utils.instantiate(self.hparams.encoder)
        self.decoder = hydra.utils.instantiate(self.hparams.decoder)

        self.ads_embed = nn.Embedding(self.hparams.num_adsorbates, self.hparams.ads_embedding_dim)
        self.z_project = nn.Linear(self.hparams.latent_dim + self.hparams.ads_embedding_dim,
                           self.hparams.latent_dim)

        self.fc_mu = nn.Linear(self.hparams.latent_dim,
                               self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim,
                                self.hparams.latent_dim)

        self.fc_num_atoms = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                      self.hparams.fc_num_layers, self.hparams.max_atoms+1)
        self.fc_lattice = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                    self.hparams.fc_num_layers, 6)
        self.fc_composition = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                        self.hparams.fc_num_layers, MAX_ATOMIC_NUM)
        
        self.layernorm = nn.LayerNorm(self.hparams.latent_dim)
        self.ads_layernorm = nn.LayerNorm(self.hparams.ads_embedding_dim)
        
        # # ✅ Miller index classifiers (each predicts -2, -1, 0,1,2)

        self.classifier_h = nn.Linear(self.hparams.latent_dim, 5)
        self.classifier_k = nn.Linear(self.hparams.latent_dim, 5)
        self.classifier_l = nn.Linear(self.hparams.latent_dim, 5)

        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_begin),
            np.log(self.hparams.sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.type_sigma_begin),
            np.log(self.hparams.type_sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        # Miller sigmas 노이즈 정의
        miller_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.miller_sigma_begin),
            np.log(self.hparams.miller_sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.miller_sigmas = nn.Parameter(miller_sigmas, requires_grad=False)


        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        # obtain from datamodule.
        self.lattice_scaler = None
        self.scaler = None

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):

        batch = sanitize_batch_dtype(batch)
        """
        encode crystal structures to latents.
        """
        hidden = self.encoder(batch)
        if hidden is None:
            return None
        mu = self.fc_mu(hidden)
        log_var = torch.clamp(self.fc_var(hidden), min=-5, max=5)  # ⛑ 안정화 추가
        log_var = self.fc_var(hidden)

        z = self.reparameterize(mu, log_var)
        z = self.layernorm(z)
        ads_embed = self.ads_embed(batch.absorbate_id.squeeze(-1))  # [B, 256]
        ads_embed = self.ads_layernorm(ads_embed)                   # normalize

        z = torch.cat([z, ads_embed], dim=-1)       # [B, 512]
        z = self.z_project(z)        

        # 🧼 안정성 체크
        if (
            torch.isnan(mu).any() or torch.isinf(mu).any() or
            torch.isnan(log_var).any() or torch.isinf(log_var).any() or
            torch.isnan(z).any() or torch.isinf(z).any()
        ):
            print("❌ NaN or Inf in mu/logvar/z – skipping batch")
            return None

        # z = self.layernorm(z)

        return mu, log_var, z

    def decode_stats(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                     teacher_forcing=False):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))
            composition_per_atom = self.predict_composition(z, num_atoms)


        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None, allowed_atom_types=None, absorbate_id=None):
        """
        decode crystral structure from latent embeddings.
        ld_kwargs: args for doing annealed langevin dynamics sampling:
            n_step_each:  number of steps for each sigma level.
            step_lr:      step size param.
            min_sigma:    minimum sigma to use in annealed langevin dynamics.
            save_traj:    if <True>, save the entire LD trajectory.
            disable_bar:  disable the progress bar of langevin dynamics.
        gt_num_atoms: if not <None>, use the ground truth number of atoms.
        gt_atom_types: if not <None>, use the ground truth atom types.
        """
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []
        # Step 1: absorbate_id (정수)를 one-hot → 임베딩
        assert absorbate_id is not None, "absorbate_id must be provided for generation."

        # --- 추가: absorbate 정보 임베딩 ---
        ads_embed = self.ads_embed(absorbate_id)          # [B, 256]
        ads_embed = self.ads_layernorm(ads_embed)
        z = torch.cat([z, ads_embed], dim=-1)             # [B, 512]
        z = self.z_project(z)                             # [B, 256] → same as training

        # obtain key stats.
        num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(
            z, gt_num_atoms)
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # obtain atom types.
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(
                composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # init coords.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # annealed langevin dynamics.
        for sigma in tqdm(self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(
                    cur_frac_coords) * torch.sqrt(step_size * 2)
                decode_out = self.decoder(
                    z, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles)
            
                if decode_out is None:
                    # 🔴 그래프 형성 실패 → LD 중단
                    return None

                pred_cart_coord_diff, pred_atom_types,pred_miller = decode_out
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms)
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms)
                
                logits_h, logits_k, logits_l = pred_miller

                # 샘플링 방식: argmax (원하면 stochastic으로 교체 가능)
                pred_h = logits_h.argmax(dim=-1) - 2
                pred_k = logits_k.argmax(dim=-1) - 2
                pred_l = logits_l.argmax(dim=-1) - 2
                cur_miller_index = torch.stack([pred_h, pred_k, pred_l], dim=1)

                # Atom type update if not fixed
                if gt_atom_types is None and allowed_atom_types is not None:
                    num_type_classes = pred_atom_types.shape[1]
                    allowed_indices = [i - 1 for i in allowed_atom_types if 0 <= i - 1 < num_type_classes]

                    mask = torch.full_like(pred_atom_types, float('-inf'))
                    mask[:, allowed_indices] = 0.0
                    masked_logits = pred_atom_types + mask
                    probs = F.softmax(masked_logits, dim=-1)
                    probs[torch.isnan(probs)] = 0.0  # Prevent NaN
                    cur_atom_types = torch.multinomial(probs, num_samples=1).squeeze(-1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types, 'miller_index': cur_miller_index,
                       'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict

    def sample(self, num_samples, ld_kwargs):
        z = torch.randn(num_samples, self.hparams.hidden_dim,
                        device=self.device)
        samples = self.langevin_dynamics(z, ld_kwargs)
        if samples is None:
        # 🔴 샘플 생성 실패 시 처리
            print("⚠️ Sampling skipped due to invalid graph.")
            return {}
        return samples

    def forward(self, batch, teacher_forcing, training):

        batch = sanitize_batch_dtype(batch)
        # hacky way to resolve the NaN issue. Will need more careful debugging later.
        encode_out = self.encode(batch)
        if encode_out is None:
            return None
        mu, log_var, z = encode_out

        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
         pred_composition_per_atom) = self.decode_stats(
            z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)

        # sample noise levels.
        noise_level = torch.randint(0, self.sigmas.size(0),
                                    (batch.num_atoms.size(0),),
                                    device=self.device)
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms, dim=0)

        type_noise_level = torch.randint(0, self.type_sigmas.size(0),
                                         (batch.num_atoms.size(0),),
                                         device=self.device)
        used_type_sigmas_per_atom = (
            self.type_sigmas[type_noise_level].repeat_interleave(
                batch.num_atoms, dim=0))
        



        # add noise to atom types and sample atom types.
        pred_composition_per_atom = torch.clamp(pred_composition_per_atom, -10, 10) #softmax 전에 입력값의 수치를 안정화
        pred_composition_probs = F.softmax(pred_composition_per_atom.detach(), dim=-1)
        pred_composition_probs = torch.nan_to_num(pred_composition_probs, nan=0.0, posinf=1.0, neginf=0.0) #클램핑으로 NaN/inf 방지
        atom_type_probs = (
            F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM) +
            pred_composition_probs * used_type_sigmas_per_atom[:, None])
        
        atom_type_probs = atom_type_probs + 1e-8 # 확률이 모두 0인 경우 대비: epsilon 더해주기

        
        atom_type_probs = atom_type_probs / atom_type_probs.sum(dim=-1, keepdim=True) # 확률 정규화

        
        atom_type_probs = torch.nan_to_num(atom_type_probs, nan=0.0, posinf=1.0, neginf=0.0) # 다시 NaN/inf 체크

        rand_atom_types = torch.multinomial(
            atom_type_probs, num_samples=1).squeeze(1) + 1

        # add noise to the cart coords
        cart_noises_per_atom = (
            torch.randn_like(batch.frac_coords) *
            used_sigmas_per_atom[:, None])
        cart_coords = frac_to_cart_coords(
            batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(
            cart_coords, pred_lengths, pred_angles, batch.num_atoms)

        
        decode_out = self.decoder(z, noisy_frac_coords, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles)
        if not decode_out or any(x is None for x in decode_out):
            return None

        pred_cart_coord_diff, pred_atom_types, pred_miller = decode_out
        # compute loss.
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)
        type_loss = self.type_loss(pred_atom_types, batch.atom_types,
                                   used_type_sigmas_per_atom, batch)

        kld_loss = self.kld_loss(mu, log_var)

        # Miller loss
        miller_loss = self.miller_loss(pred_miller, batch.miller_index)

        if self.hparams.predict_property:
            property_loss = self.property_loss(z, batch)
        else:
            property_loss = 0.

        return {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            'miller_loss': miller_loss,
            'property_loss': property_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_composition_per_atom': pred_composition_per_atom,
            'pred_miller' : pred_miller,
            # 'pred_miller_h': logits_h,
            # 'pred_miller_k': logits_k,
            # 'pred_miller_l': logits_l,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'z': z,
        }

    def generate_rand_init(self, pred_composition_per_atom, pred_lengths,
                           pred_angles, num_atoms, batch):
        rand_frac_coords = torch.rand(num_atoms.sum(), 3,
                                      device=num_atoms.device)
        pred_composition_per_atom = F.softmax(pred_composition_per_atom,
                                              dim=-1)
        rand_atom_types = self.sample_composition(
            pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

    def sample_composition(self, composition_prob, num_atoms):
        """
        Samples composition such that it exactly satisfies composition_prob
        """
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(
            composition_prob, index=batch, dim=0, reduce='mean')

        all_sampled_comp = []

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)

            # if the rounded composition gives less atoms, sample the rest
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)

                left_comp_prob = comp_prob - comp_num.float() / num_atom

                left_comp_prob[left_comp_prob < 0.] = 0.
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True)
                # convert to atomic number
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        self.scaler.match_device(z)
        return self.scaler.inverse_transform(self.fc_property(z))

    def predict_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom
    
    
    def predict_miller_index(self, z):
        return self.classifier_h(z), self.classifier_k(z), self.classifier_l(z)

    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)
    

    def property_loss(self, z, batch):
        return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.lattice_scale_method == 'scale_length':
            target_lengths = batch.lengths / \
                batch.num_atoms.view(-1, 1).float()**(1/3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch.batch, reduce='mean').mean()

    def coord_loss(self, pred_cart_coord_diff, noisy_frac_coords,
                   used_sigmas_per_atom, batch):
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles,
            batch.num_atoms, self.device, return_vector=True)

        target_cart_coord_diff = target_cart_coord_diff / \
            used_sigmas_per_atom[:, None]**2
        pred_cart_coord_diff = pred_cart_coord_diff / \
            used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(self, pred_atom_types, target_atom_types,
                  used_type_sigmas_per_atom, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none')
        # rescale loss according to noise
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()
    
    def miller_loss(self, logits_tuple, target_miller_index):
        logits_h, logits_k, logits_l = logits_tuple
        h_target = target_miller_index[:, 0] + 2
        k_target = target_miller_index[:, 1] + 2
        l_target = target_miller_index[:, 2] + 2

        loss_h = F.cross_entropy(logits_h, h_target)
        loss_k = F.cross_entropy(logits_k, k_target)
        loss_l = F.cross_entropy(logits_l, l_target)

        return loss_h + loss_k + loss_l

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        return kld_loss


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, teacher_forcing, training=True)
        if outputs is None:
            self.log("skipped_batch", 1, on_step=True, on_epoch=True, prog_bar=True)
            return None  # Skip this batch
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        if outputs is None:
            return None  # Skip this batch
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        if outputs is None:
            return None  # Skip this batch
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, batch, outputs, prefix):
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        property_loss = outputs['property_loss']
        miller_loss = outputs['miller_loss']

        loss = (
            self.hparams.cost_natom * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_type * type_loss +
            self.hparams.beta * kld_loss +
            self.hparams.cost_composition * composition_loss +
            self.hparams.cost_property * property_loss +
            self.hparams.cost_miller * miller_loss
            )

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_composition_loss': composition_loss,
            f'{prefix}_miller_loss': miller_loss
        }

        if prefix != 'train':
            # validation/test loss only has coord and type
            loss = (
                self.hparams.cost_coord * coord_loss +
                self.hparams.cost_type * type_loss +
                self.hparams.cost_miller * miller_loss)

            # evaluate num_atom prediction.
            pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            num_atom_accuracy = (
                pred_num_atoms == batch.num_atoms).sum() / batch.num_graphs

            # evalute lattice prediction.
            pred_lengths_and_angles = outputs['pred_lengths_and_angles']
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles)
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            # evaluate atom type prediction.
            pred_atom_types = outputs['pred_atom_types']
            target_atom_types = outputs['target_atom_types']
            type_accuracy = pred_atom_types.argmax(
                dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            # evaluate Miller index accuracy
            logits_h, logits_k, logits_l = outputs['pred_miller']

            # 각 축별 클래스 예측 (0~4 → Miller index -2 ~ 2)
            pred_h = logits_h.argmax(dim=-1) - 2  # [B]
            pred_k = logits_k.argmax(dim=-1) - 2  # [B]
            pred_l = logits_l.argmax(dim=-1) - 2  # [B]

            pred_miller_index = torch.stack([pred_h, pred_k, pred_l], dim=1)  # [B, 3]  
            target = batch.miller_index                                # [B, 3]


            miller_acc_per_axis = (pred_miller_index == target).float()     # [B, 3]
            miller_acc_total = (miller_acc_per_axis.sum(dim=1) == 3).float()  # [B], all 3 match

            acc_h = miller_acc_per_axis[:, 0].mean()
            acc_k = miller_acc_per_axis[:, 1].mean()
            acc_l = miller_acc_per_axis[:, 2].mean()
            acc_total = miller_acc_total.mean()

            log_dict.update({
                
                f'{prefix}_loss': loss,
                f'{prefix}_property_loss': property_loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,
                f'{prefix}_acc_h': acc_h,
                f'{prefix}_acc_k': acc_k,
                f'{prefix}_acc_l': acc_l,
                f'{prefix}_acc_total': acc_total
            })
            

        return log_dict, loss

# @hydra.main(config_path=Path("/home/kaist/projects/CatalystGen-main/config"), config_name="default") 
# def main(cfg: omegaconf.DictConfig):
#     model: pl.LightningModule = hydra.utils.instantiate(
#         cfg.model,
#         optim=cfg.optim,
 
#         logging=cfg.logging,
#         _recursive_=False,
#     )
#     return model