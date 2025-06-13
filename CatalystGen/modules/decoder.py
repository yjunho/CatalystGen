import torch
import torch.nn as nn
import torch.nn.functional as F

from CatalystGen.modules.embeddings import MAX_ATOMIC_NUM
from CatalystGen.modules.gemnet.gemnet import GemNetT


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(fc_num_layers - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class GemNetTDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        max_neighbors=15.,
        radius=10.0,
        scale_file=None,
    ):
        super().__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors



        self.gemnet = GemNetT(
            num_targets=1,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_forces=True,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=True,
            scale_file=scale_file,
        )

        self.fc_atom = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)
        self.classifier_h = nn.Linear(hidden_dim, 5)
        self.classifier_k = nn.Linear(hidden_dim, 5)
        self.classifier_l = nn.Linear(hidden_dim, 5)

    

    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms,
                lengths, angles):

        output = self.gemnet(
            z=z,
            frac_coords=pred_frac_coords,
            atom_types=pred_atom_types,
            num_atoms=num_atoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
        )
        if output is None:
            return None, None, None
        
        h, pred_cart_coord_diff, graph_repr = output

        pred_atom_types = self.fc_atom(h)
        logits_h = self.classifier_h(graph_repr)  # [B, 5]
        logits_k = self.classifier_k(graph_repr)  # [B, 5]
        logits_l = self.classifier_l(graph_repr)  # [B, 5]

        # # 🔸 incorporate pred_miller_index (with noise) into prediction
        # miller_class = pred_miller_index + 2  # [B, 3]
        # miller_onehot = F.one_hot(miller_class, num_classes=5).float()  # [B, 3, 5]

        # graph_repr_expand = graph_repr.unsqueeze(1).expand(-1, 3, -1)  # [B, 3, latent_dim]
        # decoder_input = torch.cat([graph_repr_expand, miller_onehot], dim=-1)  # [B, 3, latent_dim + 5]

        # decoder_input_flat = decoder_input.view(-1, decoder_input.shape[-1])  # [B*3, latent+5]
        # miller_logits_flat = self.fc_miller(decoder_input_flat)  # [B*3, 5]
        # pred_miller = miller_logits_flat.view(-1, 3, 5)  # [B, 3, 5]
        
        return pred_cart_coord_diff, pred_atom_types, (logits_h, logits_k, logits_l)
