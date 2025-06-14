import torch
import ast
from torch.utils.data import Dataset
from torch_geometric.data import Data as Dt

from CatalystGen.common.data_utils import (
    preprocess,
    add_scaled_lattice_prop,
)

# miller 지수가 concat이 되지 않게 하기 위함
class Data(Dt):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "miller_index":
            return None  
        return super().__cat_dim__(key, value, *args, **kwargs)
    def __inc__(self, key, value, *args, **kwargs):
        if key == "miller_index":
            return 0  # 🔥 요게 없으면 concat 시 offset을 잘못 더함
        return super().__inc__(key, value, *args, **kwargs)

class CatalystDataset(Dataset):
    def __init__(self, path, niggli=True, primitive=False,
                 graph_method="crystalnn", lattice_scale_method="scale_length",
                 preprocess_workers=8, **kwargs):
        super().__init__()
        self.path = path
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method
        )

        original_len = len(self.cached_data)
        self.cached_data = [d for d in self.cached_data if d['graph_arrays'][-1] > 0]
        filtered_len = len(self.cached_data)
        print(f"🧹 Filtered out {original_len - filtered_len} graphs with 0 atoms")

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles,
         edge_indices, to_jimages, num_atoms) = data_dict['graph_arrays']

        absorbate_id = data_dict.get('absorbate_id', -1)
        absorbate_id = torch.tensor([absorbate_id], dtype=torch.long)

        miller_index = data_dict.get("miller_index")
        miller_index = torch.as_tensor(miller_index, dtype=torch.long).clone().detach()


        
        return Data(
            frac_coords=torch.tensor(frac_coords, dtype=torch.float),
            atom_types=torch.tensor(atom_types, dtype=torch.long),
            lengths=torch.tensor(lengths).view(1, -1),
            angles=torch.tensor(angles).view(1, -1),
            edge_index=torch.tensor(edge_indices.T, dtype=torch.long).contiguous(),
            to_jimages=torch.tensor(to_jimages, dtype=torch.long),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,

            absorbate_id=absorbate_id,  # 반드시 Tensor
            miller_index = miller_index,
        )

    def __repr__(self):
        return f"CatalystDataset(path={self.path})"
