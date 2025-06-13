import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from ase import Atoms
from ase.io import write
from ase.visualize.plot import plot_atoms
from ase.geometry import cellpar_to_cell
import matplotlib.pyplot as plt

# 🔧 gen.pt 경로 직접 지정
pt_path = "/home/kaist/projects/CatalystGen-main/generated2/COCH2O/gen_v1.pt"

def visualize_from_gen_pt(pt_path):
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"{pt_path} not found")

    # ✅ pt 파일이 있는 폴더 기준으로 하위 폴더 만들기
    pt_dir = os.path.dirname(pt_path)
    base_name = os.path.splitext(os.path.basename(pt_path))[0]
    cif_dir = os.path.join(pt_dir, "cif")
    png_dir = os.path.join(pt_dir, "png")
    os.makedirs(cif_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # 📦 데이터 로드
    data = torch.load(pt_path)
    frac_coords = data["frac_coords"]
    atom_types = data["atom_types"]
    lengths = data["lengths"]
    angles = data["angles"]
    num_atoms_list = data["num_atoms"]

    assert len(num_atoms_list) == len(lengths), "Mismatch between num_atoms and lengths"

    offset = 0
    total_structures = len(num_atoms_list)

    for i in range(total_structures):
        n_atoms = num_atoms_list[i].item()
        coords = frac_coords[offset:offset+n_atoms].numpy()
        types = atom_types[offset:offset+n_atoms].numpy()
        a, b, c = lengths[i].numpy()
        alpha, beta, gamma = angles[i].numpy()
        offset += n_atoms

        cell_matrix = cellpar_to_cell([a, b, c, alpha, beta, gamma])
        atoms = Atoms(numbers=types, scaled_positions=coords, cell=cell_matrix, pbc=True)

        # CIF 저장
        cif_path = os.path.join(cif_dir, f"{base_name}_{i:03d}.cif")
        write(cif_path, atoms)

        # PNG 저장
        fig, ax = plt.subplots()
        plot_atoms(atoms, ax, rotation=("90x,90y,90z"), show_unit_cell=2)
        ax.set_axis_off()
        fig.savefig(os.path.join(png_dir, f"{base_name}_{i:03d}.png"), bbox_inches="tight")
        plt.close(fig)

    print(f"✅ {total_structures} structures saved to:\n - {cif_dir}\n - {png_dir}")

# 실행
visualize_from_gen_pt(pt_path)
