
import os
import time
import argparse
import torch
from types import SimpleNamespace
from pathlib import Path
from omegaconf import OmegaConf

from CatalystGen.modules.model import CatalystGen

# ✅ 흡착종 목록 (README 참고)
adsorbate_list = ["CHCO", "CHOCHO", "NO2NO2", "COCH2O"]

# ✅ 허용 atom 타입 세트 (README 참고)
allowed_atom_sets = {
    0: [82, 16, 19, 34],
    1: [19, 33, 16, 46],
    2: [47, 16, 17, 40],
    3: [19, 51, 14, 45],
}

def parse_args():
    parser = argparse.ArgumentParser(description="CatalystGen Structure Generator")
    parser.add_argument("--ads_idx", type=int, choices=range(4), required=True,
                        help="Index of adsorbate (0: CHCO, 1: CHOCHO, 2: NO2NO2, 3: COCH2O)")
    parser.add_argument("--atom_set", type=int, choices=range(4), required=True,
                        help="Atom set index (0–3)")
    parser.add_argument("--label", type=str, default="v1", help="Label for output file")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the .ckpt file to load")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=100)
    return parser.parse_args()

# ✅ 모델 로딩
def load_model(ckpt_path):
    ckpt_path = Path(ckpt_path).resolve()
    weights_dir = ckpt_path.parent
    hparams_path = weights_dir / "hparams.yaml"

    cfg = OmegaConf.load(hparams_path)

    model = CatalystGen.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        **cfg.model,
        optim=cfg.optim,
        logging=cfg.logging,
        _recursive_=False,
    )

    model.lattice_scaler = torch.load(weights_dir / "lattice_scaler.pt")
    model.scaler = torch.load(weights_dir / "prop_scaler.pt")
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"📦 Loaded checkpoint: {ckpt_path.name}")
    return model, cfg

# ✅ 샘플 생성
def generate_samples(model, ld_kwargs, num_samples, latent_dim, batch_size, absorbate_id, allowed_atom_types):
    outputs = []
    for _ in range(num_samples // batch_size):
        z = torch.randn(batch_size, latent_dim, device=model.device)
        absorbate_id_tensor = torch.full((batch_size,), absorbate_id, device=model.device, dtype=torch.long)
        result = model.langevin_dynamics(z, ld_kwargs, absorbate_id=absorbate_id_tensor, allowed_atom_types=allowed_atom_types)
        outputs.extend(result if isinstance(result, list) else [result])
    return outputs

# ✅ 저장
def save_outputs(outputs, out_path):
    def stack(key):
        return torch.cat([out[key].cpu() for out in outputs], dim=0)

    torch.save({
        "frac_coords": stack("frac_coords"),
        "atom_types": stack("atom_types"),
        "miller_index": stack("miller_index"),
        "lengths": stack("lengths"),
        "angles": stack("angles"),
        "num_atoms": stack("num_atoms"),
    }, out_path)

# ✅ 메인
if __name__ == "__main__":
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    adsorbate = adsorbate_list[args.ads_idx]
    atom_types = allowed_atom_sets[args.atom_set]

    output_dir = project_root / "generated" / adsorbate
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gen_{args.label}.pt"

    model, cfg = load_model(args.ckpt_path)

    ld_kwargs = SimpleNamespace(
        n_step_each=60,
        step_lr=0.0005,
        min_sigma=0.0,
        save_traj=False,
        disable_bar=False,
    )

    print(f"🚀 Generating for [{adsorbate}] with atom set {args.atom_set}")
    start = time.time()
    outputs = generate_samples(
        model=model,
        ld_kwargs=ld_kwargs,
        num_samples=args.num_samples,
        latent_dim=cfg.model.latent_dim,
        batch_size=args.batch_size,
        absorbate_id=args.ads_idx,
        allowed_atom_types=atom_types,
    )
    save_outputs(outputs, output_path)
    print(f"✅ Done. Saved to: {output_path}")
    print(f"⏱️ Elapsed: {time.time() - start:.2f} sec")
