

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
from types import SimpleNamespace
from omegaconf import OmegaConf
from pathlib import Path
from CatalystGen.modules.model import CatalystGen


# ✅ 흡착종 인덱스 매핑
adsorbate_dict =  {
    "CHCO": 0,
    "CHOCHO": 1,
    "NO2NO2": 2,
    "COCH2O": 3,
}

# ✅ 모델 로딩 함수 (고정 경로 사용)
def load_model():
    ckpt_path = "/home/kaist/projects/CatalystGen-main/weights/all_ads/epoch=14-step=2505.ckpt"
    hparams_path = ckpt_path.replace(".ckpt", "/../hparams.yaml")
    hparams_path = os.path.abspath(hparams_path)

    cfg = OmegaConf.load(hparams_path)

    model = CatalystGen.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        **cfg.model,
        optim=cfg.optim,
        logging=cfg.logging,
        _recursive_=False,
    )

    model.lattice_scaler = torch.load(os.path.join(os.path.dirname(ckpt_path), "lattice_scaler.pt"))
    model.scaler = torch.load(os.path.join(os.path.dirname(ckpt_path), "prop_scaler.pt"))
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, cfg


# ✅ 구조 생성 함수
def generate_samples(model, ld_kwargs, num_samples, latent_dim, batch_size, absorbate_id, allowed_atom_types):
    outputs = []
    for _ in range(num_samples // batch_size):
        z = torch.randn(batch_size, latent_dim, device=model.device)
        absorbate_id_tensor = torch.full((batch_size,), absorbate_id, device=model.device, dtype=torch.long)
        result = model.langevin_dynamics(z, ld_kwargs, absorbate_id=absorbate_id_tensor, allowed_atom_types=allowed_atom_types)
        outputs.extend(result if isinstance(result, list) else [result])
    return outputs


# ✅ 저장 함수
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


# ✅ 메인 실행
if __name__ == "__main__":
    adsorbate = "COCH2O"   # ← 여기를 바꿔서 다른 흡착종도 실행 가능
    label = "v1"
    num_samples = 300
    batch_size = 100
    n_step_each = 60
    step_lr = 0.0005
    min_sigma = 0.0
    allowed_atom_types_0 = [82, 16, 19, 34]
    allowed_atom_types_1 = [19, 33, 16, 46]
    allowed_atom_types_2 = [47, 16, 17, 40]
    allowed_atom_types_3 = [19, 51, 14, 45]

    # 경로 설정
    project_root = "/home/kaist/projects/CatalystGen-main"
    base_dir = Path(project_root) / "generated2" / adsorbate
    base_dir.mkdir(parents=True, exist_ok=True)
    output_file = base_dir / f"gen_{label}.pt"

    # 모델 로딩
    model, cfg = load_model()

    # Langevin config
    ld_kwargs = SimpleNamespace(
        n_step_each=n_step_each,
        step_lr=step_lr,
        min_sigma=min_sigma,
        save_traj=False,
        disable_bar=False,
    )

    print(f"🚀 Generating for [{adsorbate}]")
    start = time.time()
    absorbate_id = adsorbate_dict[adsorbate]
    outputs = generate_samples(
        model=model,
        ld_kwargs=ld_kwargs,
        num_samples=num_samples,
        latent_dim=cfg.model.latent_dim,
        batch_size=batch_size,
        absorbate_id=absorbate_id,
        allowed_atom_types=allowed_atom_types_3,
    )
    save_outputs(outputs, output_file)
    print(f"✅ Done. Saved to: {output_file}")
    print(f"⏱️ Elapsed: {time.time() - start:.2f} sec")

