
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hydra
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from CatalystGen.modules.model import CatalystGen
from CatalystGen.dataset.datamodule import CatalystDataModule

@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig):
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    project_root = "/home/kaist/projects/CatalystGen-main"
    data_root = os.path.join(project_root, "data/selected_data")
    save_dir = os.path.join(project_root, "weights/all_ads")
    os.makedirs(save_dir, exist_ok=True)

    # ✅ 통합된 csv 경로
    train_csv = os.path.join(data_root, "train.csv")
    val_csv = os.path.join(data_root, "val.csv")
    test_csv = os.path.join(data_root, "test.csv")

    datamodule = CatalystDataModule(
        datasets={"train": train_csv, "val": val_csv, "test": test_csv},
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    model = CatalystGen(
        **cfg.model,
        optim=cfg.optim,
        logging=cfg.logging,
        _recursive_=False,
    )
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    torch.save(datamodule.lattice_scaler, os.path.join(save_dir, "lattice_scaler.pt"))
    torch.save(datamodule.scaler, os.path.join(save_dir, "prop_scaler.pt"))

    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir,
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            save_top_k=2,
            verbose=cfg.train.model_checkpoints.verbose,
        ),
        EarlyStopping(
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            patience=cfg.train.early_stopping.patience,
            verbose=cfg.train.early_stopping.verbose,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        max_epochs=cfg.train.pl_trainer.max_epochs,
        accelerator="gpu",
        devices=[0],
        precision="16-mixed",
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    OmegaConf.save(config=cfg, f=os.path.join(save_dir, "hparams.yaml"))

    if hasattr(trainer.logger, "log_dir") and os.path.exists(trainer.logger.log_dir):
        metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

if __name__ == "__main__":
    main()


