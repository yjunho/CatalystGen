import os
import pytorch_lightning as pl
from typing import Optional
from torch_geometric.data import DataLoader

from CatalystGen.dataset.dataset import CatalystDataset
from CatalystGen.common.data_utils import StandardScalerTorch, get_scaler_from_data_list

from torch_geometric.loader import DataLoader
from CatalystGen.dataset.dataset import CatalystDataset

class CatalystDataModule (pl.LightningDataModule):
    def __init__(self,
                 datasets: dict,
                 batch_size: dict,
                 num_workers: dict):
        """
        :param datasets: dict 형태로 train, val, test 각각의 경로 문자열
        :param batch_size: {'train': int, 'val': int, 'test': int}
        :param num_workers: {'train': int, 'val': int, 'test': int}
        """
        super().__init__()
        self.datasets_config = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.lattice_scaler = None  
        self.scaler = None

    def setup(self, stage=None):
        # ✅ 직접 CatalystDataset 생성 (instantiate 제거)
        self.train_dataset = CatalystDataset(self.datasets_config["train"])
        self.val_dataset = CatalystDataset(self.datasets_config["val"])
        self.test_dataset = CatalystDataset(self.datasets_config["test"])

        # ✅ 스케일러 계산
        self.lattice_scaler = get_scaler_from_data_list(
            self.train_dataset.cached_data, key="scaled_lattice"
        )
        
        self.train_dataset.lattice_scaler = self.lattice_scaler
        self.val_dataset.lattice_scaler = self.lattice_scaler
        self.test_dataset.lattice_scaler = self.lattice_scaler
      

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size["train"],
                          shuffle=True,
                          num_workers=self.num_workers["train"])

    def val_dataloader(self):
        print("✅ val_dataloader 호출됨")
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size["val"],
                          shuffle=False,
                          num_workers=self.num_workers["val"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size["test"],
                          shuffle=False,
                          num_workers=self.num_workers["test"])

    def prepare_data(self):
        # 데이터 다운로드 등 전처리 로직이 필요한 경우에 사용
        pass


