import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from full.datasets_2d_v1 import ZJUMOCAPDataset


ZJUMOCAPDataConfig = {
    
    "CoreView_315": {
        "train_start_frame": 0,
        "train_end_frame": 1100,
        "train_sampling_rate": 1,
        
        "val_start_frame": 0,
        "val_end_frame": 1401,
        "val_sampling_rate": 200,
        
        "nv_test_start_frame": 0,
        "nv_test_end_frame": 1100,

        "np_test_start_frame": 1100,
        "np_test_end_frame": 1401,
        
        "test_sampling_rate": 15,
        
        "train_views": ['1', '2', '3', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15', '17', '18', '19', '23'],
        "val_views": ['1', '7', '13', '19'],
        "nv_test_views": ['4', '10', '16', '22'],
        "np_test_views": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '22', '23'],
    },

    "CoreView_386": {
        "train_start_frame": 0,
        "train_end_frame": 500,
        "train_sampling_rate": 1,
        
        "val_start_frame": 0,
        "val_end_frame": 600,
        "val_sampling_rate": 100,
        
        "nv_test_start_frame": 0,
        "nv_test_end_frame": 500,

        "np_test_start_frame": 500,
        "np_test_end_frame": 647,
        
        "test_sampling_rate": 15,
        
        "train_views": ['1', '2', '3', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15', '17', '18', '19', '20', '21', '23'],
        "val_views": ['1', '7', '13', '19'],
        "nv_test_views": ['4', '10', '16', '22'],
        "np_test_views": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
    },
}

class ZJUMOCAPDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "path/to/dir",
            subject: str = "CoreView_315",
            train_batch_size: int = 1,
            val_batch_size: int = 1,
            test_batch_size: int = 1,
            image_size: int = 512,
            num_workers: int = 4,
            erode_mask = False,
            dataset_folder_aux_undistorted = None,
            train_dataset_repeat = 1,
            test_mode = "nv",
            train_sample_views = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.subject = subject
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.erode_mask = erode_mask
        self.dataset_folder_aux_undistorted = dataset_folder_aux_undistorted
        self.train_dataset_repeat = train_dataset_repeat
        self.test_mode = test_mode
        self.train_sample_views = train_sample_views

    def setup(self, stage: str):
        if stage == "fit": 
            self.train_dataset = ZJUMOCAPDataset(
                dataset_folder = self.data_dir,
                dataset_folder_aux_undistorted = self.dataset_folder_aux_undistorted,
                subjects = [self.subject],
                mode = 'train',
                img_size = (self.image_size, self.image_size),
                sampling_rate = ZJUMOCAPDataConfig[self.subject]['train_sampling_rate'],
                start_frame = ZJUMOCAPDataConfig[self.subject]['train_start_frame'],
                end_frame = ZJUMOCAPDataConfig[self.subject]['train_end_frame'],
                views = ZJUMOCAPDataConfig[self.subject]['train_views'],
                box_margin = ZJUMOCAPDataConfig[self.subject].get("box_margin", 0.05),
                erode_mask = self.erode_mask,
                sample_views = self.train_sample_views,
                repeat = self.train_dataset_repeat,
            )
        if stage == "fit" or stage == "validate":
            self.val_dataset = ZJUMOCAPDataset(
                dataset_folder = self.data_dir,
                dataset_folder_aux_undistorted = self.dataset_folder_aux_undistorted,
                subjects = [self.subject],
                mode = 'val',
                img_size = (self.image_size, self.image_size),
                sampling_rate = ZJUMOCAPDataConfig[self.subject]['val_sampling_rate'],
                start_frame = ZJUMOCAPDataConfig[self.subject]['val_start_frame'],
                end_frame = ZJUMOCAPDataConfig[self.subject]['val_end_frame'],
                views = ZJUMOCAPDataConfig[self.subject]['val_views'],
                box_margin = ZJUMOCAPDataConfig[self.subject].get("box_margin", 0.05),
                erode_mask = self.erode_mask,
                sample_views = -1,
            )
        if stage == "test":
            self.test_dataset = ZJUMOCAPDataset(
                dataset_folder = self.data_dir,
                dataset_folder_aux_undistorted = self.dataset_folder_aux_undistorted,
                subjects = [self.subject],
                mode = 'test',
                img_size = (self.image_size, self.image_size),
                sampling_rate = ZJUMOCAPDataConfig[self.subject]['test_sampling_rate'],
                start_frame = ZJUMOCAPDataConfig[self.subject][self.test_mode + "_test_start_frame"],
                end_frame = ZJUMOCAPDataConfig[self.subject][self.test_mode + "_test_end_frame"],
                views = ZJUMOCAPDataConfig[self.subject][self.test_mode + "_test_views"],
                box_margin = ZJUMOCAPDataConfig[self.subject].get("box_margin", 0.05),
                erode_mask = self.erode_mask,
                sample_views = -1,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers)
