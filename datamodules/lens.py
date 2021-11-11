from typing import Any, Callable, Optional, Union

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from torchvision import transforms as transform_lib
from astrovision.datasets import LensChallengeSpace1


class LensChallengeSpace1DataModule(VisionDataModule):
    name = "celeb"
    dataset_cls = LensChallengeSpace1
    dims = (1, 101, 101)

    def __init__(
            self,
            data_dir: Optional[str] = None,
            val_split: Union[int, float] = 0.2,
            num_workers: int = 0,
            normalize: bool = False,
            batch_size: int = 32,
            seed: int = 42,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """

        super().__init__(*args, **kwargs)

        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        return 40

    def default_transforms(self) -> Callable:
        if self.normalize:
            transforms = transform_lib.Compose([transform_lib.ToTensor()])
        else:
            transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return transforms
