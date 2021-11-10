from typing import Any, Callable, Optional, Union

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import transforms as transform_lib
from torchvision.datasets import CelebA


class CelebADataModule(LightningDataModule):
    EXTRA_ARGS: dict = {}
    name = "celeb"
    dataset_cls = CelebA
    dims = (3, 178, 218)

    def __init__(
        self,
        data_dir: Optional[str] = None,
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
        self.normalize = normalize
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        return 40

    def prepare_data(self) -> None:
        """Downloads the train, validation and test split."""
        dataset = CelebA(self.data_dir, split="train", download=True, transform=transform_lib.ToTensor())
        dataset = CelebA(self.data_dir, split="valid", download=True, transform=transform_lib.ToTensor())
        dataset = CelebA(self.data_dir, split="test", download=True, transform=transform_lib.ToTensor())

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            self.dataset_train = self.dataset_cls(self.data_dir, split="train", transform=train_transforms, **self.EXTRA_ARGS)
            self.dataset_val = self.dataset_cls(self.data_dir, split="valid", transform=val_transforms, **self.EXTRA_ARGS)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir, split="test", transform=test_transforms, **self.EXTRA_ARGS
            )

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def default_transforms(self) -> Callable:
        if self.normalize:
            transforms = transform_lib.Compose([transform_lib.ToTensor()])
        else:
            transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return transforms
