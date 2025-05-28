from abc import ABCMeta, abstractmethod
from omegaconf import OmegaConf
from types.pixel_type import PixelSource

class AbstractDrivingDataset(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the dataset
        :return dataset name
        """
        pass
