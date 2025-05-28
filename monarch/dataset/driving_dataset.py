from dataset.abstract_driving_dataset import AbstractDrivingDataset

class DrivingDataset(AbstractDrivingDataset):
    def __init__(self, name: str, dataset: ) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """
        Inherited, see super class
        """
        return self._name
