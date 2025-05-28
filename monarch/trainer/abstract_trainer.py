from abc import ABCMeta, abstractmethod

class AbstractTrainer(metaclass=ABCMeta):
    @abstractmethod
    def forward(self):

        pass

    @abstractmethod
    def translate_instances(self):

        pass

    @abstractmethod
    def rotate_instances(self):

        pass
