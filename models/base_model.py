from abc import abstractmethod


class BaseModel:
    def __len__(self):
        return len(self.params)

    @property
    @abstractmethod
    def params(self):
        pass
    
    @abstractmethod
    def __call__(self, *args):
        pass

    @abstractmethod
    def update(self, *args):
        pass

    @abstractmethod
    def loss(self, *args):
        pass
    
    @abstractmethod
    def total_loss(self, *args):
        pass
