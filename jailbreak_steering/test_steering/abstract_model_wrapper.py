import torch
import abc

class AbstractModelWrapper(abc.ABC):
    def __init__(self):
        super().__init__()
        self.hf_model = None
        self.tokenizer = None

    def __call__(self, *args, **kwargs):
        return self.hf_model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.hf_model.generate(*args, **kwargs)

    @abc.abstractmethod
    def wrap_model(self, hf_model, tokenizer):
        pass

    @abc.abstractmethod
    def unwrap_model(self):
        pass
