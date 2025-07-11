from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self):
        self.memory = None  # Optional, for logging compatibility

    @abstractmethod
    def get_action(self, state, train: bool):
        """Return action for given state. If train is True, may use exploration."""
        pass

    @abstractmethod
    def learn(self):
        """Perform a learning step (if applicable)."""
        pass

    @abstractmethod
    def save(self, path):
        """Save model/checkpoint to path."""
        pass

    @abstractmethod
    def load(self, path):
        """Load model/checkpoint from path."""
        pass 