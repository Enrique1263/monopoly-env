import abc

class Player(metaclass=abc.ABCMeta):
    def __init__(self, name, color):
        """Constructor for the player
        :param name: the name of the player
        :param color: the color of the player
        """
        self.name = name
        self.color = color
        self.order = None

    @abc.abstractmethod
    def action(self, observation) -> int:
        """Given the observation, return the action to be taken by the player
        :param observation: the current observation
        :return: the action to be taken
        """
        
    def __str__(self) -> str:
        """Return the name of the player"""
        return self.name