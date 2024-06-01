from player import Player
import numpy as np

class Randy1(Player):
    def __init__(self):
        super().__init__('Randy1', (255, 0, 0))

    def action(self, observation) -> int:
        position = observation[1][self.order]
        owner = observation[0][position]
        if owner == -2:
            return 0
        elif owner == -1:
            return 1 if np.random.rand() < 0.5 else 0
        elif owner == self.order:
            # If it can yet build houses, do it
            if observation[5][self.order] < 5:
                return 2 if np.random.rand() < 0.5 else 0
            return 0
        else:
            return 0