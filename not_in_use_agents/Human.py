from player import Player
import numpy as np

class Human(Player):
    def __init__(self):
        super().__init__('Human', (255, 255, 0))

    def action(self, observation) -> int:
        a = input('Enter action: ')
        return int(a) if a.isdigit() else 0