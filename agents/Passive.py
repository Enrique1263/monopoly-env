from player import Player
import numpy as np

class Passive(Player):
    def __init__(self, name='Passive', color=(0, 255, 0)):
        """
        Constructor actualizado para permitir cambiar el nombre y color
        si se instancia directamente.
        """
        super().__init__(name, color)

    def action(self, observation, action_mask=None) -> int:
        """
        Decide una acción basada en el game_state.
        'observation' que recibe este método es la lista 'game_state' completa.
        """
        return 0 # No-op siempre