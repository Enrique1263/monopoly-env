from player import Player
import numpy as np

class Terry(Player):
    def __init__(self, name='Terry', color=(0, 0, 255)):
        """
        Constructor actualizado para permitir cambiar el nombre y color
        si se instancia directamente.
        """
        super().__init__(name, color)

    def action(self, observation, action_mask=None) -> int:
        """
        Acción terrateniente, compra siempre que pueda pero no edifica.
        """
        # --- Índices del game_state (para claridad) ---
        PROPERTIES = 0
        POSITIONS = 1
        MONEY = 2
        COST = 3
        BUILDINGS = 5

        # self.order es asignado por el env.reset()
        player_id = self.order
        
        position = observation[POSITIONS][player_id]
        owner = observation[PROPERTIES][position]
        money = observation[MONEY][player_id]
        cost = observation[COST][position][-1]

        if owner == -2:
            # Casilla no comprable (Cárcel, Suerte, etc.)
            return 0 # No-op
        
        elif owner == -1:
            # Casilla sin dueño, siempre compra si puede
            if money >= cost:
                return 41 # Acción para "Comprar"
            else:
                return 0 # No-op
        
        elif owner == player_id:
            # Es nuestra propiedad, no edifica nunca
            return 0 # No-op
        
        else:
            # Propiedad de otro jugador, no puede hacer nada
            return 0 # No-op
        