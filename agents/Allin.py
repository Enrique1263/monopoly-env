from player import Player
import numpy as np

class Allin(Player):
    def __init__(self, name='Allyn', color=(255, 255, 0)):
        """
        Constructor actualizado para permitir cambiar el nombre y color
        si se instancia directamente.
        """
        super().__init__(name, color)

    def action(self, observation, action_mask=None) -> int:
        """
        Compra y edifica siempre que pueda.
        """
        # --- Índices del game_state (para claridad) ---
        PROPERTIES = 0
        POSITIONS = 1
        BUILDINGS = 5
        MONEY = 2
        COST = 3
        # --- Fin Índices ---

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
            # Es nuestra propiedad
            
            current_build_level = observation[BUILDINGS][position]
            
            if current_build_level < 5: # 5 = Hotel
                # Siempre edifica si puede
                build_cost = 0
                if current_build_level == 0:
                    build_cost = 50  # Casa
                elif current_build_level < 4:
                    build_cost = 50 + (current_build_level * 50)  # Casas incrementales
                else:
                    build_cost = 200  # Hotel
                
                if money >= build_cost:
                    return 1 + position 
                else:
                    return 0 # No-op
            else:
                return 0 # No-op
        
        else:
            # Propiedad de otro jugador, no puede hacer nada
            return 0 # No-op