from player import Player
import numpy as np

class Randy1(Player):
    def __init__(self, name='Randy1', color=(255, 255, 0)):
        """
        Constructor actualizado para permitir cambiar el nombre y color
        si se instancia directamente.
        """
        super().__init__(name, color)

    def action(self, observation) -> int:
        """
        Decide una acción basada en el game_state.
        'observation' que recibe este método es la lista 'game_state' completa.
        """
        
        # --- Índices del game_state (para claridad) ---
        PROPERTIES = 0
        POSITIONS = 1
        BUILDINGS = 5
        # --- Fin Índices ---

        # self.order es asignado por el env.reset()
        player_id = self.order
        
        position = observation[POSITIONS][player_id]
        owner = observation[PROPERTIES][position]

        if owner == -2:
            # Casilla no comprable (Cárcel, Suerte, etc.)
            return 0 # No-op
        
        elif owner == -1:
            # Casilla sin dueño, 50% de probabilidad de comprar
            if np.random.rand() < 0.5:
                # ¡ACCIÓN ACTUALIZADA!
                return 41 # Acción para "Comprar"
            else:
                return 0 # No-op
        
        elif owner == player_id:
            # Es nuestra propiedad
            
            # --- LÓGICA CORREGIDA ---
            # Comprobamos el nivel de edificación de la *casilla actual* (position),
            # no de la casilla 'player_id'.
            current_build_level = observation[BUILDINGS][position]
            
            # (Este bot simple no comprueba si tiene el monopolio o dinero suficiente,
            # pero la máscara de acción del entorno lo bloqueará si es inválido)
            if current_build_level < 5: # 5 = Hotel
                # 50% de probabilidad de edificar
                if np.random.rand() < 0.5:
                    # ¡ACCIÓN ACTUALIZADA!
                    # Devuelve la acción específica para edificar en *esta* casilla
                    return 1 + position 
                else:
                    return 0 # No-op
            else:
                return 0 # Ya tiene hotel, No-op
        
        else:
            # Casilla de otro jugador
            return 0 # No-op