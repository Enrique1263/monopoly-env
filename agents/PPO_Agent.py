from player import Player
from sb3_contrib import MaskablePPO
import numpy as np
import os

class PPO_Agent(Player):
    def __init__(self, name='PPO_Agent', color=(255, 0, 0), model_path='models/ppo_league/best_model'):
        super().__init__(name, color)
        self.model = None
        
        real_path = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
        if os.path.exists(real_path):
            self.model = MaskablePPO.load(real_path)
        else:
            print(f"ADVERTENCIA: No se encontró modelo en {real_path}. {name} no hará nada.")

    def action(self, game_state, action_mask=None) -> int:
        if self.model is None:
            return 0 
        
        # Si no recibimos máscara (por ejemplo, desde un script antiguo), fallback permisivo
        if action_mask is None:
            action_mask = np.ones(43, dtype=bool) 

        # 1. Reconstruir observación (NORMALIZADA)
        obs_vector = self._reconstruct_obs(game_state)
        
        # 2. Predecir usando la máscara entregada
        action, _ = self.model.predict(obs_vector, action_masks=action_mask, deterministic=True)
        return int(action)

    def _reconstruct_obs(self, gs):
        """
        Reconstruye el vector de observación aplicando:
        1. Normalización del dinero.
        2. Rotación de perspectiva (Ego-céntrica).
        3. Padding (Relleno) si faltan jugadores respecto al entrenamiento.
        """
        PROPERTIES, POSITIONS, MONEY = 0, 1, 2
        BUILDINGS, JAIL_TURNS, GET_OUT_OF_JAIL = 5, 6, 8
        
        # Datos actuales
        num_current_players = len(gs[POSITIONS])

        # --- 1. PROPIEDADES (Transformación de Dueño) ---
        # Si soy el Jugador 2, mis propiedades (valor 2) deben ser 0.
        # Las del Jugador 3 (valor 3) deben ser 1.
        properties = np.array(gs[PROPERTIES], dtype=np.float32)
        owned_mask = properties >= 0 # Ignorar -1 (Libre) y -2 (No comprable)
        
        # Fórmula: (Dueño_Original - Mi_Orden) % Num_Jugadores_Actuales
        properties[owned_mask] = (properties[owned_mask] - self.order) % num_current_players

        # Edificios (Globales, no cambian)
        buildings = np.array(gs[BUILDINGS], dtype=np.float32)

        # --- 2. DATOS DE JUGADORES (Rotación de Arrays) ---
        # Extraer datos crudos
        raw_pos = np.array(gs[POSITIONS], dtype=np.float32)
        raw_money = np.array(gs[MONEY], dtype=np.float32) / 20000.0 # Normalizado
        raw_jail = np.array(gs[JAIL_TURNS], dtype=np.float32)
        raw_cards = np.array(gs[GET_OUT_OF_JAIL], dtype=np.float32)

        # Rotar arrays: Mover mi índice (self.order) a la posición 0
        shift = -self.order
        rot_pos = np.roll(raw_pos, shift)
        rot_money = np.roll(raw_money, shift)
        rot_jail = np.roll(raw_jail, shift)
        rot_cards = np.roll(raw_cards, shift)
        
        # --- 3. PADDING (Relleno si faltan jugadores) ---
        expected_obs_size = self.model.observation_space.shape[0]
        # 80 bytes fijos (props+builds). El resto son 4 campos por jugador.
        expected_num_players = (expected_obs_size - 80) // 4
        
        if num_current_players < expected_num_players:
            diff = expected_num_players - num_current_players
            
            # Rellenamos con ceros al final (son jugadores fantasma)
            rot_pos = np.concatenate([rot_pos, np.zeros(diff, dtype=np.float32)])
            rot_money = np.concatenate([rot_money, np.zeros(diff, dtype=np.float32)])
            rot_jail = np.concatenate([rot_jail, np.zeros(diff, dtype=np.float32)])
            rot_cards = np.concatenate([rot_cards, np.zeros(diff, dtype=np.float32)])
            
        elif num_current_players > expected_num_players:
            print("ERROR CRÍTICO: Hay más jugadores que los que el modelo soporta.")
            # Recorte de emergencia
            rot_pos = rot_pos[:expected_num_players]
            rot_money = rot_money[:expected_num_players]
            rot_jail = rot_jail[:expected_num_players]
            rot_cards = rot_cards[:expected_num_players]

        # 4. Concatenar todo
        return np.concatenate([
            properties, 
            buildings, 
            rot_pos, 
            rot_money, 
            rot_jail, 
            rot_cards
        ])