import gymnasium as gym
import numpy as np
import inspect

class SingleAgentMonopolyWrapper(gym.Wrapper):
    def __init__(self, env, opponents):
        super().__init__(env)
        
        self.env_players = self.env.unwrapped.players
        assert len(self.env_players) == len(opponents) + 1, \
            "El entorno debe tener 1 jugador más que la lista de oponentes."
        
        self.opponents = self.env_players[1:]

        # Cacheamos si los bots aceptan máscara para no usar inspect en el bucle
        self.bot_accepts_mask = []
        for bot in self.opponents:
            sig = inspect.signature(bot.action)
            self.bot_accepts_mask.append('action_mask' in sig.parameters)

        # Normalización del espacio de observación
        old_space = self.env.observation_space
        low = old_space.low.copy()
        high = old_space.high.copy()
        
        money_start = 40 + 40 + len(self.env_players)
        money_end = money_start + len(self.env_players)
        low[money_start:money_end] = 0.0
        high[money_start:money_end] = 20.0
        
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _normalize_obs(self, obs):
        if obs is None: return None
        normalized_obs = obs.copy()
        num_players = len(self.env_players)
        money_start = 40 + 40 + num_players
        money_end = money_start + num_players
        normalized_obs[money_start:money_end] /= 1000.0
        return normalized_obs

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        if self.env_players[0].name != "Learning_Hero":
             print(f"CRITICAL: Agente descolocado. Es {self.env_players[0].name}.")

        if info["player_on_turn"] != 0:
            obs, _, terminated, truncated, info = self._play_opponents_turns(info, current_obs=obs)
            if terminated or truncated:
                return self.reset(seed=seed, options=options)
        
        return self._normalize_obs(obs), info

    def step(self, action):
        # 1. Turno del Agente
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check rápido de bancarrota del agente
        if info["game_state"][7][0]: # 7 = BANKRUPT
            terminated = True

        if terminated or truncated:
            return self._normalize_obs(obs), reward, terminated, truncated, info
        
        # 2. Turnos de Oponentes
        obs, _, terminated, truncated, info = self._play_opponents_turns(info, current_obs=obs)
        
        if info["game_state"][7][0]: 
            terminated = True

        return self._normalize_obs(obs), reward, terminated, truncated, info

    def _play_opponents_turns(self, info, current_obs):
        terminated = False
        truncated = False
        obs = current_obs 
        
        while info["player_on_turn"] != 0 and not (terminated or truncated):
            
            if info["game_state"][7][0]: # Agente en bancarrota
                terminated = True
                break

            current_player_idx = info["player_on_turn"]
            game_state = info["game_state"]
            opponent_idx = current_player_idx - 1
            
            if 0 <= opponent_idx < len(self.opponents):
                bot = self.opponents[opponent_idx]
                
                # --- OPTIMIZACIÓN: USAR MÁSCARA DEL INFO ---
                # El entorno ya calculó la máscara válida para 'player_on_turn' 
                # y la puso en info['action_mask']. ¡La usamos!
                if self.bot_accepts_mask[opponent_idx]:
                    action_mask = info['action_mask']
                    action = bot.action(game_state, action_mask=action_mask)
                else:
                    action = bot.action(game_state)
            else:
                action = 0 
            
            # Al hacer step, 'info' se actualiza con la máscara del SIGUIENTE jugador
            obs, _, terminated, truncated, info = self.env.step(action)
            
        return obs, 0.0, terminated, truncated, info

    def action_masks(self):
        """
        Requerido por MaskablePPO para el Agente Principal.
        Aquí sí llamamos al entorno porque no tenemos un 'info' reciente fuera del step.
        """
        return self.env.unwrapped._get_action_mask()