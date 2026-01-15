import os
import sys
import gymnasium as gym
import numpy as np
from collections import Counter, defaultdict
from sb3_contrib import MaskablePPO

# Importamos para registro de entorno
import monopoly_env
from player import Player

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/ppo_league/best_model" # Ruta a tu campeón
N_GAMES = 1000
# ---------------------

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

class RotatedPPOAgent(Player):
    """
    Agente que usa un modelo PPO entrenado como 'Jugador 0', pero puede jugar
    en cualquier posición (0, 1, 2, 3) rotando su percepción del estado
    para que siempre parezca que él es el Jugador 0 (Ego-céntrico).
    Incluye padding automático si el modelo espera más jugadores.
    """
    def __init__(self, name, color, model_path, player_index):
        super().__init__(name, color)
        self.order = player_index # Su posición real en la mesa (0-3)
        self.model = None
        
        # Cargar modelo
        base_path = model_path.replace(".zip", "")
        full_path = base_path + ".zip"
        if os.path.exists(full_path):
            self.model = MaskablePPO.load(full_path)
        else:
            print(f"ERROR: No se encontró modelo en {full_path}")

    def action(self, game_state, action_mask=None) -> int:
        if self.model is None: return 0
        if action_mask is None: action_mask = np.ones(43, dtype=bool)

        # 1. Rotar y construir observación
        # El agente necesita ver el mundo como si él fuera el índice 0.
        obs_vector = self._reconstruct_rotated_obs(game_state)
        
        # 2. Predecir
        action, _ = self.model.predict(obs_vector, action_masks=action_mask, deterministic=True)
        return int(action)

    def _reconstruct_rotated_obs(self, gs):
        """
        Construye el vector de observación rotando los arrays de jugadores
        y TRANSFOMANDO los dueños de propiedades basado en self.order.
        """
        PROPERTIES, POSITIONS, MONEY = 0, 1, 2
        BUILDINGS, JAIL_TURNS, GET_OUT_OF_JAIL = 5, 6, 8
        
        num_players = len(gs[POSITIONS])

        # --- 1. ROTACIÓN DE PROPIEDADES (CRÍTICO) ---
        # El array properties contiene índices de jugadores (0, 1, 2...).
        # Si yo soy el jugador 2, una propiedad mía (valor 2) debe parecer valor 0 (mía) al modelo.
        # Una propiedad del jugador 3 (valor 3) debe parecer valor 1 (del siguiente).
        
        properties = np.array(gs[PROPERTIES], dtype=np.float32)
        
        # Máscara para solo rotar casillas que tienen dueño (>= 0)
        # Ignoramos -1 (Libre) y -2 (No comprable)
        owned_mask = properties >= 0
        
        # Aplicamos la rotación modular relativa a mi orden
        # Formula: (Dueño_Original - Mi_Orden) % Num_Jugadores
        properties[owned_mask] = (properties[owned_mask] - self.order) % num_players

        # --- 2. RESTO DE DATOS ---
        buildings = np.array(gs[BUILDINGS], dtype=np.float32)
        
        # Datos específicos de jugador (SÍ rotan de posición en el array)
        raw_pos = np.array(gs[POSITIONS], dtype=np.float32)
        raw_money = np.array(gs[MONEY], dtype=np.float32) / 20000.0 # Normalizado
        raw_jail = np.array(gs[JAIL_TURNS], dtype=np.float32)
        raw_cards = np.array(gs[GET_OUT_OF_JAIL], dtype=np.float32)
        
        # --- ROTACIÓN DE POSICIONES EN EL VECTOR ---
        # Si soy el jugador 2, mis datos están en el índice 2.
        # Quiero que pasen al índice 0. np.roll con shift negativo hace esto.
        shift = -self.order
        
        rot_pos = np.roll(raw_pos, shift)
        rot_money = np.roll(raw_money, shift)
        rot_jail = np.roll(raw_jail, shift)
        rot_cards = np.roll(raw_cards, shift)
        
        # --- PADDING INTELIGENTE (Corrección de Shape) ---
        expected_obs_size = self.model.observation_space.shape[0] 
        expected_num_players = (expected_obs_size - 80) // 4
        current_num_players = len(rot_pos)
        
        if current_num_players < expected_num_players:
            diff = expected_num_players - current_num_players
            rot_pos = np.concatenate([rot_pos, np.zeros(diff, dtype=np.float32)])
            rot_money = np.concatenate([rot_money, np.zeros(diff, dtype=np.float32)])
            rot_jail = np.concatenate([rot_jail, np.zeros(diff, dtype=np.float32)])
            rot_cards = np.concatenate([rot_cards, np.zeros(diff, dtype=np.float32)])
        
        return np.concatenate([
            properties, buildings, rot_pos, rot_money, rot_jail, rot_cards
        ])

def plot_board_heatmap(data_dict, title):
    if not VISUALIZATION_AVAILABLE: return
    grid = np.full((11, 11), np.nan)
    # Lógica de mapeo idéntica a analyze_agent.py
    for i in range(11): # Bottom
        val = data_dict.get(i, 0)
        grid[10, 10-i] = val if val > 0 else np.nan
    for i in range(11): # Left
        val = data_dict.get(10+i, 0)
        grid[10-i, 0] = val if val > 0 else np.nan
    for i in range(11): # Top
        val = data_dict.get(20+i, 0)
        grid[0, i] = val if val > 0 else np.nan
    for i in range(10): # Right
        val = data_dict.get(30+i, 0)
        grid[i, 10] = val if val > 0 else np.nan

    plt.figure(figsize=(8, 7))
    sns.heatmap(grid, annot=True, fmt=".0f", cmap="YlGnBu", cbar=False, square=True,
                xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def print_selfplay_stats(stats, board_names):
    print("\n" + "="*80)
    print(f"       ANÁLISIS DE SELF-PLAY (1v1v1v1) - {stats['games_played']} Partidas")
    print("="*80)
    
    # --- ESTADÍSTICA NUEVA: Ventaja del Primer Jugador ---
    print(f"\n⚖️  EQUILIBRIO DEL JUEGO (¿Importa quién empieza?)")
    print(f"   Como todos los agentes son idénticos, lo ideal sería ~25% cada uno.")
    print("-" * 60)
    total_wins = sum(stats['wins_by_order'].values())
    if total_wins > 0:
        for i in range(4):
            wins = stats['wins_by_order'][i]
            pct = (wins / total_wins) * 100
            diff = pct - 25.0
            sign = "+" if diff > 0 else ""
            print(f"   Jugador {i} (Turno {i+1}): {wins} victorias ({pct:.1f}%) -> Desviación: {sign}{diff:.1f}%")
    else:
        print("   No hubo ganadores (¿Tablas por límite de tiempo?)")

    # --- ANÁLISIS DEL AGENTE DE MUESTRA (Jugador 0) ---
    print("\n" + "-"*80)
    print("   ESTADÍSTICAS DETALLADAS DEL JUGADOR 0 (Muestra Representativa)")
    print("-"*80)
    
    # Compras
    print(f"\n💰 COMPRAS (Jugador 0)")
    # Ordenar por tasa
    buy_data = []
    for tile, opps in stats["buy_opportunities"].items():
        buys = stats["buy_actions"][tile]
        rate = (buys / opps * 100) if opps > 0 else 0.0
        buy_data.append((tile, opps, buys, rate))
    buy_data.sort(key=lambda x: x[3])
    
    print(f"  {'Propiedad':<30} | {'Tasa %'}")
    for tile, _, _, rate in buy_data:
        name = board_names.get(tile, f"Casilla {tile}")
        marker = "🔥" if rate > 90 else ("❄️" if rate < 10 else "")
        print(f"  {name:<30} | {rate:.1f}% {marker}")

    # Edificación
    print(f"\n🏗️ EDIFICACIÓN (Jugador 0)")
    b_ops = stats['build_ops']
    b_acts = stats['build_acts']
    if b_ops > 0:
        print(f"  Agresividad: Construyó el {b_acts/b_ops*100:.1f}% de las veces que pudo.")
    
    print("="*80)


if __name__ == '__main__':
    # 1. Configurar 4 Clones del Agente
    print(f"Cargando 4 clones del modelo: {MODEL_PATH}...")
    
    players = [
        RotatedPPOAgent("PPO_Pro", (0, 255, 0), MODEL_PATH, player_index=0),
        RotatedPPOAgent("PPO_Clone_1", (255, 0, 0), MODEL_PATH, player_index=1),
        RotatedPPOAgent("PPO_Clone_2", (0, 0, 255), MODEL_PATH, player_index=2),
        RotatedPPOAgent("PPO_Clone_3", (255, 255, 0), MODEL_PATH, player_index=3)
    ]

    # 2. Crear Entorno
    env = gym.make('MonopolyEnv-v0', 
                   players=players, 
                   render_mode=None, 
                   max_steps=2000, 
                   board_names_path='cards/board_names.txt', 
                   community_chest_path='cards/community_chest.txt', 
                   chance_path='cards/chance.txt', 
                   hard_rules=False, 
                   image_path='cards/monopoly.png')

    # Desactivar barajado interno para mantener índices fijos (0,1,2,3)
    env.unwrapped.star_order = lambda: None
    # Asegurar orden interno
    for i, p in enumerate(env.unwrapped.players):
        p.order = i

    stats = {
        "games_played": 0,
        "wins_by_order": defaultdict(int), # Nueva estadística de Self-Play
        
        # Stats centradas en Jugador 0
        "buy_opportunities": defaultdict(int), 
        "buy_actions": defaultdict(int),
        "build_ops": 0,
        "build_acts": 0,
        "build_heatmap": defaultdict(int), # Para gráfico
        "buy_heatmap": defaultdict(int)    # Para gráfico
    }

    print(f"\nComenzando torneo Self-Play de {N_GAMES} partidas...")

    for i in range(N_GAMES):
        observation, info = env.reset()
        done = False
        
        while not done:
            current_player_idx = info["player_on_turn"]
            current_game_state = info["game_state"]
            action_mask = info["action_mask"]

            current_player = players[current_player_idx]
            
            # El agente Rotated ya maneja internamente la máscara si se le pasa
            action = current_player.action(current_game_state, action_mask=action_mask)

            # --- RECOLECCIÓN DE DATOS (Centrado en Jugador 0) ---
            if current_player_idx == 0:
                # Compras (Acción 41)
                pos = current_game_state[1][0]
                if action_mask[41]:
                    stats["buy_opportunities"][pos] += 1
                    if action == 41: 
                        stats["buy_actions"][pos] += 1
                        stats["buy_heatmap"][pos] += 1
                
                # Edificación (1-40)
                can_build = np.any(action_mask[1:41])
                if can_build:
                    stats["build_ops"] += 1
                    if 1 <= action <= 40:
                        stats["build_acts"] += 1
                        stats["build_heatmap"][action-1] += 1
            # ----------------------------------------------------

            observation, reward, Terminated, Truncated, info = env.step(action)
            
            if Terminated:
                winner_idx = env.unwrapped._check_for_winner()
                if winner_idx != -1:
                    stats["wins_by_order"][winner_idx] += 1
            
            done = Terminated or Truncated
        
        stats["games_played"] += 1
        if (i+1) % 100 == 0:
            print(f"Progreso: {i+1}/{N_GAMES}", end="\r")

    env.close()
    
    print_selfplay_stats(stats, env.unwrapped.board_names)
    
    if VISUALIZATION_AVAILABLE:
        print("\nGenerando gráficos de Self-Play (Jugador 0)...")
        plot_board_heatmap(stats["buy_heatmap"], f"Self-Play: Compras del Jugador 0 ({N_GAMES} partidas)")
        plot_board_heatmap(stats["build_heatmap"], f"Self-Play: Edificaciones del Jugador 0")