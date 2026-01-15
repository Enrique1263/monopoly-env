import os
import torch
import numpy as np
from collections import Counter, defaultdict
from agents.dqn_player import DQNPlayer
from agents.Randy import Randy
import gymnasium as gym
import monopoly_env

# Intentamos importar librerías de visualización
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("AVISO: matplotlib/seaborn no instalados, no se generarán gráficos.")

# =========================
# Configuración
# =========================
STATE_DIM = 200
ACTION_DIM = 43
NUM_EPISODES = 1000
NUM_PLAYERS = 4

# =========================
# Crear agentes
# =========================
dqn_player = DQNPlayer(name="DQN_Hero", state_dim=STATE_DIM, action_dim=ACTION_DIM, num_players=NUM_PLAYERS, color=(0, 255, 0))
bot1 = Randy(name="Randy1")
bot2 = Randy(name="Randy2")
bot3 = Randy(name="Randy3")
players = [dqn_player, bot1, bot2, bot3]

# Mantener orden fijo
for i, player in enumerate(players):
    player.order = i

# =========================
# Cargar modelo entrenado
# =========================
model_path = "models/dqn/best_model.pt"
dqn_player.load(model_path)
dqn_player.epsilon = 0.0  # Explotar completamente

# =========================
# Crear entorno
# =========================
env = gym.make('MonopolyEnv-v0',
               players=players,
               render_mode=None,
               max_steps=1200)

# =========================
# Estadísticas extendidas
# =========================
stats = {
    "games_played": 0,
    "wins": 0,
    "actions_taken": Counter(),
    "noop_forced": 0,
    "noop_strategic": 0,
    "noop_buy_ignored_details": defaultdict(int),
    "noop_strategic_details": defaultdict(set),
    "buy_opportunities": defaultdict(int),
    "buy_actions": defaultdict(int),
    "build_actions": defaultdict(int),
    "first_buy": {"taken":0, "ignored":0},
    "first_buy_ignored_details": defaultdict(set),
    "conflict_buy_vs_build": {"total":0, "chose_buy":0, "chose_build":0, "chose_other":0},
    "build_opportunities": {"total":0, "taken":0},
    "build_selection": {"cheapest":0, "most_expensive":0, "middle":0}
}

# =========================
# Funciones de visualización
# =========================
def plot_board_heatmap(data_dict, title):
    if not VISUALIZATION_AVAILABLE:
        return
    grid = np.full((11,11), np.nan)
    # Bottom row
    for i in range(11):
        val = data_dict.get(i,0)
        grid[10,10-i] = val if val>0 else np.nan
    # Left col
    for i in range(11):
        val = data_dict.get(10+i,0)
        grid[10-i,0] = val if val>0 else np.nan
    # Top row
    for i in range(11):
        val = data_dict.get(20+i,0)
        grid[0,i] = val if val>0 else np.nan
    # Right col
    for i in range(10):
        val = data_dict.get(30+i,0)
        grid[i,10] = val if val>0 else np.nan
    plt.figure(figsize=(10,9))
    sns.heatmap(grid, annot=True, fmt=".0f", cmap="YlOrRd", square=True, cbar=True, linewidths=.5, linecolor='gray', xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.show()

def print_statistics(stats, board_names):
    print("\n" + "="*80)
    print(f"REPORTE COMPLETO DE DQN - {stats['games_played']} partidas")
    print("="*80)
    win_pct = (stats['wins']/stats['games_played'])*100
    print(f"\n🏆 Victorias: {stats['wins']} ({win_pct:.2f}%)")
    print(f"\n💤 Inacción (NO-OP) total: {stats['actions_taken'][0]}")
    print(f"  Forzadas: {stats['noop_forced']}, Estratégicas: {stats['noop_strategic']}")
    print(f"\n💰 Compras y primeras oportunidades: {stats['first_buy']}")
    print(f"\n⚔️ Conflicto comprar vs edificar: {stats['conflict_buy_vs_build']}")
    print(f"\n🏗️ Edificaciones: {stats['build_opportunities']}, selección: {stats['build_selection']}")
    print(f"\nDetalle por propiedad (compras):")
    for tile, opp in stats['buy_opportunities'].items():
        buys = stats['buy_actions'][tile]
        rate = (buys/opp*100) if opp>0 else 0
        print(f"  Casilla {tile}: Oportunidades={opp}, Compró={buys}, Tasa={rate:.1f}%")
    print(f"\nDetalle por propiedad (edificaciones):")
    for tile, count in stats['build_actions'].items():
        print(f"  Casilla {tile}: Construido {count} veces")
    if VISUALIZATION_AVAILABLE:
        plot_board_heatmap(stats["buy_actions"], "Mapa de Compras DQN")
        plot_board_heatmap(stats["build_actions"], "Mapa de Edificaciones DQN")

# =========================
# Simulación de partidas
# =========================
for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    agent_idx = 0
    game_vars = {"first_buy_opportunity_seen": False}
    while not done:
        current_idx = info["player_on_turn"]
        player = players[current_idx]
        valid_mask = info.get("action_mask", None)
        # Acción
        if isinstance(player, DQNPlayer):
            action = player.action(info["game_state"], valid_mask)
        else:
            action = player.action(info["game_state"])
        # Estadísticas DQN
        if current_idx == agent_idx:
            stats["actions_taken"][action] += 1
            # No-ops
            if action==0:
                if np.sum(valid_mask)==1: stats["noop_forced"]+=1
                else:
                    stats["noop_strategic"]+=1
                    pos = info["game_state"][1][agent_idx]
                    if valid_mask[41]:
                        stats["noop_buy_ignored_details"][pos]+=1
            # Primera compra
            if valid_mask[41] and not game_vars["first_buy_opportunity_seen"]:
                game_vars["first_buy_opportunity_seen"]=True
                if action==41: stats["first_buy"]["taken"]+=1
                else:
                    stats["first_buy"]["ignored"]+=1
                    pos = info["game_state"][1][agent_idx]
                    stats["first_buy_ignored_details"][pos].add(episode)
            # Conflicto comprar vs edificar
            can_buy = valid_mask[41]
            can_build = np.any(valid_mask[1:41])
            if can_buy and can_build:
                stats['conflict_buy_vs_build']['total'] += 1
                if action==41: stats['conflict_buy_vs_build']['chose_buy']+=1
                elif 1<=action<=40: stats['conflict_buy_vs_build']['chose_build']+=1
                else: stats['conflict_buy_vs_build']['chose_other']+=1
            # Edificaciones
            if can_build:
                stats['build_opportunities']['total']+=1
                if 1<=action<=40:
                    stats['build_opportunities']['taken']+=1
                    stats['build_actions'][action-1]+=1
            # Compras
            pos = info["game_state"][1][agent_idx]
            if valid_mask[41]:
                stats["buy_opportunities"][pos]+=1
                if action==41:
                    stats["buy_actions"][pos]+=1
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Ganador
        if terminated:
            money_list = info["game_state"][2]
            winner_idx = int(np.argmax(money_list))
            if winner_idx==agent_idx:
                stats["wins"]+=1
    stats["games_played"]+=1
    if (episode+1)%50==0:
        print(f"Progreso: {episode+1}/{NUM_EPISODES}")

# =========================
# Resultados
# =========================
print_statistics(stats, env.unwrapped.board_names)
env.close()
