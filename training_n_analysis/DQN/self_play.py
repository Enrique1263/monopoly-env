import os
import numpy as np
from collections import Counter, defaultdict
import gymnasium as gym
import monopoly_env
from agents.dqn_player import DQNPlayer

# =========================
# CONFIGURACIÓN
# =========================
STATE_DIM = 200
ACTION_DIM = 43
NUM_EPISODES = 500  # Número de partidas de self-play
NUM_PLAYERS = 4
MODEL_PATH = "models/dqn/best_model.pt"

# =========================
# Cargar Agentes (4 clones DQN)
# =========================
players = [
    DQNPlayer(name="DQN_0", state_dim=STATE_DIM, action_dim=ACTION_DIM, num_players=NUM_PLAYERS, color=(0,255,0)),
    DQNPlayer(name="DQN_1", state_dim=STATE_DIM, action_dim=ACTION_DIM, num_players=NUM_PLAYERS, color=(255,0,0)),
    DQNPlayer(name="DQN_2", state_dim=STATE_DIM, action_dim=ACTION_DIM, num_players=NUM_PLAYERS, color=(0,0,255)),
    DQNPlayer(name="DQN_3", state_dim=STATE_DIM, action_dim=ACTION_DIM, num_players=NUM_PLAYERS, color=(255,255,0)),
]

# Orden fijo
for i, p in enumerate(players):
    p.order = i

# Cargar modelo entrenado en todos los clones
for p in players:
    p.load(MODEL_PATH)
    p.epsilon = 0.0  # Explorar nada, solo explotación

# =========================
# Crear entorno
# =========================
env = gym.make('MonopolyEnv-v0',
               players=players,
               render_mode=None,
               max_steps=1200)

# =========================
# Estadísticas completas
# =========================
stats = {
    "games_played": 0,
    "wins_by_order": defaultdict(int),
    
    "actions_taken": Counter(),
    "noop_forced": 0,
    "noop_strategic": 0,
    "noop_buy_ignored_details": defaultdict(int),
    "noop_strategic_details": defaultdict(set),

    "first_buy": {"taken":0, "ignored":0},
    "first_buy_ignored_details": defaultdict(set),

    "conflict_buy_vs_build": {"total":0, "chose_buy":0, "chose_build":0, "chose_other":0},
    "build_opportunities": {"total":0, "taken":0},
    "build_selection": {"cheapest":0, "most_expensive":0, "middle":0},

    "buy_opportunities": defaultdict(int),
    "buy_actions": defaultdict(int),
    "build_actions": defaultdict(int)
}

# =========================
# Función de reporte
# =========================
from collections import OrderedDict
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

def plot_board_heatmap(data_dict, title):
    if not VISUALIZATION_AVAILABLE:
        return
    grid = np.full((11,11), np.nan)
    # Bottom
    for i in range(11):
        grid[10,10-i] = data_dict.get(i,0) or np.nan
    # Left
    for i in range(11):
        grid[10-i,0] = data_dict.get(10+i,0) or np.nan
    # Top
    for i in range(11):
        grid[0,i] = data_dict.get(20+i,0) or np.nan
    # Right
    for i in range(10):
        grid[i,10] = data_dict.get(30+i,0) or np.nan

    plt.figure(figsize=(10,9))
    sns.heatmap(grid, annot=True, fmt=".0f", cmap="YlOrRd", cbar=True, square=True)
    plt.title(title)
    plt.show()

def print_selfplay_stats(stats, board_names):
    print("\n" + "="*80)
    print(f"        REPORTE COMPLETO SELF-PLAY ({stats['games_played']} partidas)")
    print("="*80)

    # Victorias por turno
    print("\n🏆 VICTORIAS POR TURNO")
    total_wins = sum(stats['wins_by_order'].values())
    for i in range(NUM_PLAYERS):
        wins = stats['wins_by_order'][i]
        pct = (wins / total_wins * 100) if total_wins > 0 else 0.0
        print(f"  Jugador {i}: {wins} victorias ({pct:.1f}%)")

    # Acciones totales
    print("\n💤 INACCIÓN (NO-OP)")
    total_noops = stats["noop_forced"] + stats["noop_strategic"]
    print(f"  Total No-Ops: {total_noops}")
    if total_noops > 0:
        print(f"   Forzadas: {stats['noop_forced']} ({stats['noop_forced']/total_noops*100:.1f}%)")
        print(f"   Estratégicas: {stats['noop_strategic']} ({stats['noop_strategic']/total_noops*100:.1f}%)")
        if stats["noop_buy_ignored_details"]:
            print("   Propiedades ignoradas estratégicamente:")
            for tile, count in sorted(stats["noop_buy_ignored_details"].items()):
                name = board_names.get(tile,f"Casilla {tile}")
                print(f"     {name}: {count} veces")

    # Primera oportunidad de compra
    first_ops = stats['first_buy']['taken'] + stats['first_buy']['ignored']
    if first_ops > 0:
        pct_taken = stats['first_buy']['taken']/first_ops*100
        print(f"\n🏁 PRIMERA OPORTUNIDAD DE COMPRA:")
        print(f"   Tomada: {stats['first_buy']['taken']} ({pct_taken:.1f}%)")
        print(f"   Ignorada: {stats['first_buy']['ignored']} ({100-pct_taken:.1f}%)")

    # Conflictos comprar vs edificar
    c_total = stats['conflict_buy_vs_build']['total']
    if c_total > 0:
        print(f"\n⚔️ CONFLICTO COMPRAR vs EDIFICAR ({c_total} turnos)")
        for k in ['chose_buy','chose_build','chose_other']:
            val = stats['conflict_buy_vs_build'][k]
            print(f"   {k}: {val} ({val/c_total*100:.1f}%)")

    # Edificación
    b_total = stats['build_opportunities']['total']
    if b_total > 0:
        taken = stats['build_opportunities']['taken']
        print(f"\n🏗️ EDIFICACIÓN: {taken}/{b_total} ({taken/b_total*100:.1f}%)")
        sel_total = sum(stats['build_selection'].values())
        if sel_total > 0:
            print("   Selección de propiedad:")
            for k,v in stats['build_selection'].items():
                print(f"    {k}: {v} ({v/sel_total*100:.1f}%)")

    # Detalle por propiedad
    print("\n💰 DETALLE COMPRAS POR PROPIEDAD")
    print(f"{'Propiedad':<30} | {'Oportunidades':<10} | {'Comprado':<10} | {'Tasa %'}")
    for tile, opp in sorted(stats["buy_opportunities"].items(), key=lambda x: x[0]):
        buys = stats["buy_actions"][tile]
        pct = buys/opp*100 if opp>0 else 0
        name = board_names.get(tile,f"Casilla {tile}")
        print(f"{name:<30} | {opp:<10} | {buys:<10} | {pct:.1f}%")

    print("\n🏗️ DETALLE EDIFICACIÓN POR PROPIEDAD")
    for tile, count in sorted(stats["build_actions"].items()):
        name = board_names.get(tile,f"Casilla {tile}")
        print(f"{name:<30} | {count}")

    if VISUALIZATION_AVAILABLE:
        print("\n📊 GENERANDO HEATMAPS...")
        plot_board_heatmap(stats["buy_actions"],"Compras Self-Play")
        plot_board_heatmap(stats["build_actions"],"Edificación Self-Play")

# =========================
# EJECUCIÓN DE PARTIDAS
# =========================
board_names = env.unwrapped.board_names

for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    first_buy_seen = [False]*NUM_PLAYERS

    while not done:
        idx = info["player_on_turn"]
        player = players[idx]
        state = info["game_state"]
        mask = info["action_mask"]

        action = player.action(state, mask)

        # --- RECOLECCIÓN DE DATOS DQN ---
        if isinstance(player,DQNPlayer):
            # Acciones
            stats["actions_taken"][action] +=1
            # No-ops
            if action==0:
                if np.sum(mask)==1:
                    stats["noop_forced"] +=1
                else:
                    stats["noop_strategic"] +=1
                    pos = state[1][idx]
                    if mask[41]:
                        stats["noop_buy_ignored_details"][pos] +=1
            # Primera compra
            if mask[41] and not first_buy_seen[idx]:
                first_buy_seen[idx] = True
                if action==41:
                    stats['first_buy']['taken'] +=1
                else:
                    stats['first_buy']['ignored'] +=1
                    pos = state[1][idx]
                    stats['first_buy_ignored_details'][pos].add(ep)
            # Conflicto comprar vs edificar
            can_buy = mask[41]
            can_build = np.any(mask[1:41])
            if can_buy and can_build:
                stats['conflict_buy_vs_build']['total'] +=1
                if action==41: stats['conflict_buy_vs_build']['chose_buy']+=1
                elif 1<=action<=40: stats['conflict_buy_vs_build']['chose_build']+=1
                else: stats['conflict_buy_vs_build']['chose_other']+=1
            # Edificación
            if can_build:
                stats['build_opportunities']['total'] +=1
                if 1<=action<=40:
                    stats['build_opportunities']['taken']+=1
                    stats['build_selection']['cheapest']+=0 # opcional: agregar lógica de costo
                    stats['build_actions'][action-1]+=1
            # Compras
            pos = state[1][idx]
            if mask[41]:
                stats['buy_opportunities'][pos]+=1
                if action==41:
                    stats['buy_actions'][pos]+=1

        # Step
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        # Victorias por orden
        if term:
            winner_idx = env.unwrapped._check_for_winner()
            if winner_idx != -1:
                stats["wins_by_order"][winner_idx] +=1

    stats["games_played"] +=1
    if (ep+1)%50==0:
        print(f"Progreso {ep+1}/{NUM_EPISODES}",end="\r")

env.close()
print_selfplay_stats(stats, board_names)
