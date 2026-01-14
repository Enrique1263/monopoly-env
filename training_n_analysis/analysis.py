import os
import sys
import importlib.util
import inspect
from typing import Type
import monopoly_env
from player import Player
import gymnasium as gym
import numpy as np
from collections import Counter, defaultdict

# Intentamos importar librerías de visualización
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("AVISO: 'matplotlib' y 'seaborn' no están instalados. No se generarán gráficos.")

# Configuración del Análisis
N_GAMES = 1000
# El nombre de la clase o del agente que queremos rastrear (para ponerlo primero)
TARGET_AGENT_ID = "PPO" 

def find_subclasses(path: str, cls: Type[Player]):
    """Find all subclasses of a given class in a given path."""
    player_classes = []
    sys.path.append(path) 

    for filename in os.listdir(path):
        if filename.endswith('.py'):
            mod_name = filename[:-3]
            mod_path = os.path.join(path, filename)
            spec = importlib.util.spec_from_file_location(mod_name, mod_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name, o in inspect.getmembers(module):
                if inspect.isclass(o) and issubclass(o, cls) and o is not cls:
                    player_classes.append(o)
    return player_classes

def fixed_order(players):
    """
    Ordena los jugadores para asegurar que el Agente PPO sea el Jugador 0.
    Esto es crucial para que las estadísticas se recojan sobre el agente correcto.
    """
    # Buscamos al agente principal por nombre de clase o nombre de instancia
    players.sort(key=lambda p: 0 if TARGET_AGENT_ID in p.__class__.__name__ or TARGET_AGENT_ID in p.name else 1)
    
    # Asignamos el atributo order
    for i, player in enumerate(players):
        player.order = i
    return players

def plot_board_heatmap(data_dict, title, label_fmt=".0f"):
    """
    Genera un mapa de calor con la forma del tablero de Monopoly (11x11).
    :param data_dict: Diccionario {indice_casilla: valor}
    :param title: Título del gráfico
    """
    if not VISUALIZATION_AVAILABLE:
        return

    # Crear matriz 11x11 llena de NaNs (para que el centro quede blanco/vacío)
    grid = np.full((11, 11), np.nan)
    
    # --- Mapeo de índices lineales (0-39) a coordenadas de matriz (row, col) ---
    # El tablero empieza en la esquina inferior derecha (10, 10) y va hacia la izquierda.
    
    # 1. Fila Inferior (Indices 0-10): De Der a Izq -> Grid[10, 10] a Grid[10, 0]
    for i in range(11):
        idx = i
        val = data_dict.get(idx, 0)
        grid[10, 10-i] = val if val > 0 else np.nan # Usar NaN para 0 ayuda a la visualización

    # 2. Columna Izquierda (Indices 10-20): De Abajo a Arriba -> Grid[10, 0] a Grid[0, 0]
    for i in range(11):
        idx = 10 + i
        val = data_dict.get(idx, 0)
        # Sobrescribimos la esquina (10) para asegurar continuidad visual
        grid[10-i, 0] = val if val > 0 else np.nan

    # 3. Fila Superior (Indices 20-30): De Izq a Der -> Grid[0, 0] a Grid[0, 10]
    for i in range(11):
        idx = 20 + i
        val = data_dict.get(idx, 0)
        grid[0, i] = val if val > 0 else np.nan

    # 4. Columna Derecha (Indices 30-39): De Arriba a Abajo -> Grid[0, 10] a Grid[9, 10]
    # (El índice 40 sería el 0, que ya está puesto)
    for i in range(10): # 0 a 9 steps
        idx = 30 + i
        val = data_dict.get(idx, 0)
        grid[i, 10] = val if val > 0 else np.nan

    # Configuración del gráfico
    plt.figure(figsize=(10, 9))
    ax = sns.heatmap(grid, annot=True, fmt=label_fmt, cmap="YlOrRd", 
                     cbar=True, square=True, linewidths=.5, linecolor='gray',
                     xticklabels=False, yticklabels=False)
    
    plt.title(title, fontsize=16)
    # Etiquetas de referencia en las esquinas
    plt.text(10.5, 10.5, "GO (0)", ha='center', va='center', fontsize=8, fontweight='bold')
    plt.text(0.5, 10.5, "Jail (10)", ha='center', va='center', fontsize=8, fontweight='bold')
    plt.text(0.5, 0.5, "Parking (20)", ha='center', va='center', fontsize=8, fontweight='bold')
    plt.text(10.5, 0.5, "GoToJail (30)", ha='center', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def print_statistics(stats, board_names):
    print("\n" + "="*80)
    print(f"       REPORTE DETALLADO DE ESTRATEGIA ({stats['games_played']} Partidas)")
    print("="*80)
    
    # --- VICTORIAS ---
    win_rate = (stats['wins'] / stats['games_played']) * 100
    print(f"\n🏆 RENDIMIENTO")
    print(f"  Victorias: {stats['wins']} de {stats['games_played']} ({win_rate:.2f}%)")

    # --- NO-OP ---
    total_noops = stats["actions_taken"][0]
    print(f"\n💤 INACCIÓN (NO-OP) - Total: {total_noops}")
    if total_noops > 0:
        pct_forced = (stats["noop_forced"] / total_noops) * 100
        pct_strategic = (stats["noop_strategic"] / total_noops) * 100
        print(f"  - Forzadas (Única opción): {stats['noop_forced']} ({pct_forced:.1f}%)")
        print(f"  - Estratégicas (Eligió esperar): {stats['noop_strategic']} ({pct_strategic:.1f}%)")
        
        # Detalle de COMPRAS ignoradas via NO-OP
        # stats["noop_buy_ignored_details"] es {tile_idx: count}
        if stats["noop_buy_ignored_details"]:
            print(f"\n  🔍 Propiedades ESPECÍFICAS que decidió NO comprar (teniendo opción y dinero):")
            # Ordenar de menor a mayor
            sorted_noop_buys = sorted(stats["noop_buy_ignored_details"].items(), key=lambda x: x[1])
            
            for tile_idx, count in sorted_noop_buys:
                prop_name = board_names.get(tile_idx, f"Casilla {tile_idx}")
                print(f"    - {prop_name:<30} -> Ignorada {count} veces")
        else:
            print("    * Nunca ignoró una compra disponible eligiendo No-Op.")

    # --- PRIMERA IMPRESIÓN ---
    print(f"\n🏁 PRIMERA OPORTUNIDAD DE COMPRA (En cada partida)")
    first_ops = stats['first_buy']['taken'] + stats['first_buy']['ignored']
    if first_ops > 0:
        pct_taken = (stats['first_buy']['taken'] / first_ops) * 100
        print(f"  De {first_ops} partidas donde cayó en una propiedad libre al inicio:")
        print(f"  - Compró inmediatamente: {stats['first_buy']['taken']} ({pct_taken:.1f}%)")
        print(f"  - La ignoró: {stats['first_buy']['ignored']} ({100-pct_taken:.1f}%)")
        
        # Detalle de qué ignoró
        if stats['first_buy']['ignored'] > 0:
            print(f"\n  🔍 Propiedades ignoradas como PRIMERA compra (Orden: Menor a Mayor):")
            # Ordenar de menor a mayor por conteo
            sorted_ignored = sorted(stats["first_buy_ignored_details"].items(), key=lambda x: len(x[1]))
            for tile_idx, games_set in sorted_ignored:
                count = len(games_set)
                prop_name = board_names.get(tile_idx, f"Casilla {tile_idx}")
                print(f"    - {prop_name:<30} -> Ignorada en {count} partidas")
    
    # --- CONFLICTO COMPRAR vs EDIFICAR ---
    print(f"\n⚔️ DILEMA: ¿COMPRAR O EDIFICAR?")
    conflicts = stats['conflict_buy_vs_build']['total']
    if conflicts > 0:
        c_buy = stats['conflict_buy_vs_build']['chose_buy']
        c_build = stats['conflict_buy_vs_build']['chose_build']
        c_pass = stats['conflict_buy_vs_build']['chose_other']
        print(f"  En {conflicts} turnos donde podía AMBOS (comprar actual o edificar propia):")
        print(f"  - Priorizó COMPRAR (Expansión): {c_buy} ({c_buy/conflicts*100:.1f}%)")
        print(f"  - Priorizó EDIFICAR (Consolidación): {c_build} ({c_build/conflicts*100:.1f}%)")
        print(f"  - No hizo nada: {c_pass} ({c_pass/conflicts*100:.1f}%)")
    else:
        print("  Nunca se dio la situación de poder elegir entre comprar y edificar.")

    # --- LÓGICA DE EDIFICACIÓN ---
    print(f"\n🏗️ LÓGICA DE EDIFICACIÓN")
    build_ops = stats['build_opportunities']['total']
    if build_ops > 0:
        b_taken = stats['build_opportunities']['taken']
        print(f"  Agresividad Constructora: Edificó en {b_taken} de {build_ops} oportunidades ({b_taken/build_ops*100:.1f}%)")
        
        sel_cheapest = stats['build_selection']['cheapest']
        sel_expensive = stats['build_selection']['most_expensive']
        sel_middle = stats['build_selection']['middle']
        total_sel = sel_cheapest + sel_expensive + sel_middle
        
        if total_sel > 0:
            print(f"  Cuando tuvo varias opciones para edificar, eligió:")
            print(f"  - La más BARATA: {sel_cheapest} ({sel_cheapest/total_sel*100:.1f}%)")
            print(f"  - La más CARA: {sel_expensive} ({sel_expensive/total_sel*100:.1f}%)")
            print(f"  - Intermedia: {sel_middle} ({sel_middle/total_sel*100:.1f}%)")
    else:
        print("  Nunca tuvo oportunidad de edificar (¿No completó monopolios?).")

    # --- DETALLE POR PROPIEDAD (ORDENADO POR TASA) ---
    print(f"\n💰 DETALLE DE COMPRAS POR PROPIEDAD (Orden: Menor a Mayor Tasa)")
    print(f"  {'Propiedad':<30} | {'Oportunidades':<13} | {'Comprado':<10} | {'Tasa %'}")
    print("-" * 65)
    
    # Preparar datos para ordenar
    buy_data = []
    for tile, opps in stats["buy_opportunities"].items():
        buys = stats["buy_actions"][tile]
        rate = (buys / opps * 100) if opps > 0 else 0.0
        buy_data.append((tile, opps, buys, rate))
    
    # Ordenar por tasa (rate) ascendente
    buy_data.sort(key=lambda x: x[3])
    
    for tile, opps, buys, rate in buy_data:
        name = board_names.get(tile, f"Casilla {tile}")
        marker = "🔥" if rate > 90 else ("❄️" if rate < 10 else "")
        print(f"  {name:<30} | {opps:<13} | {buys:<10} | {rate:.1f}% {marker}")

    # --- DETALLE EDIFICACIÓN (ORDENADO POR CANTIDAD) ---
    print(f"\n🏗️ DETALLE DE EDIFICACIÓN POR PROPIEDAD (Orden: Menor a Mayor)")
    print(f"  {'Propiedad':<30} | {'Veces Edificado'}")
    print("-" * 50)
    
    # Ordenar por cantidad ascendente
    sorted_builds = sorted(stats["build_actions"].items(), key=lambda x: x[1])
    
    if not sorted_builds:
        print("  Sin datos.")
    else:
        for tile, count in sorted_builds:
            name = board_names.get(tile, f"Casilla {tile}")
            print(f"  {name:<30} | {count}")

    print("="*80)
    
    # --- GENERAR GRÁFICOS VISUALES ---
    if VISUALIZATION_AVAILABLE:
        print("\n📊 Generando Mapas de Calor del Tablero...")
        plot_board_heatmap(stats["buy_actions"], f"Mapa de Calor: Frecuencia de COMPRAS (Total Partidas: {stats['games_played']})")
        plot_board_heatmap(stats["build_actions"], f"Mapa de Calor: Frecuencia de EDIFICACIÓN (Total Partidas: {stats['games_played']})")

if __name__ == '__main__':
    # 1. Cargar Clases dinámicamente
    players_classes = find_subclasses('agents', Player)
    
    if not players_classes:
        print("Error: No se encontraron agentes en la carpeta 'agents/'.")
        sys.exit()

    # 2. Instanciar todos los agentes encontrados
    players = [clazz() for clazz in players_classes]
    
    print(f"Agentes cargados: {[p.name for p in players]}")

    # 3. Fijar orden (PPO siempre índice 0 para análisis)
    players = fixed_order(players)
    
    target_agent = players[0]
    print(f"--> Analizando estadísticas de: {target_agent.name} (Clase: {target_agent.__class__.__name__})")

    # 4. Crear Entorno
    env = gym.make('MonopolyEnv-v0', 
                   players=players, 
                   render_mode=None, 
                   max_steps=2000, 
                   board_names_path='cards/board_names.txt', 
                   community_chest_path='cards/community_chest.txt', 
                   chance_path='cards/chance.txt', 
                   hard_rules=False, 
                   image_path='cards/monopoly.png')

    # Estructuras de estadísticas extendidas
    stats = {
        "games_played": 0,
        "wins": 0,
        "actions_taken": Counter(), 
        "noop_forced": 0, "noop_strategic": 0,
        # NUEVO: Detalle de qué propiedad ignoró comprar (key: tile_idx, value: count)
        "noop_buy_ignored_details": defaultdict(int),
        "noop_strategic_details": defaultdict(set), # Mantenemos el antiguo para contexto general
        
        "buy_opportunities": defaultdict(int), "buy_actions": defaultdict(int),
        "build_actions": defaultdict(int),
        
        "first_buy": {"taken": 0, "ignored": 0},
        "first_buy_ignored_details": defaultdict(set),
        
        "conflict_buy_vs_build": {"total": 0, "chose_buy": 0, "chose_build": 0, "chose_other": 0},
        "build_opportunities": {"total": 0, "taken": 0},
        "build_selection": {"cheapest": 0, "most_expensive": 0, "middle": 0}
    }

    print(f"\nComenzando análisis de {N_GAMES} partidas...")

    for i in range(N_GAMES):
        observation, info = env.reset()
        done = False
        
        agent_idx = 0
        game_vars = {"first_buy_opportunity_seen": False}
        
        while not done:
            current_player_idx = info["player_on_turn"]
            current_game_state = info["game_state"]
            action_mask = info["action_mask"]

            # Ejecutar acción
            current_player = players[current_player_idx]
            try:
                action = current_player.action(current_game_state, action_mask=action_mask)
            except TypeError:
                action = current_player.action(current_game_state)

            # --- RECOLECCIÓN DE DATOS (Agente Objetivo) ---
            if current_player_idx == agent_idx:
                stats["actions_taken"][action] += 1
                
                # A. Análisis No-Op
                if action == 0:
                    if np.sum(action_mask) == 1: 
                        stats["noop_forced"] += 1
                    else: 
                        stats["noop_strategic"] += 1
                        
                        # --- NUEVO: ¿Ignoró comprar? ---
                        if action_mask[41]:
                            # Estaba habilitado comprar, pero eligió 0
                            pos = current_game_state[1][agent_idx]
                            stats["noop_buy_ignored_details"][pos] += 1
                        # -------------------------------

                        # Mantenemos el detalle general
                        available_indices = np.where(action_mask)[0]
                        other_options = tuple([idx for idx in available_indices if idx != 0])
                        stats["noop_strategic_details"][other_options].add(i)
                
                # B. Primera Oportunidad de Compra
                if action_mask[41] and not game_vars["first_buy_opportunity_seen"]:
                    game_vars["first_buy_opportunity_seen"] = True
                    if action == 41: 
                        stats["first_buy"]["taken"] += 1
                    else: 
                        stats["first_buy"]["ignored"] += 1
                        pos = current_game_state[1][agent_idx]
                        stats["first_buy_ignored_details"][pos].add(i)

                # C. Conflicto: Comprar vs Edificar
                can_buy = action_mask[41]
                can_build = np.any(action_mask[1:41])
                
                if can_buy and can_build:
                    stats['conflict_buy_vs_build']['total'] += 1
                    if action == 41: stats['conflict_buy_vs_build']['chose_buy'] += 1
                    elif 1 <= action <= 40: stats['conflict_buy_vs_build']['chose_build'] += 1
                    else: stats['conflict_buy_vs_build']['chose_other'] += 1

                # D. Análisis de Edificación
                if can_build:
                    stats['build_opportunities']['total'] += 1
                    if 1 <= action <= 40:
                        stats['build_opportunities']['taken'] += 1
                        
                        valid_build_idxs = np.where(action_mask[1:41])[0] 
                        costs = []
                        for b_idx in valid_build_idxs:
                            c = current_game_state[4][b_idx // 10]
                            costs.append(c)
                        
                        chosen_prop_idx = action - 1
                        chosen_cost = current_game_state[4][chosen_prop_idx // 10]
                        
                        min_c = min(costs)
                        max_c = max(costs)
                        
                        if len(costs) > 1 and min_c != max_c:
                            if chosen_cost == min_c: stats['build_selection']['cheapest'] += 1
                            elif chosen_cost == max_c: stats['build_selection']['most_expensive'] += 1
                            else: stats['build_selection']['middle'] += 1

                # E. Análisis de Compras por Casilla
                pos = current_game_state[1][agent_idx]
                if action_mask[41]:
                    stats["buy_opportunities"][pos] += 1
                    if action == 41: stats["buy_actions"][pos] += 1
                
                # F. Conteo Edificación General
                if 1 <= action <= 40:
                    stats["build_actions"][action - 1] += 1
            # ------------------------------------------------------

            observation, reward, Terminated, Truncated, info = env.step(action)
            
            # Checkear Ganador
            if Terminated:
                winner_idx = env.unwrapped._check_for_winner()
                if winner_idx == agent_idx:
                    stats["wins"] += 1
            
            done = Terminated or Truncated
        
        stats["games_played"] += 1
        if (i+1) % 100 == 0:
            print(f"Progreso: {i+1}/{N_GAMES}", end="\r")

    print("\n--- Análisis Completado ---")
    env.close()
    
    # Imprimir reporte
    print_statistics(stats, env.unwrapped.board_names)