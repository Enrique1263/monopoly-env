import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
import pygame
from typing import List, Dict, Any

# --- Importamos las clases reales ---
from monopoly_env.board.board import Board
from player import Player
# --- Fin de Imports ---


# Window dimensions
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800

# Board dimensions
BOARD_WIDTH = 800
BOARD_HEIGHT = 800

# Cells in the board
NUM_CELLS = 10 # 10 celdas por lado
NUM_BOARD_TILES = 40 # Número total de casillas en Monopoly


class MonopolyEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, players: List[Player], max_steps=1000, render_mode='human', board_names_path='cards/board_names.txt', community_chest_path='cards/community_chest.txt', chance_path='cards.txt', hard_rules=False, image_path=None):
        
        self.num_players = len(players)
        self.hard_rules = hard_rules
        # --- Action Space (Espacio de Acciones) ---
        # 0: No-op / Terminar turno (o pagar fianza si está en la cárcel)
        # 1-40: Intentar Edificar en la casilla (index 0-39)
        # 41: Intentar Comprar la propiedad (si está en una casilla comprable)
        # 42: Usar carta "Salir de la Cárcel"
        self.action_space = spaces.Discrete(43)

        # --- Observation Space (Espacio de Observación) ---        
        # Tamaño del vector de observación:
        # 40 (PROPERTIES: dueño de cada una)
        # 40 (BUILDINGS: nivel de edificación)
        # num_players (POSITIONS: posición de cada jugador)
        # num_players (MONEY: dinero de cada jugador, normalizado)
        # num_players (JAIL_TURNS: turnos en la cárcel)
        # num_players (GET_OUT_OF_JAIL_CARDS: cuántas cartas tiene cada uno)
        self.obs_vector_size = 40 + 40 + (self.num_players * 4)
        
        # Definimos los límites (low/high) del Box.
        low = np.full(self.obs_vector_size, -2.0, dtype=np.float32)
        high = np.full(self.obs_vector_size, 1.0, dtype=np.float32)
        
        # Ajustamos los límites para partes específicas del vector
        start_idx = 0
        # PROPERTIES (0-39): -2 (no edificable) -1 (sin dueño) a num_players-1 (dueño)
        high[start_idx : start_idx + 40] = self.num_players - 1
        start_idx += 40
        # BUILDINGS (40-79): -2 (no edificable) a 5 (hotel)
        low[start_idx : start_idx + 40] = -2
        high[start_idx : start_idx + 40] = 5
        start_idx += 40
        # POSITIONS (80-..): 0 a 39
        low[start_idx : start_idx + self.num_players] = 0
        high[start_idx : start_idx + self.num_players] = 39
        start_idx += self.num_players
        # MONEY (..-..): 0.0 a 1.0 (normalizado)
        low[start_idx : start_idx + self.num_players] = 0
        high[start_idx : start_idx + self.num_players] = 20000
        start_idx += self.num_players
        # JAIL_TURNS (..-..): 0 a 3
        low[start_idx : start_idx + self.num_players] = 0
        high[start_idx : start_idx + self.num_players] = 3
        start_idx += self.num_players
        # GET_OUT_OF_JAIL_CARDS (..-..): 0 a N (pongamos 4)
        low[start_idx : start_idx + self.num_players] = 0
        high[start_idx : start_idx + self.num_players] = 4
        start_idx += self.num_players
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.board_names_path = board_names_path
        self.community_chest_path = community_chest_path
        self.chance_path = chance_path
        # --- Estado del Juego (self.game_state) ---
        self.game_state = [] 
        self.PROPERTIES = 0 # Quién posee
        self.POSITIONS = 1  # Posición
        self.MONEY = 2      # Dinero (sin normalizar)
        self.PRICES = 3     # Precios (estático)
        self.EDIFICATE = 4  # Coste edificar (estático)
        self.BUILDINGS = 5  # Nivel edificios
        self.JAIL_TURNS = 6 # Turnos cárcel
        self.BANKRUPT = 7   # Bancarrota
        self.GET_OUT_OF_JAIL = 8 # Cartas cárcel
        self.PROPERTY_GROUPS = {
            1: {"group": "brown", "tiles": [1, 3]},
            3: {"group": "brown", "tiles": [1, 3]},
            
            6: {"group": "light_blue", "tiles": [6, 8, 9]},
            8: {"group": "light_blue", "tiles": [6, 8, 9]},
            9: {"group": "light_blue", "tiles": [6, 8, 9]},
            
            11: {"group": "pink", "tiles": [11, 13, 14]},
            13: {"group": "pink", "tiles": [11, 13, 14]},
            14: {"group": "pink", "tiles": [11, 13, 14]},
            
            16: {"group": "orange", "tiles": [16, 18, 19]},
            18: {"group": "orange", "tiles": [16, 18, 19]},
            19: {"group": "orange", "tiles": [16, 18, 19]},
            
            21: {"group": "red", "tiles": [21, 23, 24]},
            23: {"group": "red", "tiles": [21, 23, 24]},
            24: {"group": "red", "tiles": [21, 23, 24]},
            
            26: {"group": "yellow", "tiles": [26, 27, 29]},
            27: {"group": "yellow", "tiles": [26, 27, 29]},
            29: {"group": "yellow", "tiles": [26, 27, 29]},
            
            31: {"group": "green", "tiles": [31, 32, 34]},
            32: {"group": "green", "tiles": [31, 32, 34]},
            34: {"group": "green", "tiles": [31, 32, 34]},
            
            37: {"group": "dark_blue", "tiles": [37, 39]},
            39: {"group": "dark_blue", "tiles": [37, 39]},
        }
        
        self.initial_money = 1500
        self.net_worths = [self.initial_money] * self.num_players

        self.players = players
        self.board_names = self.load_board_names()
        self.community_chest_cards = self.load_cards(self.community_chest_path)
        self.chance_cards = self.load_cards(self.chance_path)
        
        self.shuffled_chance_cards = []
        self.shuffled_community_chest_cards = []
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.steps_done = 0
        self.player_on_turn = 0
        self.on_double = False
        self.turns_on_double = 0
        self.dices = 0

        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Monopoly")
            self.board = Board(SCREEN_WIDTH, SCREEN_HEIGHT, BOARD_WIDTH, BOARD_HEIGHT, NUM_CELLS, image_path)

    def load_board_names(self):
        board_names = {}
        try:
            with open(self.board_names_path, 'r') as file:
                for line in file:
                    position, name = line.strip().split(':')
                    board_names[int(position)] = name
        except FileNotFoundError:
            print(f"Warning: Board names file not found at {self.board_names_path}. Using defaults.")
            board_names = {i: f"Casilla {i}" for i in range(NUM_BOARD_TILES)}
        return board_names
    
    def load_cards(self, file_path):
        cards = []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split('|')
                    card_text = parts[0]
                    money = int(parts[1])
                    affects_others = parts[2].lower() == 'true'
                    tile = int(parts[3])
                    cards.append((card_text, money, affects_others, tile))
        except FileNotFoundError:
             print(f"Warning: Cards file not found at {file_path}. Using defaults.")
             cards = [("Ve a la cárcel", 0, False, 10)]
        return np.array(cards, dtype=object)

    def _calculate_net_worth(self, player_id):
        """
        NUEVA FUNCIÓN: Calcula el Patrimonio Neto (Net Worth) de un jugador.
        Patrimonio = Dinero + Valor Propiedades + Valor Edificios
        """
        if self.game_state[self.BANKRUPT][player_id]:
            return 0
        
        net_worth = self.game_state[self.MONEY][player_id]
        
        for i in range(NUM_BOARD_TILES):
            if self.game_state[self.PROPERTIES][i] == player_id:
                # Suma el precio de compra de la propiedad
                try:
                    # El precio se almacena como negativo (ej. -60)
                    purchase_price = -self.game_state[self.PRICES][i][-1]
                    net_worth += purchase_price
                except (IndexError, TypeError):
                    pass # No es una propiedad comprable (ej. Cárcel, Go)
                
                # Suma el coste de los edificios
                build_level = self.game_state[self.BUILDINGS][i]
                if build_level > 0:
                    coste_edificio = self.game_state[self.EDIFICATE][i // 10]
                    net_worth += build_level * coste_edificio
        
        return net_worth
    
    def _has_monopoly(self, player_id, group_tiles):
        """Comprueba si un jugador tiene todas las propiedades de un grupo."""
        for tile_index in group_tiles:
            if self.game_state[self.PROPERTIES][tile_index] != player_id:
                return False
        return True

    def _get_min_build_level(self, group_tiles):
        """Encuentra el nivel de construcción más bajo en un grupo (para construir uniformemente)."""
        min_level = 6 # Más alto que el hotel (5)
        for tile_index in group_tiles:
            level = self.game_state[self.BUILDINGS][tile_index]
            # Nos aseguramos de que sea un nivel de construcción (0-5)
            if 0 <= level <= 5:
                min_level = min(min_level, level)
        return min_level

    def _get_obs(self):
        """
        NUEVA FUNCIÓN: Convierte self.game_state en el vector de observación aplanado
        para la red neuronal.
        """
        obs = np.zeros(self.obs_vector_size, dtype=np.float32)
        
        start_idx = 0
        obs[start_idx : start_idx + 40] = self.game_state[self.PROPERTIES]
        start_idx += 40
        obs[start_idx : start_idx + 40] = self.game_state[self.BUILDINGS]
        start_idx += 40
        obs[start_idx : start_idx + self.num_players] = self.game_state[self.POSITIONS]
        start_idx += self.num_players
         
        obs[start_idx : start_idx + self.num_players] = self.game_state[self.MONEY]
        start_idx += self.num_players
        
        obs[start_idx : start_idx + self.num_players] = self.game_state[self.JAIL_TURNS]
        start_idx += self.num_players
        obs[start_idx : start_idx + self.num_players] = self.game_state[self.GET_OUT_OF_JAIL]
        start_idx += self.num_players
        
        return obs

    def _get_action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        player = self.player_on_turn
        pos = self.game_state[self.POSITIONS][player]
        
        # 0: No-op (siempre válido)
        mask[0] = True
        
        # 1-40: Edificar
        checked_groups = set() # Para no comprobar el monopolio 3 veces
        for i in range(NUM_BOARD_TILES):
            # Comprobación 1: ¿Es una casilla edificable?
            if i not in self.PROPERTY_GROUPS:
                continue
            
            # Comprobación 2: ¿La posee el jugador?
            if self.game_state[self.PROPERTIES][i] == player:
                group_info = self.PROPERTY_GROUPS[i]
                group_name = group_info["group"]
                
                if self.hard_rules:
                    # Solo procesamos cada grupo de color una vez
                    if group_name not in checked_groups:
                        checked_groups.add(group_name)
                        group_tiles = group_info["tiles"]
                        
                        # Comprobación 3: ¿Tiene el monopolio?
                        if self._has_monopoly(player, group_tiles):
                            # Comprobación 4: Regla de construir uniformemente
                            min_level = self._get_min_build_level(group_tiles)
                            
                            # Si el nivel mínimo es 5 (hotel), no se puede construir más en este grupo
                            if min_level == 5:
                                continue
                                
                            # Si tiene monopolio, puede construir en *todas* las casillas 
                            # que tengan el nivel de construcción mínimo
                            for tile_index in group_tiles:
                                if self.game_state[self.BUILDINGS][tile_index] == min_level:
                                    # Comprobación 5: ¿Tiene dinero?
                                    cost = self.game_state[self.EDIFICATE][tile_index // 10]
                                    if self.game_state[self.MONEY][player] >= cost:
                                        mask[1 + tile_index] = 1 # ¡Acción Válida!
                else:
                    # Modo sencillo: Solo comprobar si puede edificar en esta casilla
                    if self.game_state[self.BUILDINGS][i] < 5:
                        cost = self.game_state[self.EDIFICATE][i // 10]
                        if self.game_state[self.MONEY][player] >= cost:
                            mask[1 + i] = True
        
        # 41: Comprar (re-indexado desde 81)
        owner = self.game_state[self.PROPERTIES][pos]
        if owner == -1: # -1 = sin dueño
            try:
                # ¡¡CORREGIDO!!: Usamos el último valor para el precio
                price = -self.game_state[self.PRICES][pos][-1]
                if self.game_state[self.MONEY][player] >= price:
                    mask[41] = True
            except (IndexError, TypeError):
                pass # No es una propiedad comprable

        # 42: Usar carta cárcel (re-indexado desde 82)
        if self.game_state[self.JAIL_TURNS][player] > 0 and \
           self.game_state[self.GET_OUT_OF_JAIL][player] > 0:
            mask[42] = True
            
        # Lógica de Pagar Fianza (sobrescribe No-op)
        if self.game_state[self.JAIL_TURNS][player] > 0 and \
           self.game_state[self.MONEY][player] >= 50:
            mask[0] = True # Acción 0 también significa "Pagar Fianza"

        return mask
        
    def _get_info(self):
        """
        NUEVA FUNCIÓN: Devuelve el diccionario de información.
        Añadimos 'game_state' para los bots.
        """
        return {
            "action_mask": self._get_action_mask(),
            "player_on_turn": self.player_on_turn,
            "game_state": self.game_state # Para los bots que usan la obs original
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            random.seed(seed)
            np.random.seed(seed)
            
        # Tu lógica de reseteo
        self.game_state = [
            # 0: PROPERTIES
            [-2]+[-1]+[-2]+[-1]+[-2]+[-1]*2+[-2]+[-1]*2+[-2]+[-1]*6+[-2]+[-1]*2+[-2]+[-1]+[-2]+[-1]*7+[-2]+[-1]*2+[-2]+[-1]*2+[-2]+[-1]+[-2]+[-1],
            # 1: POSITIONS
            [0] * self.num_players,
            # 2: MONEY
            [self.initial_money] * self.num_players,
            # 3: PRICES
            [[200], [-2, -10, -30, -90, -160, -250, -60], [0], [-4, -20, -60, -180, -320, -450, -60], [-200], [-25, -50, -100, -200, -200],
             [-6, -30, -90, -270, -400, -550, -100], [0], [-6, -30, -90, -270, -400, -550, -100], 
             [-8, -40, -100, -300, -450, -600, -120], [0], [-10, -50, -150, -450, -625, -750, -140], [-4, -10, -150], [-10, -50, -150, -450, -625, -750, -140],
             [-12, -60, -180, -500, -700, -900, -160], [-25, -50, -100, -200, -200], [-14, -70, -200, -550, -750, -950, -180], [0], [-14, -70, -200, -550, -750, -950, -180],
             [-16, -80, -220, -600, -800, -1000, -200], [0], [-18, -90, -250, -700, -875, -1050, -220], [0], [-18, -90, -250, -700, -875, -1050, -220],
             [-20, -100, -300, -750, -925, -1100, -240], [-25, -50, -100, -200, -200], [-22, -110, -330, -800, -975, -1150, -260], [-22, -110, -330, -800, -975, -1150, -260], [-4, -10, -150],
             [-24, -120, -360, -850, -1025, -1200, -280], [0], [-26, -130, -390, -900, -1100, -1275, -300], [-26, -130, -390, -900, -1100, -1275, -300], [0],
             [-28, -150, -450, -1000, -1200, -1400, -320], [-25, -50, -100, -200, -200], [0], [-35, -175, -500, -1100, -1300, -1500, -350], [-100], [-50, -200, -600, -1400, -1700, -2000, -400]],
            # 4: EDIFICATE
            [50, 100, 150, 200],
            # 5: BUILDINGS
            [-2 if i in {0, 2, 4, 7, 10, 17, 20, 22, 30, 33, 36, 38} else -1 for i in range(40)],
            # 6: JAIL_TURNS
            [0] * self.num_players,
            # 7: BANKRUPT
            [False] * self.num_players,
            # 8: GET_OUT_OF_JAIL
            [0] * self.num_players
        ]
        
        ### NUEVO: Barajamos las cartas ###
        self.shuffled_chance_cards = list(self.chance_cards)
        random.shuffle(self.shuffled_chance_cards)
        self.shuffled_community_chest_cards = list(self.community_chest_cards)
        random.shuffle(self.shuffled_community_chest_cards)
        
        self.net_worths = [self._calculate_net_worth(i) for i in range(self.num_players)]
        
        self.player_on_turn = 0
        self.steps_done = 0
        self.on_double = False
        self.turns_on_double = 0
        
        # Tirada inicial
        dices, dobles = self.roll_dice()
        self.on_double = dobles
        self.game_state[self.POSITIONS][self.player_on_turn] = (self.game_state[self.POSITIONS][self.player_on_turn] + dices) % 40
        self.dices = dices
        
        self._resolve_land_on_tile(self.player_on_turn)
        
        print(f'{self.players[self.player_on_turn].name} landed on {self.board_names[self.game_state[self.POSITIONS][self.player_on_turn]]}')
        
        return self._get_obs(), self._get_info()
    
    def roll_dice(self):
        dice1 = random.randint(1, 6)
        dice2 = random.randint(1, 6)
        return dice1 + dice2, dice1 == dice2
    
    def reset_properties(self, player):
        for prop in range(NUM_BOARD_TILES):
            if self.game_state[self.PROPERTIES][prop] == player:
                # Resetea al estado inicial (-1 sin dueño, -2 no comprable)
                initial_build_state = -2 if prop in {0, 2, 4, 7, 10, 17, 20, 22, 30, 33, 36, 38} else -1
                self.game_state[self.BUILDINGS][prop] = initial_build_state
                self.game_state[self.PROPERTIES][prop] = -1 # Asumimos -1 para "sin dueño"

    def render(self, mode='human'):
        if mode == 'human':
            self.board.draw(self.screen)
            # Pasamos game_state (la lista) a draw_players
            self.board.draw_players(self.screen, self.game_state, self.player_on_turn, self.dices, self.players)
            pygame.display.flip()
            time.sleep(1)

    def _check_for_winner(self):
        players_active = [i for i, bankrupt in enumerate(self.game_state[self.BANKRUPT]) if not bankrupt]
        if len(players_active) == 1:
            return players_active[0]
        return -1

    ### NUEVO: Función para comprobar bancarrota (evita duplicar código) ###
    def _check_bankruptcy(self, player_id):
        if self.game_state[self.MONEY][player_id] <= 0 and not self.game_state[self.BANKRUPT][player_id]:
            print(f"{self.players[player_id].name} entra en bancarrota.")
            self.game_state[self.MONEY][player_id] = 0
            self.game_state[self.BANKRUPT][player_id] = True
            self.reset_properties(player_id)
            self.players[player_id].color = (0, 0, 0)
            self.on_double = False
            return True
        return False

    ### NUEVO: Funciones para coger y ejecutar cartas ###
    def _draw_chance_card(self, player_id):
        if not self.shuffled_chance_cards:
            print("  > Barajando de nuevo el mazo de Suerte.")
            self.shuffled_chance_cards = list(self.chance_cards)
            random.shuffle(self.shuffled_chance_cards)
        
        card = self.shuffled_chance_cards.pop(0)
        self._execute_card_action(player_id, card)

    def _draw_community_chest_card(self, player_id):
        if not self.shuffled_community_chest_cards:
            print("  > Barajando de nuevo el mazo de Caja de Comunidad.")
            self.shuffled_community_chest_cards = list(self.community_chest_cards)
            random.shuffle(self.shuffled_community_chest_cards)
        
        card = self.shuffled_community_chest_cards.pop(0)
        self._execute_card_action(player_id, card)

    def _execute_card_action(self, player_id, card):
        text, money, affects_others, tile = card
        text_lower = text.lower()
        print(f"  > Carta: {text}")

        # 1. "Get out of Jail Free"
        if "get out of jail free" in text_lower:
            self.game_state[self.GET_OUT_OF_JAIL][player_id] += 1
            return # No move, no money change

        # 2. Movement cards
        new_pos = -1
        current_pos = self.game_state[self.POSITIONS][player_id]
        
        if "go to jail" in text_lower:
            self.game_state[self.POSITIONS][player_id] = 10
            self.game_state[self.JAIL_TURNS][player_id] = 3
            self.on_double = False # Pierde los dobles si los tuviera
            return # No further action, don't pass Go

        elif "go back three spaces" in text_lower:
            new_pos = (current_pos - 3 + NUM_BOARD_TILES) % NUM_BOARD_TILES
        
        elif "go back to old kent road" in text_lower:
            new_pos = 1 # Old Kent Road (Asumo que es la casilla 1)
        
        elif tile != 0:
            new_pos = tile
            # Check for passing Go (Solo si la carta lo especifica)
            if new_pos < current_pos and "collect $200" in text_lower:
                print(f"  > {self.players[player_id].name} cobra $200 por pasar por GO.")
                self.game_state[self.MONEY][player_id] += 200

        # If we moved, update position and resolve landing
        if new_pos != -1:
            self.game_state[self.POSITIONS][player_id] = new_pos
            print(f"  > {self.players[player_id].name} se mueve a {self.board_names[new_pos]}.")
            # ¡¡IMPORTANTE!! Resolver la casilla donde cae (pagar alquiler, etc.)
            self._resolve_land_on_tile(player_id)
            return

        # 3. Repair cards
        cost = 0
        if "repairs" in text_lower or "pay $20 for each house" in text_lower:
            num_houses = 0
            num_hotels = 0
            for i in range(NUM_BOARD_TILES):
                if self.game_state[self.PROPERTIES][i] == player_id:
                    build_level = self.game_state[self.BUILDINGS][i]
                    if build_level == 5:
                        num_hotels += 1
                    elif 0 < build_level < 5:
                        num_houses += build_level
            
            if "general repairs" in text_lower: # 25 per house, 100 per hotel
                cost = (num_houses * 25) + (num_hotels * 100)
            elif "street repairs" in text_lower: # 40 per house, 115 per hotel
                cost = (num_houses * 40) + (num_hotels * 115)
            elif "pay $20 for each house" in text_lower: # 20 per house, 100 per hotel
                cost = (num_houses * 20) + (num_hotels * 100)
            
            if cost > 0:
                print(f"  > {self.players[player_id].name} paga ${cost} en reparaciones.")
                self.game_state[self.MONEY][player_id] -= cost

        # 4. Money cards (Pay/Collect)
        elif affects_others:
            if "pay each player" in text_lower:
                payment = money * (self.num_players - 1)
                print(f"  > {self.players[player_id].name} paga ${money} a cada jugador.")
                self.game_state[self.MONEY][player_id] -= payment
                for i in range(self.num_players):
                    if i != player_id and not self.game_state[self.BANKRUPT][i]:
                        self.game_state[self.MONEY][i] += money
            
            elif "collect" in text_lower: # Opera Night, Birthday
                collection = 0
                print(f"  > {self.players[player_id].name} cobra ${money} de cada jugador.")
                for i in range(self.num_players):
                    if i != player_id and not self.game_state[self.BANKRUPT][i]:
                        self.game_state[self.MONEY][i] -= money
                        collection += money
                        self._check_bankruptcy(i) # Comprueba si esto lleva a la bancarrota al oponente
                self.game_state[self.MONEY][player_id] += collection


        elif money > 0: # Simple pay/collect
            if "pay" in text_lower or "fees" in text_lower or "tax" in text_lower or "fine" in text_lower:
                print(f"  > {self.players[player_id].name} paga ${money}.")
                self.game_state[self.MONEY][player_id] -= money
                if "tax" in text_lower:
                    self.game_state[self.PRICES][20][0] += money # Add to free parking
            else: # Assume collect
                print(f"  > {self.players[player_id].name} cobra ${money}.")
                self.game_state[self.MONEY][player_id] += money

        # 5. Final bankruptcy check for the current player
        self._check_bankruptcy(player_id)

    def _resolve_land_on_tile(self, player_id):
        """
        Lógica de lo que pasa al *caer* en una casilla.
        """
        pos = self.game_state[self.POSITIONS][player_id]
        owner = self.game_state[self.PROPERTIES][pos]
        
        # Casilla de otro jugador
        if owner >= 0 and owner != player_id:
            level = self.game_state[self.BUILDINGS][pos]
            rent = self.game_state[self.PRICES][pos][level] # Es negativo
            
            print(f"{self.players[player_id].name} paga {rent} a {self.players[owner].name}")
            self.game_state[self.MONEY][player_id] += rent
            self.game_state[self.MONEY][owner] -= rent
            
            self._check_bankruptcy(player_id) # Usamos la nueva función
        
        # Casillas especiales (Cárcel, Impuestos, etc.)
        elif owner == -2:
            if pos == 30: # Ir a la cárcel
                self.game_state[self.JAIL_TURNS][player_id] = 3
                self.game_state[self.POSITIONS][player_id] = 10
            elif pos == 20: # Parking (cobrar)
                parking_pot = self.game_state[self.PRICES][pos][0]
                self.game_state[self.MONEY][player_id] += parking_pot
                self.game_state[self.PRICES][pos][0] = 0 # Resetear bote
            elif pos in {4, 38}: # Impuestos
                tax = self.game_state[self.PRICES][pos][0] # Es negativo
                self.game_state[self.MONEY][player_id] += tax
                self.game_state[self.PRICES][20][0] -= tax # Añade al parking
                self._check_bankruptcy(player_id) # Usamos la nueva función
            
            ### NUEVO: Lógica de cartas ###
            elif pos in {2, 17, 33}: # Community Chest
                print(f"{self.players[player_id].name} coge una carta de Caja de Comunidad.")
                self._draw_community_chest_card(player_id)
            elif pos in {7, 22, 36}: # Chance
                print(f"{self.players[player_id].name} coge una carta de Suerte.")
                self._draw_chance_card(player_id)
            
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        
        player = self.player_on_turn # Jugador que ejecuta la acción
        
        # 1. Obtenemos el Net Worth ANTES de la acción
        net_worth_before = self._calculate_net_worth(player)
        
        event_bonus = 0.0
        terminal_reward = 0.0
        reward = 0.0

        terminated = False
        truncated = False
        position = self.game_state[self.POSITIONS][player]
        
        # Comprobamos si la acción es válida (usando la máscara)
        valid_mask = self._get_action_mask()
        if not valid_mask[action]:
            reward = -10.0 # Penalización por acción inválida
        else:
            if action == 0:
                # "No-op" o "Pagar Fianza"
                if self.game_state[self.JAIL_TURNS][player] > 0:
                    self.game_state[self.MONEY][player] -= 50
                    self.game_state[self.JAIL_TURNS][player] = 0
                    event_bonus = -5.0 
                
            elif 1 <= action <= 40:
                # "Edificar" en casilla (action - 1)
                prop_idx = action - 1
                cost = self.game_state[self.EDIFICATE][prop_idx // 10]
                self.game_state[self.MONEY][player] -= cost
                self.game_state[self.BUILDINGS][prop_idx] += 1
                event_bonus = 1.0 
                
            # Omitido: 41-80 (Hipotecar)

            elif action == 41:
                # "Comprar" propiedad actual (re-indexado desde 81)
                # ¡¡CORREGIDO!!: Usamos el último valor para el precio
                price = -self.game_state[self.PRICES][position][-1] # Precio es negativo
                self.game_state[self.MONEY][player] -= price
                self.game_state[self.PROPERTIES][position] = player
                # Tu lógica original hacía +1 a -1, resultando en 0. Esto es equivalente.
                self.game_state[self.BUILDINGS][position] = 0 # Nivel 0 (solo terreno)
                event_bonus = 2.0 
                
                # TODO: Comprobar si esta compra completa un monopolio
                # if self._check_monopoly(player, position):
                #    event_bonus += 100.0 

            elif action == 42:
                # "Usar carta cárcel" (re-indexado desde 82)
                self.game_state[self.GET_OUT_OF_JAIL][player] -= 1
                self.game_state[self.JAIL_TURNS][player] = 0
                event_bonus = 5.0 

            money_after = self.game_state[self.MONEY][player]
            if money_after < 50:
                event_bonus -= 5.0 # ¡Peligro de bancarrota!
            elif money_after < 150:
                event_bonus -= 1.0
                
            # 2. Comprobamos bancarrota por la acción (ej. pagar fianza, edificar)
            if self.game_state[self.MONEY][player] < 0:
                print(f"{self.players[player].name} entra en bancarrota por su propia acción.")
                # Usamos la nueva función
                self._check_bankruptcy(player)
                terminal_reward = -500.0 # Penalización MUY grande por bancarrota
                # terminated = True


        # 3. Avanzar turno
        # Solo si no sacó dobles y no está en la cárcel (o acaba de salir)
        if not self.on_double or self.game_state[self.JAIL_TURNS][player] > 0:
            self.player_on_turn = (self.player_on_turn + 1) % self.num_players
            while self.game_state[self.BANKRUPT][self.player_on_turn]:
                self.player_on_turn = (self.player_on_turn + 1) % self.num_players
            self.on_double = False
            self.turns_on_double = 0
        
        elif self.on_double:
            self.turns_on_double += 1
            if self.turns_on_double == 3:
                self.game_state[self.JAIL_TURNS][player] = 3
                self.game_state[self.POSITIONS][player] = 10
                self.on_double = False
                self.turns_on_double = 0
                # Pasamos turno
                self.player_on_turn = (self.player_on_turn + 1) % self.num_players
                while self.game_state[self.BANKRUPT][self.player_on_turn]:
                    self.player_on_turn = (self.player_on_turn + 1) % self.num_players
        
        
        # 4. Comprobar si hay un ganador
        winner = self._check_for_winner()
        if winner != -1:
            print(f"¡El ganador es {self.players[winner].name}!")
            terminated = True
            if winner == player:
                terminal_reward = 1000.0 # Recompensa ENORME por ganar
        else:
            # 5. Tirar dados y mover al *siguiente* jugador
            next_player = self.player_on_turn
            
            # Lógica de la cárcel para el siguiente jugador
            if self.game_state[self.JAIL_TURNS][next_player] > 0:
                dices, dobles = self.roll_dice()
                self.dices = dices
                self.on_double = dobles
                if dobles:
                    self.game_state[self.JAIL_TURNS][next_player] = 0
                    self.on_double = False 
                    print(f"{self.players[next_player].name} saca dobles y sale de la cárcel.")
                else:
                    self.game_state[self.JAIL_TURNS][next_player] -= 1
                    if self.game_state[self.JAIL_TURNS][next_player] == 0:
                        print(f"{self.players[next_player].name} falla dados, paga fianza (auto).")
                        self.game_state[self.MONEY][next_player] -= 50
                        self._check_bankruptcy(next_player) # Usamos la nueva función
                    else:
                        print(f"{self.players[next_player].name} falla dados, sigue en cárcel.")
                        self.on_double = False 
                        self.player_on_turn = (self.player_on_turn + 1) % self.num_players
                        while self.game_state[self.BANKRUPT][self.player_on_turn]:
                            self.player_on_turn = (self.player_on_turn + 1) % self.num_players
            
            # Lógica de movimiento normal (si no está o acaba de salir de la cárcel)
            if self.game_state[self.JAIL_TURNS][next_player] == 0 and not terminated:
                dices, dobles = self.roll_dice()
                self.dices = dices
                self.on_double = dobles
                
                old_pos = self.game_state[self.POSITIONS][next_player]
                new_pos = (old_pos + dices) % NUM_BOARD_TILES
                self.game_state[self.POSITIONS][next_player] = new_pos
                
                if new_pos < old_pos: # Pasar por GO
                    self.game_state[self.MONEY][next_player] += 200

                # 5. Resolver caída en casilla (pagar alquileres, etc.)
                print(f'{self.players[next_player].name} landed on {self.board_names[new_pos]}')
                self._resolve_land_on_tile(next_player)
                
                # La bancarrota ya se comprueba dentro de _resolve_land_on_tile
                # if self.game_state[self.BANKRUPT][next_player]:
                #     terminated = True
            
            winner = self._check_for_winner()
            if winner != -1:
                print(f"¡El ganador es {self.players[winner].name}!")
                terminated = True
                if winner == player:
                    terminal_reward = 1000.0 # Recompensa ENORME por ganar
            
        # 7. Comprobar max_steps
        self.steps_done += 1
        if self.steps_done >= self.max_steps:
            truncated = True
            
        # 8. Calcular el Net Worth DESPUÉS de la acción
        net_worth_after = self._calculate_net_worth(player)
        
        # --- ¡RECOMPENSA FINAL! ---
        delta_net_worth = net_worth_after - net_worth_before
        reward = (delta_net_worth / 100.0) + event_bonus + terminal_reward if reward == 0.0 else reward
        
        # 9. Actualizar el net worth almacenado
        self.net_worths[player] = net_worth_after
        
        # 10. Devolver todo
        obs = self._get_obs()
        info = self._get_info()
        info["player_who_moved"] = player # Para el loop de training
        
        if self.render_mode == 'human':
            self.render(self.render_mode)
            
        if terminated or truncated:
            print(f"Juego terminado. Steps: {self.steps_done}")
            if self.render_mode == 'human':
                self.close()

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.render_mode == 'human':
            self.board.shut_down()
            pygame.quit()