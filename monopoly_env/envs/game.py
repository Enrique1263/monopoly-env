import gym
from gym import spaces
import numpy as np
import random
import time
from monopoly_env.board.board import Board
import pygame
from player import Player
from typing import List


# Window dimensions
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800

# Board dimensions
BOARD_WIDTH = 800
BOARD_HEIGHT = 800

# Cells in the board
NUM_CELLS = 10

pygame.init()

# define the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Monopoly")

# define the board
board = Board(SCREEN_WIDTH, SCREEN_HEIGHT, BOARD_WIDTH, BOARD_HEIGHT, NUM_CELLS)

'''
In reset, players, the board, and properties are initialized. The first roll is made so that the first player has an observation.
In step, the player's action is executed and the game state is updated. Dice are rolled so that the next player has an observation.
'''
# TO DO: Community Chest and Chance cards logic
class MonopolyEnv(gym.Env):
    def __init__(self, players: List[Player], max_steps=1000, render_mode='Human', board_names_path='cards/board_names.txt', community_chest_path='cards/community_chest.txt', chance_path='cards/chance.txt'):
        self.num_players = len(players)
        self.action_space = spaces.Discrete(3)
        self.board_names_path = board_names_path
        self.community_chest_path = community_chest_path
        self.chance_path = chance_path

        self.observation_space = spaces.Box(low=-1, high=self.num_players, shape=(8,), dtype=np.int32)
        self.observation = [[-2]+[-1]+[-2]+[-1]+[0]+[-1]*2+[-2]+[-1]*2+[-2]+[-1]*6+[-2]+[-1]*2+[-2]+[-1]+[-2]+[-1]*7+[-2]+[-1]*2+[-2]+[-1]*2+[-1]+[-1]+[0]+[-1] , [0 for _ in range(self.num_players)], [1500 for _ in range(self.num_players)],
                                    [[200], [-2, -10, -30, -90, -160, -250, -60], [0], [-4, -20, -60, -180, -320, -450, -60], [-200], [-25, -50, -100, -200, -200],
                                     [-6, -30, -90, -270, -400, -550, -100], [0], [-6, -30, -90, -270, -400, -550, -100], 
                                     [-8, -40, -100, -300, -450, -600, -120], [0], [-10, -50, -150, -450, -625, -750, -140], [-4, -10, -150], [-10, -50, -150, -450, -625, -750, -140],
                                     [-12, -60, -180, -500, -700, -900, -160], [-25, -50, -100, -200, -200], [-14, -70, -200, -550, -750, -950, -180], [0], [-14, -70, -200, -550, -750, -950, -180],
                                     [-16, -80, -220, -600, -800, -1000, -200], [0], [-18, -90, -250, -700, -875, -1050, -220], [0], [-18, -90, -250, -700, -875, -1050, -220],
                                     [-20, -100, -300, -750, -925, -1100, -240], [-25, -50, -100, -200, -200], [-22, -110, -330, -800, -975, -1150, -260], [-22, -110, -330, -800, -975, -1150, -260], [-4, -10, -150],
                                     [-24, -120, -360, -850, -1025, -1200, -280], [0], [-26, -130, -390, -900, -1100, -1275, -300], [-26, -130, -390, -900, -1100, -1275, -300], [0],
                                     [-28, -150, -450, -1000, -1200, -1400, -320], [-25, -50, -100, -200, -200], [0], [-35, -175, -500, -1100, -1300, -1500, -350], [-100], [-50, -200, -600, -1400, -1700, -2000, -400]],
                                     [50,100,150,200], [-2 if i in {2, 4, 7, 10, 17, 20, 22, 30, 33, 36, 38} else -1 for i in range(40)], [0 for _ in range(self.num_players)], [False for _ in range(self.num_players)]]
        self.PROPERTIES = 0 # Who owns each property (-2 if it can't be bought, -1 if it's not owned, 0, 1, 2, ... if it's owned by a player)
        self.POSITIONS = 1 # Position of each player
        self.MONEY = 2  # Money of each player
        self.PRICES = 3 # Prices of each property
        self.EDIFICATE = 4 # Cost of building on each property
        self.BUILDINGS = 5 # Level of building on each property (-2 if it can't be built, -1 if it's not someones property yet, 0, 1, 2, 3, 4, 5 if it's built)
        self.JAIL_TURNS = 6 # Turns left in jail
        self.BANKRUPT = 7 # Bankruptcy status of each player
        self.players = players
        self.board = self.load_board()
        self.community_chest_cards = self.load_cards(self.community_chest_path)
        self.chance_cards = self.load_cards(self.chance_path)
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.steps_done = 0
        self.player_on_turn = None
        self.on_double = False
        self.turns_on_double = 0
        self.dices = 0

    def load_board(self):
        board_names = {}
        with open(self.board_names_path, 'r') as file:
            for line in file:
                position, name = line.strip().split(':')
                board_names[int(position)] = name
        return board_names
    
    def load_cards(self, file_path):
        cards = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('|')
                card_text = parts[0]
                money = int(parts[1])
                affects_others = parts[2].lower() == 'true'
                tile = int(parts[3])
                cards.append((card_text, money, affects_others, tile))
        return np.array(cards, dtype=object)

    def reset(self):
        # Reset the game to the initial state
        self.observation = [[-2]+[-1]+[-2]+[-1]+[-2]+[-1]*2+[-2]+[-1]*2+[-2]+[-1]*6+[-2]+[-1]*2+[-2]+[-1]+[-2]+[-1]*7+[-2]+[-1]*2+[-2]+[-1]*2+[-2]+[-1]+[-2]+[-1] , [0 for _ in range(self.num_players)], [1500 for _ in range(self.num_players)],
                                    [[200], [-2, -10, -30, -90, -160, -250, -60], [0], [-4, -20, -60, -180, -320, -450, -60], [-200], [-25, -50, -100, -200, -200],
                                     [-6, -30, -90, -270, -400, -550, -100], [0], [-6, -30, -90, -270, -400, -550, -100], 
                                     [-8, -40, -100, -300, -450, -600, -120], [0], [-10, -50, -150, -450, -625, -750, -140], [-4, -10, -150], [-10, -50, -150, -450, -625, -750, -140],
                                     [-12, -60, -180, -500, -700, -900, -160], [-25, -50, -100, -200, -200], [-14, -70, -200, -550, -750, -950, -180], [0], [-14, -70, -200, -550, -750, -950, -180],
                                     [-16, -80, -220, -600, -800, -1000, -200], [0], [-18, -90, -250, -700, -875, -1050, -220], [0], [-18, -90, -250, -700, -875, -1050, -220],
                                     [-20, -100, -300, -750, -925, -1100, -240], [-25, -50, -100, -200, -200], [-22, -110, -330, -800, -975, -1150, -260], [-22, -110, -330, -800, -975, -1150, -260], [-4, -10, -150],
                                     [-24, -120, -360, -850, -1025, -1200, -280], [0], [-26, -130, -390, -900, -1100, -1275, -300], [-26, -130, -390, -900, -1100, -1275, -300], [0],
                                     [-28, -150, -450, -1000, -1200, -1400, -320], [-25, -50, -100, -200, -200], [0], [-35, -175, -500, -1100, -1300, -1500, -350], [-100], [-50, -200, -600, -1400, -1700, -2000, -400]],
                                     [50,100,150,200], [-2 if i in {2, 4, 7, 10, 17, 20, 22, 30, 33, 36, 38} else -1 for i in range(40)], [0 for _ in range(self.num_players)], [False for _ in range(self.num_players)]]
        self.star_order()
        for i, player in enumerate(self.players):
            player.order = i
            print(f'Player {player.name} es el jugador {player.order}')
        self.player_on_turn = 0
        dices, dobles = self.roll_dice()
        self.on_double = dobles
        # Move player to the corresponding tile
        self.observation[self.POSITIONS][self.player_on_turn] = (self.observation[self.POSITIONS][self.player_on_turn] + dices) % 40

        self.dices = dices
        self.render(self.render_mode)
        print(f'Player {self.player_on_turn} ha caído en la casilla {self.board[self.observation[self.POSITIONS][self.player_on_turn]]}')
        
        return self.observation, 0, False, False, {}
    
    def roll_dice(self):
        # Simulate the roll of two dice and return the result
        dice1 = random.randint(1, 6)
        dice2 = random.randint(1, 6)
        return dice1 + dice2, dice1 == dice2

    
    def reset_properties(self, player):
        # Remove houses and hotels from the player's properties and return the properties to the bank
        for property in range(len(self.observation[self.PROPERTIES])):
            if self.observation[self.PROPERTIES][property] == player:
                self.observation[self.BUILDINGS][property] = -1
                self.observation[self.PROPERTIES][property] = -1


    def render(self, mode='Human'):
        # Show board and players on the screen
        if mode == 'Human':
            board.draw(screen)
            board.draw_players(screen, self.observation, self.player_on_turn, self.dices, self.players)
            pygame.display.flip()
            time.sleep(0.01)

    def step(self, action):
        player = self.player_on_turn
        # action: 0 - No hacer nada, 1 - Comprar propiedad, 2 - Edificar
        # Ver donde está el jugador
        position = self.observation[self.POSITIONS][player]
        # Comprobar a quien pertenece la propiedad
        owner = self.observation[self.PROPERTIES][position]
        if owner == -2:
            # Tile without owner, player can't buy it
            if position == 30:
                # Jail tile, player goes to jail
                self.observation[self.JAIL_TURNS][player] = 3
                self.observation[self.POSITIONS][player] = 10
                reward = 0
            elif position == 20:
                reward = self.observation[self.PRICES][position][0]
                self.observation[self.PRICES][20][0] = 0
            elif position == 4 or position == 38:
                # Taxes, goes to parking
                reward = self.observation[self.PRICES][position][0]
                self.observation[self.PRICES][20][0] -= reward
            else:
                # Community Chest and Chance, TODO
                reward = 0
            self.observation[self.MONEY][player] += reward
            if self.observation[self.MONEY][player] < 0:
                self.observation[self.BANKRUPT][player] = True
                self.reset_properties(player)
                self.players[player].color = (0, 0, 0)
                print(f'Player {player} ha caído en bancarrota por no poder pagar la casilla {self.board[position]}')
                # No players left, end game
                if len([player for player in self.observation[self.BANKRUPT] if not player]) == 1:
                    if self.render_mode == 'Human':
                        board.shut_down()
                    return self.observation, reward, True, False, {}
            # If not a valid action, penalize player
            if action != 0:
                reward -= 100
        
        elif owner == -1:
            # Tile without owner, player can buy it
            # TODO: Water Works and Electric Company level is set different, stations too
            # While under process, the player can't buy the property
            if position not in {5, 12, 15, 25, 28, 35}:
                if action == 1:
                    level = self.observation[self.BUILDINGS][position]
                    price = self.observation[self.PRICES][position][level]
                    if self.observation[self.MONEY][player] >= -price:
                        self.observation[self.MONEY][player] += price
                        self.observation[self.PROPERTIES][position] = player
                        self.observation[self.BUILDINGS][position] += 1
                        reward = price
                    else:
                        # If player can't buy the property, penalize him
                        reward = -100
                elif action == 2:
                    # If player tries to build on a property he doesn't own, penalize him
                    reward = -100
                else:
                    reward = 0
            else:
                reward = 0

        elif owner == player:
            # Tile owned by the player, player can build on it
            # TODO: Water Works and Electric Company level is set different, stations too
            if action == 2 and position not in {5, 12, 15, 25, 28, 35}:
                level = self.observation[self.BUILDINGS][position]
                cost = self.observation[self.EDIFICATE][position//10]
                if level < 5 and self.observation[self.MONEY][player] >= cost:
                    self.observation[self.MONEY][player] -= cost
                    self.observation[self.BUILDINGS][position] += 1
                    reward = -cost
                else:
                    # If player can't build on the property, penalize him
                    reward = -100
            elif action == 1:
                # If player tries to buy a property he already owns, penalize him
                reward = -100
            else:
                reward = 0

        else:
            # Tile owned by another player, player must pay rent
            # TODO: Water Works and Electric Company level is set different, stations too
            level = self.observation[self.BUILDINGS][position]
            rent = self.observation[self.PRICES][position][level]
            self.observation[self.MONEY][player] += rent
            self.observation[self.MONEY][owner] -= rent
            reward = rent
            # If player can't pay the rent, he goes bankrupt
            if self.observation[self.MONEY][player] < 0:
                self.observation[self.BANKRUPT][player] = True
                self.reset_properties(player)
                self.players[player].color = (0, 0, 0)
                print(f'Player {player} ha caído en bancarrota por no poder pagar el alquiler de la casilla {self.board[position]} al jugador {owner}')
                # No players left, end game
                if len([player for player in self.observation[self.BANKRUPT] if not player]) == 1:
                    if self.render_mode == 'Human':
                        board.shut_down()
                    return self.observation, reward, True, False, {}

        if not self.on_double or self.observation[self.JAIL_TURNS][player] > 0:
            # print(f'Player {player} ha terminado su turno, dinero restante: {self.observation[self.MONEY][player]}')
            # Pass turn to the next player, skipping bankrupt players
            self.player_on_turn = (self.player_on_turn + 1) % self.num_players
            while self.observation[self.BANKRUPT][self.player_on_turn]:
                self.player_on_turn = (self.player_on_turn + 1) % self.num_players
        elif self.on_double:
            self.turns_on_double += 1
            if self.turns_on_double == 3:
                print(f'Player {player} ha sacado dobles 3 veces seguidas, va a la cárcel')
                self.observation[self.JAIL_TURNS][player] = 3
                self.observation[self.POSITIONS][player] = 10
                self.on_double = False
                self.turns_on_double = 0
                self.player_on_turn = (self.player_on_turn + 1) % self.num_players
                while self.observation[self.BANKRUPT][self.player_on_turn]:
                    self.player_on_turn = (self.player_on_turn + 1) % self.num_players


        # Throw dice for the next player
        dices, dobles = self.roll_dice()
        self.on_double = dobles
        # Check if the player is in jail
        while self.observation[self.JAIL_TURNS][self.player_on_turn] > 0:
            if self.on_double:
                # If the player has thrown doubles, he can leave jail
                self.observation[self.JAIL_TURNS][self.player_on_turn] = 0
                # but he can't throw dices again
                self.on_double = False
            else:
                # If the player hasn't thrown doubles, discount a turn in jail
                self.observation[self.JAIL_TURNS][self.player_on_turn] -= 1
                if self.observation[self.JAIL_TURNS][self.player_on_turn] > 0:
                    # If the player is still in jail, pass turn to the next player
                    self.player_on_turn = (self.player_on_turn + 1) % self.num_players
                    while self.observation[self.BANKRUPT][self.player_on_turn]:
                        self.player_on_turn = (self.player_on_turn + 1) % self.num_players
                    dices, dobles = self.roll_dice()
                    self.on_double = dobles
        
        # Move player to the corresponding tile
        self.observation[self.POSITIONS][self.player_on_turn] = (self.observation[self.POSITIONS][self.player_on_turn] + dices) % 40
        # Look up if player has passed through the start tile
        if self.observation[self.POSITIONS][self.player_on_turn] < dices:
            self.observation[self.MONEY][self.player_on_turn] += 200

        self.dices = dices
        self.render(self.render_mode)
        print(f'Player {self.player_on_turn} ha caído en la casilla {self.board[self.observation[self.POSITIONS][self.player_on_turn]]}')

        self.steps_done += 1
        if self.steps_done >= self.max_steps:
            if self.render_mode == 'Human':
                board.shut_down()
            return self.observation, reward, False, True, {}

        return self.observation, reward, False, False, {}

    def star_order(self):
        # Shuffle the players order
        self.players = np.random.permutation(self.players)

    def close(self):
        if self.render_mode == 'Human':
            board.shut_down()