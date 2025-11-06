import pygame
import numpy as np

class Board:
    def __init__(self, screen_width, screen_height, board_width, board_height, num_cells):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.board_width = board_width
        self.board_height = board_height
        self.num_cells = num_cells

        self.position_coordinates = np.array([None] * 4 * (self.num_cells))
        cell_width = self.board_width / (self.num_cells + 1)
        cell_height = self.board_height / (self.num_cells + 1)

        for i in range(self.num_cells):
            # Upper row
            self.position_coordinates[i] = ((self.screen_width - self.board_width) / 2 + (i + 1) * cell_width + cell_width/4, (self.screen_height - self.board_height) / 2 + cell_height / 4)
            # Right column
            self.position_coordinates[self.num_cells + i] = ((self.screen_width - self.board_width) / 2 + self.board_width - 3/4 * cell_width, (self.screen_height - self.board_height) / 2 + (i + 1) * cell_height + cell_height/4)
            # Lower row
            self.position_coordinates[2 * (self.num_cells) + i] = ((self.screen_width - self.board_width) / 2 + (self.num_cells - 1 - i) * cell_width + cell_width / 4, (self.screen_height - self.board_height) / 2 + self.board_height - 3/4 * cell_height)
            # Left column
            self.position_coordinates[3 * (self.num_cells) + i] = ((self.screen_width - self.board_width) / 2 + cell_width/4, (self.screen_height - self.board_height) / 2 + (self.num_cells - 1 - i) * cell_height + cell_height / 4)
        self.position_coordinates = np.roll(self.position_coordinates, 21)
        

    def draw(self, screen):
        # TODO: Improve the drawing of the board
        board_rect = pygame.Rect((self.screen_width - self.board_width) / 2, (self.screen_height - self.board_height) / 2, self.board_width, self.board_height)
        pygame.draw.rect(screen, (255, 255, 255), board_rect)

        cell_width = self.board_width / (self.num_cells + 1)
        cell_height = self.board_height / (self.num_cells + 1)
        for i in range(self.num_cells + 1):
            # Upper line
            pygame.draw.rect(screen, (0, 0, 0), (board_rect.left + i * cell_width, board_rect.top, cell_width, cell_height), 1)
            # Lower line
            pygame.draw.rect(screen, (0, 0, 0), (board_rect.left + i * cell_width, board_rect.bottom - cell_height, cell_width, cell_height), 1)
            # Left line
            pygame.draw.rect(screen, (0, 0, 0), (board_rect.left, board_rect.top + i * cell_height, cell_width, cell_height), 1)
            # Right line
            pygame.draw.rect(screen, (0, 0, 0), (board_rect.right - cell_width, board_rect.top + i * cell_height, cell_width, cell_height), 1)

    def draw_players(self, screen, observation, player, dices, players):
        font = pygame.font.Font(None, 36)
        player_name = players[player].name
        text = font.render(f'{player_name}: {str(dices)}', True, players[player].color)  # Texto fijo para representar al jugador
        text_rect = text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
        screen.blit(text, text_rect)
        
        cell_width = self.board_width / (self.num_cells + 1)
        cell_height = self.board_height / (self.num_cells + 1)
        
        font = pygame.font.Font(None, 36)
        num_players_total = len(players) # Obtenemos el número real de jugadores

        # Posiciones fijas de la UI para el dinero de cada jugador
        ui_positions = [
            (self.board_width * 5/16, self.screen_height / 8),                          # Jugador 0 (Top-Left)
            (self.screen_width * 3/4, self.screen_height / 8),                          # Jugador 1 (Top-Right)
            (self.board_width * 5/16, self.screen_height - self.board_height / 8),     # Jugador 2 (Bottom-Left)
            (self.screen_width * 3/4, self.screen_height - self.board_height / 8)       # Jugador 3 (Bottom-Right)
        ]
        
        # Recorremos los jugadores que realmente existen
        for i in range(num_players_total):
            # Comprobamos que el jugador y sus datos existan (buena práctica)
            if i < len(players) and i < len(observation[2]):
                
                money = observation[2][i] # observation[2] es la lista de MONEY
                color = players[i].color
                
                # Si el color es negro (bancarrota), mostramos el texto en gris
                if color == (0, 0, 0):
                    text_color = (150, 150, 150)
                else:
                    text_color = color
                
                text = font.render(f'Money: {money}', True, text_color)
                text_rect = text.get_rect(center=ui_positions[i])
                screen.blit(text, text_rect)
        
        for i, player_obj in enumerate(players):
            position = observation[1][i] # observation[1] es POSITIONS
            color = player_obj.color
            x, y = self.position_coordinates[position]
            if i == 1:
                x += cell_width / 2
            elif i == 2:
                y += cell_height / 2
            elif i == 3:
                x += cell_width / 2
                y += cell_height / 2       
            pygame.draw.circle(screen, color, (int(x), int(y)), 10)

    def shut_down(self):
        pygame.quit()