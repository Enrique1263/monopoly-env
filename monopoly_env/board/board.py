import pygame
import numpy as np
import time

class Board:
    def __init__(self, screen_width, screen_height, board_width, board_height, num_cells, image_path=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.board_width = board_width
        self.board_height = board_height
        self.num_cells = num_cells

        try:
            original_image = pygame.image.load(image_path).convert() 
            self.board_image = pygame.transform.scale(
                original_image, (int(self.board_width), int(self.board_height))
            )
        except FileNotFoundError:
            print(f"Error: No se encontró la imagen del tablero '{image_path}'. Se usará un fondo blanco.")
            self.board_image = pygame.Surface((self.board_width, self.board_height))
            self.board_image.fill((255, 255, 255))
        except Exception as e:
            print(f"Error al cargar la imagen del tablero: {e}. Se usará un fondo blanco.")
            self.board_image = pygame.Surface((self.board_width, self.board_height))
            self.board_image.fill((255, 255, 255))

        self.position_coordinates = np.array([None] * 4 * (self.num_cells))
        num_inner_cells = self.num_cells - 1 # 9
        
        cell_height = self.board_height / (num_inner_cells) # e.g., 800 / 11
        corner_size = cell_height

        # El ancho total se reparte entre 2 esquinas y 9 celdas interiores
        inner_cell_width = (self.board_width - 2 * corner_size) / num_inner_cells
        
        board_x_start = (self.screen_width - self.board_width) / 2
        board_y_start = (self.screen_height - self.board_height) / 2
        
        # 5. Generar coordenadas (para el *centro* de cada casilla)
        for i in range(self.num_cells):
            
            # --- Fila Superior (Índices 0-9 del array) ---
            y_upper = board_y_start + inner_cell_width / 2
            if i == 0:
                x_upper = board_x_start + inner_cell_width / 2
            else:
                x_upper = board_x_start + corner_size + (i - 1) * inner_cell_width + inner_cell_width / 2
            self.position_coordinates[i] = (x_upper, y_upper)

            # --- Columna Derecha (Índices 10-19 del array) ---
            x_right = board_x_start + self.board_width - inner_cell_width / 2
            if i == 0:
                y_right = board_y_start + inner_cell_width / 2
            else:
                y_right = board_y_start + corner_size + (i - 1) * inner_cell_width + inner_cell_width / 2
            self.position_coordinates[self.num_cells + i] = (x_right, y_right)

            # --- Fila Inferior (Índices 20-29 del array) ---
            y_lower = board_y_start + self.board_height - inner_cell_width / 2
            if i == 0:
                x_lower = board_x_start + self.board_width - inner_cell_width / 2
            else:
                # Vamos de derecha a izquierda
                x_lower = board_x_start + self.board_width - corner_size - (i - 1) * inner_cell_width - inner_cell_width / 2
            self.position_coordinates[2 * self.num_cells + i] = (x_lower, y_lower)

            # --- Columna Izquierda (Índices 30-39 del array) ---
            x_left = board_x_start + inner_cell_width / 2
            if i == 0:
                y_left = board_y_start + self.board_height - inner_cell_width / 2
            else:
                y_left = board_y_start + self.board_height - corner_size - (i - 1) * inner_cell_width - inner_cell_width / 2
            self.position_coordinates[3 * self.num_cells + i] = (x_left, y_left)
        self.position_coordinates = np.roll(self.position_coordinates, 20)
        

    def draw(self, screen):
        # TODO: Improve the drawing of the board
        board_rect = pygame.Rect((self.screen_width - self.board_width) / 2, (self.screen_height - self.board_height) / 2, self.board_width, self.board_height)
        screen.blit(self.board_image, board_rect.topleft)

        num_inner_cells = self.num_cells - 1
        cell_height = self.board_height / (num_inner_cells)
        corner_size = cell_height
        inner_cell_width = (self.board_width - 2 * corner_size) / num_inner_cells
        board_x_start = (self.screen_width - self.board_width) / 2
        board_y_start = (self.screen_height - self.board_height) / 2

        # Dibuja las 2 líneas horizontales y 2 verticales que separan esquinas
        pygame.draw.line(screen, (0,0,0), (board_x_start, board_y_start + corner_size), (board_x_start + self.board_width, board_y_start + corner_size))
        pygame.draw.line(screen, (0,0,0), (board_x_start, board_y_start + self.board_height - corner_size), (board_x_start + self.board_width, board_y_start + self.board_height - corner_size))
        pygame.draw.line(screen, (0,0,0), (board_x_start + corner_size, board_y_start), (board_x_start + corner_size, board_y_start + self.board_height))
        pygame.draw.line(screen, (0,0,0), (board_x_start + self.board_width - corner_size, board_y_start), (board_x_start + self.board_width - corner_size, board_y_start + self.board_height))

        # Dibuja las líneas de las celdas interiores
        for i in range(num_inner_cells):
            # Líneas verticales (filas superior e inferior)
            x_top = board_x_start + corner_size + i * inner_cell_width
            pygame.draw.line(screen, (0,0,0), (x_top, board_y_start), (x_top, board_y_start + corner_size))
            pygame.draw.line(screen, (0,0,0), (x_top, board_y_start + self.board_height - corner_size), (x_top, board_y_start + self.board_height))
            
            # Líneas horizontales (columnas izquierda y derecha)
            y_left = board_y_start + corner_size + i * inner_cell_width
            pygame.draw.line(screen, (0,0,0), (board_x_start, y_left), (board_x_start + corner_size, y_left))
            pygame.draw.line(screen, (0,0,0), (board_x_start + self.board_width - corner_size, y_left), (board_x_start + self.board_width, y_left))


    def draw_players(self, screen, observation, player, dices, players):
        font = pygame.font.Font(None, 36)
        player_name = players[player].name
        text = font.render(f'{player_name}: {str(dices)}', True, players[player].color)  # Texto fijo para representar al jugador
        text_rect = text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
        screen.blit(text, text_rect)
        
        font = pygame.font.Font(None, 36)
        num_players_total = len(players)

        # Posiciones fijas de la UI para el dinero de cada jugador
        ui_positions = [
            (self.board_width * 5/16, self.screen_height / 8),                          # Jugador 0 (Top-Left)
            (self.screen_width * 3/4, self.screen_height / 8),                          # Jugador 1 (Top-Right)
            (self.board_width * 5/16, self.screen_height - self.board_height / 8),     # Jugador 2 (Bottom-Left)
            (self.screen_width * 3/4, self.screen_height - self.board_height / 8)       # Jugador 3 (Bottom-Right)
        ]
        
        for i in range(num_players_total):
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
        
        num_inner_cells = self.num_cells - 1
        cell_height = self.board_height / (num_inner_cells + 2)
        corner_size = cell_height
        inner_cell_width = (self.board_width - 2 * corner_size) / num_inner_cells
        
        offsets = [
            (0, 0),                           # Player 0
            (inner_cell_width / 2.5, 0),      # Player 1
            (0, inner_cell_width / 2.5),           # Player 2
            (inner_cell_width / 2.5, inner_cell_width / 2.5) # Player 3
        ]
        
        for i, player in enumerate(players):
            position = observation[1][i]
            # position = int(time.time() * 1000 / 500) % 40  # Movimiento automático para debug
            color = player.color
            
            # Obtiene el centro de la casilla
            base_x, base_y = self.position_coordinates[position]
            
            # Determina si es esquina o fila
            is_corner = (position % 10 == 0)
            
            # Ajusta el offset para que quepan bien
            if is_corner:
                offset_x = (offsets[i][0] / inner_cell_width) * corner_size
                offset_y = offsets[i][1]
            else:
                offset_x = offsets[i][0]
                offset_y = offsets[i][1]

            # Resta la mitad del "área de offset" para centrarlos
            if is_corner:
                center_offset_x = corner_size / 4
                center_offset_y = corner_size / 4
            else:
                center_offset_x = inner_cell_width / 4
                center_offset_y = inner_cell_width / 4

            x = base_x + offset_x - center_offset_x
            y = base_y + offset_y - center_offset_y
            
            pygame.draw.circle(screen, color, (int(x), int(y)), 8)

    def shut_down(self):
        pygame.quit()