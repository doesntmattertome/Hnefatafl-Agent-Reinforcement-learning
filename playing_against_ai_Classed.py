import pygame
import sys
import numpy as np
from game import Board
import random
from dqn_classes_two import HnefataflEnv, DQNAgent
import time

# UML Diagram for Hnefatafl Game



class HnefataflGUI:
    def __init__(self):
        # Initialize Pygame and constants
        pygame.init()
        self.WINDOW_SIZE = 600
        self.GRID_SIZE = 11
        self.CELL_SIZE = self.WINDOW_SIZE // self.GRID_SIZE
        self.FPS = 60

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BROWN = (139, 69, 19)
        self.BEIGE = (245, 245, 220)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.GOLD = (255, 215, 0)
        self.COPPER = (184, 115, 51)
        self.PURPLE = (128, 0, 128)  # For corner squares

        # Load and resize images
        self.WHITE_PIECE_IMG = pygame.image.load("white.png")
        self.BLACK_PIECE_IMG = pygame.image.load("black.png")
        self.KING_PIECE_IMG = pygame.image.load("king.png")
        self.WHITE_PIECE_IMG = pygame.transform.scale(self.WHITE_PIECE_IMG, (self.CELL_SIZE, self.CELL_SIZE))
        self.BLACK_PIECE_IMG = pygame.transform.scale(self.BLACK_PIECE_IMG, (self.CELL_SIZE, self.CELL_SIZE))
        self.KING_PIECE_IMG = pygame.transform.scale(self.KING_PIECE_IMG, (self.CELL_SIZE, self.CELL_SIZE))

        # Initialize screen, clock, and board
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption("Hnefatafl")
        self.clock = pygame.time.Clock()
        self.board = Board()
        self.board.print_amount()
        self.current_player = 1  # Black starts first

        # DQN Agents
        state_size = (11, 11)
        action_size = 11 * 11 * 11 * 11
        self.dqn_agent_black = DQNAgent(state_size, action_size, player=1)
        self.dqn_agent_black.load_weights("black_model.weights.h5")
        # self.dqn_agent_black.print_weights()
        # self.dqn_agent_black.model_summary()

        self.dqn_agent_white = DQNAgent(state_size, action_size, player=1)
        self.dqn_agent_white.load_weights("white_model.weights.h5")
        # self.dqn_agent_white.print_weights()
        # self.dqn_agent_white.model_summary()

        # Game state variables
        self.selected_piece = None
        self.possible_moves = []
        self.winner = None
        self.dragging_piece = False
        self.dragged_piece_pos = None
        self.player_color = 1  # Default to playing as black

    def draw_board(self):
        """Draw the game board and pieces."""
        self.screen.fill(self.BEIGE)
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                rect = pygame.Rect(col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if (
                    (row == 0 and (col == 0 or col == self.GRID_SIZE - 1)) or
                    (row == self.GRID_SIZE - 1 and (col == 0 or col == self.GRID_SIZE - 1))
                ):
                    pygame.draw.rect(self.screen, self.PURPLE, rect)
                else:
                    pygame.draw.rect(self.screen, self.BEIGE, rect)
                pygame.draw.rect(self.screen, self.BROWN, rect, 1)
        current_board = self.board.get_board()
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                piece = current_board[row][col]
                if piece == 1:  # Attacker
                    self.screen.blit(self.BLACK_PIECE_IMG, (col * self.CELL_SIZE, row * self.CELL_SIZE))
                elif piece == -1:  # Defender
                    self.screen.blit(self.WHITE_PIECE_IMG, (col * self.CELL_SIZE, row * self.CELL_SIZE))
                elif piece == -2:  # King
                    self.screen.blit(self.KING_PIECE_IMG, (col * self.CELL_SIZE, row * self.CELL_SIZE))

    def get_cell_under_mouse(self):
        """Get the grid cell under the mouse cursor."""
        mouse_pos = pygame.mouse.get_pos()
        col = mouse_pos[0] // self.CELL_SIZE
        row = mouse_pos[1] // self.CELL_SIZE
        if 0 <= row < self.GRID_SIZE and 0 <= col < self.GRID_SIZE:
            return row, col
        return None

    def handle_mouse_click(self):
        """Handle mouse click for selecting and moving pieces."""
        if self.winner:
            return
        cell = self.get_cell_under_mouse()
        if cell is None:
            return
        row, col = cell

        if self.selected_piece:
            if (row, col) in self.possible_moves:
                if self.board.get_board()[self.selected_piece[0]][self.selected_piece[1]] == -2:
                    self.board.move(-2, self.selected_piece, (row, col))
                else:
                    self.board.move(self.current_player, self.selected_piece, (row, col))
                capture = self.board.capture_enemies(self.current_player, (row, col))
                if capture != -1:
                    time.sleep(0.3)
                self.selected_piece = None
                self.possible_moves = []
                self.current_player *= -1
                self.winner = self.board.check_winner()
            else:
                self.selected_piece = None
                self.possible_moves = []
        elif (self.board.get_board()[row][col] == self.current_player or 
              (self.current_player == -1 and self.board.get_board()[row][col] == -2)):
            self.selected_piece = (row, col)
            self.dragging_piece = True
            self.dragged_piece_pos = pygame.mouse.get_pos()
            if self.board.get_board()[row][col] == -2:
                self.possible_moves = self.board.get_possible_moves(-2, self.selected_piece)
            else:
                self.possible_moves = self.board.get_possible_moves(self.current_player if self.board.get_board()[row][col] != -2 else -1, self.selected_piece)

    def handle_mouse_release(self):
        """Handle mouse release for placing pieces."""
        if self.winner:
            return
        cell = self.get_cell_under_mouse()
        if cell is None:
            return
        row, col = cell
        if self.selected_piece and self.dragging_piece:
            if (row, col) in self.possible_moves:
                if self.board.get_board()[self.selected_piece[0]][self.selected_piece[1]] == -2:
                    self.board.move(-2, self.selected_piece, (row, col))
                else:
                    self.board.move(self.current_player, self.selected_piece, (row, col))
                self.board.capture_enemies(self.current_player, (row, col))
                self.selected_piece = None
                self.possible_moves = []
                self.current_player *= -1
                self.winner = self.board.check_winner()
            else:
                self.selected_piece = None
                self.possible_moves = []
        self.dragging_piece = False
        self.dragged_piece_pos = None

    def draw_win_screen(self):
        """Draw the win screen."""
        self.screen.fill(self.WHITE)
        font = pygame.font.Font(None, 74)
        text = font.render(f"{'White' if self.winner == -1 else 'black'} wins!", True, self.BLACK)
        text_rect = text.get_rect(center=(self.WINDOW_SIZE // 2, self.WINDOW_SIZE // 2 - 50))
        self.screen.blit(text, text_rect)
        reset_button = pygame.Rect(self.WINDOW_SIZE // 2 - 100, self.WINDOW_SIZE // 2 + 50, 200, 50)
        pygame.draw.rect(self.screen, self.GREEN, reset_button)
        font = pygame.font.Font(None, 36)
        text = font.render("Reset", True, self.BLACK)
        text_rect = text.get_rect(center=reset_button.center)
        self.screen.blit(text, text_rect)
        return reset_button

    def draw_choice_buttons(self):
        """Draw the buttons to choose player color."""
        self.screen.fill(self.WHITE)
        font = pygame.font.Font(None, 74)
        text = font.render("Choose your color", True, self.BLACK)
        text_rect = text.get_rect(center=(self.WINDOW_SIZE // 2, self.WINDOW_SIZE // 2 - 100))
        self.screen.blit(text, text_rect)
        # Black button
        black_button = pygame.Rect(self.WINDOW_SIZE // 2 - 150, self.WINDOW_SIZE // 2, 100, 50)
        pygame.draw.rect(self.screen, self.BLACK, black_button)
        font = pygame.font.Font(None, 36)
        text = font.render("Black", True, self.WHITE)
        text_rect = text.get_rect(center=black_button.center)
        self.screen.blit(text, text_rect)
        # White button
        white_button = pygame.Rect(self.WINDOW_SIZE // 2 + 50, self.WINDOW_SIZE // 2, 100, 50)
        pygame.draw.rect(self.screen, self.WHITE, white_button)
        font = pygame.font.Font(None, 36)
        text = font.render("White", True, self.BLACK)
        text_rect = text.get_rect(center=white_button.center)
        self.screen.blit(text, text_rect)
        return black_button, white_button

    def reset_game(self):
        """Reset the game to its initial state."""
        self.board = Board()
        self.current_player = 1
        self.selected_piece = None
        self.possible_moves = []
        self.winner = None

    def ai_move(self):
        """Perform an AI move based on the current player."""
        if self.current_player == 1 and not self.winner and self.player_color == -1:
            state = self.board.get_board()
            env = HnefataflEnv()
            env.board = self.board
            env.current_player = self.current_player
            valid_moves = env.valid_moves_mask()
            action = self.dqn_agent_black.act(state, valid_moves, exploit=True)
            start_x = action // (11 * 11 * 11)
            start_y = (action % (11 * 11 * 11)) // (11 * 11)
            end_x = (action % (11 * 11)) // 11
            end_y = action % 11
            self.board.move(1, (start_x, start_y), (end_x, end_y))
            capture = self.board.capture_enemies(1, (end_x, end_y))
            if capture != -1:
                time.sleep(0.3)
            self.current_player *= -1
            self.winner = self.board.check_winner()
        elif self.current_player == -1 and not self.winner and self.player_color == 1:
            state = self.board.get_board()
            env = HnefataflEnv()
            env.board = self.board
            env.current_player = self.current_player
            valid_moves = env.valid_moves_mask()
            action = self.dqn_agent_white.act(state, valid_moves, exploit=True)
            start_x = action // (11 * 11 * 11)
            start_y = (action % (11 * 11 * 11)) // (11 * 11)
            end_x = (action % (11 * 11)) // 11
            end_y = action % 11
            self.board.move(-1, (start_x, start_y), (end_x, end_y))
            capture = self.board.capture_enemies(-1, (end_x, end_y))
            if capture != -1:
                time.sleep(0.3)
            self.current_player *= -1
            self.winner = self.board.check_winner()

    def run(self):
        choosing_color = True
        black_button, white_button = self.draw_choice_buttons()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if choosing_color:
                        if black_button.collidepoint(event.pos):
                            self.player_color = 1
                            choosing_color = False
                            self.reset_game()
                        elif white_button.collidepoint(event.pos):
                            self.player_color = -1
                            choosing_color = False
                            self.reset_game()
                    elif self.winner:
                        reset_button = self.draw_win_screen()
                        if reset_button.collidepoint(event.pos):
                            choosing_color = True
                    else:
                        self.handle_mouse_click()
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_release()
                elif event.type == pygame.MOUSEMOTION and self.dragging_piece:
                    self.dragged_piece_pos = pygame.mouse.get_pos()

            if choosing_color:
                black_button, white_button = self.draw_choice_buttons()
            elif self.winner:
                self.draw_win_screen()
            else:
                self.draw_board()
                # Highlight selected piece and possible moves
                if self.selected_piece:
                    pygame.draw.rect(
                        self.screen, self.GREEN,
                        (
                            self.selected_piece[1] * self.CELL_SIZE,
                            self.selected_piece[0] * self.CELL_SIZE,
                            self.CELL_SIZE, self.CELL_SIZE
                        ), 3
                    )
                    for move in self.possible_moves:
                        pygame.draw.rect(
                            self.screen, self.GREEN,
                            (
                                move[1] * self.CELL_SIZE,
                                move[0] * self.CELL_SIZE,
                                self.CELL_SIZE, self.CELL_SIZE
                            ), 3
                        )
                # Draw dragged piece
                if self.selected_piece and self.dragging_piece and self.dragged_piece_pos:
                    piece = self.board.get_board()[self.selected_piece[0]][self.selected_piece[1]]
                    pos = (self.dragged_piece_pos[0] - self.CELL_SIZE // 2, self.dragged_piece_pos[1] - self.CELL_SIZE // 2)
                    if piece == 1:
                        self.screen.blit(self.BLACK_PIECE_IMG, pos)
                    elif piece == -1:
                        self.screen.blit(self.WHITE_PIECE_IMG, pos)
                    elif piece == -2:
                        self.screen.blit(self.KING_PIECE_IMG, pos)

                # AI moves
                self.ai_move()

            pygame.display.flip()
            self.clock.tick(self.FPS)

if __name__ == "__main__":
    game = HnefataflGUI()
    game.run()
