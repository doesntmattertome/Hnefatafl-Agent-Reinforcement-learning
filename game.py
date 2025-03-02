import numpy as np
"""

"""
import random



# classic implimentation of hnefatafl
class Board:
    def __init__(self):
        # black is 1, white is -1, empty is 0 -2 is the king 2 is edge
        # board is 11x11 - [x][y](column, row)
        self.board = np.array([[2,0,0,1,1,1,1,1,0,0,2],
                              [0,0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0],
                              [1,0,0,0,0,-1,0,0,0,0,1],
                              [1,0,0,0,-1,-1,-1,0,0,0,1],
                              [1,1,0,-1,-1,-2,-1,-1,0,1,1],
                              [1,0,0,0,-1,-1,-1,0,0,0,1],
                              [1,0,0,0,0,-1,0,0,0,0,1],
                              [0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0,0],
                              [2,0,0,1,1,1,1,1,0,0,2]])
        
    def print_board(self, positions_marked = []):
        column_index = 0
        for column in self.board:
            row_index = 0
            if (column_index < 10):
                if (column_index != 0):
                    print(column_index, end="  ")
            else:
                print(column_index, end=" ")
            for row in column:
                if (column_index == 0 and row_index == 0):
                    print("   ", end = "")
                    for i in range(11):
                        print(i, end = " ")
                    print("")
                    print("0  ", end="")                    
                if (row_index, column_index) in positions_marked:
                    print("X", end = " ")
                elif row == 0:
                    print("▢", end = " ")
                elif row == 1:
                    print("B", end = " ")
                elif row == -1:
                    print("W", end = " ")
                elif row == -2:
                    print("K", end = " ")
                elif row == 2:
                    print("▨", end = " ")
                
                
                row_index += 1
            column_index += 1
            row_index = 0
            print()
    def mask_directions(self, position, directions):
        # mask the directions that are not possible to move in
        masked_directions = []
        for dx, dy in directions:
            if (0 <= position[0] + dx < 11 and 0 <= position[1] + dy < 11):
                masked_directions.append((dx, dy))
        return masked_directions
    # check if a player has won, if black return 1, if white return -1, if no one return 0
    def check_winner(self):
        # if no black pieces are on the board, white wins
        if not (np.argwhere(self.board == 1).size):
            return -1
        #print("checking winner")
        # check if the king is in one of the four corners
        king_pos = np.argwhere(self.board == -2)
        if king_pos.size:
            kx, ky = king_pos[0]
            if (kx, ky) in [(0, 0), (0, 10), (10, 0), (10, 10)]:
                #print("white wins")
                #self.print_board()
                return -1
        else:
            #print("black wins")
            #self.print_board()
            return 1
        """
        # check if the king is surrounded by black pieces or the edge of the board
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        surrounded = True
        for dx, dy in directions:
            nx, ny = kx + dx, ky + dy
            if not (0 <= nx < 11 and 0 <= ny < 11 and self.board[nx, ny] in [1, 2]):
                surrounded = False
                break
        if surrounded:
            if not (np.argwhere(self.board == -1).size):
                return 1
        """
        return 0
    def how_much_king_surrounded(self):
      directions = [(0,1), (0,-1), (1,0), (-1,0)]
      king_pos = np.argwhere(self.board == -2)
      if king_pos.size:
        directions_masked = self.mask_directions(king_pos[0], directions)
        kx, ky = king_pos[0]
        amount_surrounded = 4
        for dx, dy in directions_masked:
            nx, ny = kx + dx, ky + dy
            if not (0 <= nx < 11 and 0 <= ny < 11 and self.board[nx, ny] == 1):
                amount_surrounded -= 1
        return amount_surrounded
    
    def capture_enemies(self, player, end):
        # the king can't capture, if king don't capture return -1
        if (self.board[end[0], end[1]] == -2):
            return -1
        # Define potential directions (up, down, left, right)
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        
        # Identify the opposing player
        opponent = -1 if player == 1 else 1
        
        for dx, dy in directions:
            adjacent_x = end[0] + dx
            adjacent_y = end[1] + dy
            capture_x = end[0] + 2*dx
            capture_y = end[1] + 2*dy

            if 0 <= adjacent_x < 11 and 0 <= adjacent_y < 11 and 0 <= capture_x < 11 and 0 <= capture_y < 11:
                # Capture opposing pieces
                if self.board[adjacent_x, adjacent_y] == opponent:
                    if self.board[capture_x, capture_y] == player or self.board[capture_x, capture_y] == 2 and self.board[capture_x, capture_y] != -2:
                        self.board[adjacent_x, adjacent_y] = 0
                        return 2

        # If black moved, check if the king is fully surrounded
        if player == 1:
            king_pos = np.argwhere(self.board == -2)
            if king_pos.size:
                directions_masked = self.mask_directions(king_pos[0], directions)
                kx, ky = king_pos[0]
                surrounded = True
                for dx, dy in directions_masked:
                    nx, ny = kx + dx, ky + dy
                    if not (0 <= nx < 11 and 0 <= ny < 11 and self.board[nx, ny] == 1):
                        surrounded = False
                        break
                if surrounded:
                    self.print_board()
                    self.board[kx, ky] = 0  # Remove king
                    print("king dead")
                    # signal king death
                    return 1
        # signal to the game that the king is not captured
        return -1
             
    def get_possible_moves_king(self, player_position = (0,0)):
        # the king can move to any free space that is not occupied by another player and is in the same row or column
        # the king can move to the edge of the board(marked with 2 in the board)
        # implement the king movement
        moves = []
        # check the left
        for i in range(player_position[0] - 1, -1, -1):
            if self.board[i][player_position[1]] == 0:
                moves.append((i, player_position[1]))
            elif self.board[i][player_position[1]] == 2:
                moves.append((i, player_position[1]))
                break
            else:
                break
        # check the right
        for i in range(player_position[0] + 1, 11):
            if self.board[i][player_position[1]] == 0:
                moves.append((i, player_position[1]))
            elif self.board[i][player_position[1]] == 2:
                moves.append((i, player_position[1]))
                break
            else:
                break
        # check the column
        # check the up
        for i in range(player_position[1] - 1, -1, -1):
            if self.board[player_position[0]][i] == 0:
                moves.append((player_position[0], i))
            elif self.board[player_position[0]][i] == 2:
                moves.append((player_position[0], i))
                break
            else:
                break
        # check the down
        for i in range(player_position[1] + 1, 11):
            if self.board[player_position[0]][i] == 0:
                moves.append((player_position[0], i))
            elif self.board[player_position[0]][i] == 2:
                moves.append((player_position[0], i))
                break
            else:
                break
        return moves
       
        
    def get_possible_moves(self, player, player_position = (0,0)):
        moves = []
        # moves dict where for each player position there is a list of possible moves
        moves_dict = {}
        # the player is the type  and the player position is the position of the player in question
        # the player can move to any free space that is not occupied by another player and is in the same row or column
        # the all players but the king can't move to the edge of the board(marked with 2 in the board)
        # the player cannot move to a space occupied by another player and all the spaces that follow that player in the same row or column
        
        # 
        if player not in [1, -1,-2]:
            print(player)
            return moves
        if player == -2:
            return self.get_possible_moves_king(player_position)

        if (player_position != (0,0)):
            for i in range(player_position[0] - 1, -1, -1):
                if self.board[i][player_position[1]] == 0:
                    moves.append((i, player_position[1]))
                elif self.board[i][player_position[1]] == player or self.board[i][player_position[1]] == 2:  # Changed line
                    break
                else:
                    break
            # check the right
            for i in range(player_position[0] + 1, 11):
                if self.board[i][player_position[1]] == 0:
                    moves.append((i, player_position[1]))
                elif self.board[i][player_position[1]] == player or self.board[i][player_position[1]] == 2:  # Changed line
                    break
                else:
                    break
            # check the column
            # check the up
            for i in range(player_position[1] - 1, -1, -1):
                if self.board[player_position[0]][i] == 0:
                    moves.append((player_position[0], i))
                elif self.board[player_position[0]][i] == player or self.board[player_position[0]][i] == 2:  # Changed line
                    break
                else:
                    break
            # check the down
            for i in range(player_position[1] + 1, 11):
                if self.board[player_position[0]][i] == 0:
                    moves.append((player_position[0], i))
                elif self.board[player_position[0]][i] == player or self.board[player_position[0]][i] == 2:  # Changed line
                    break
                else:
                    break
            
            return moves
        else:
            # get all the positions of the player with the player type
            player_positions = np.argwhere(self.board == player)
            
            for position in player_positions:
                moves_dict[position[0], position[1]] = []
                moves_dict[position[0],position[1]] += self.get_possible_moves(player, (position[0], position[1]))
            return moves_dict
    
    def get_possible_moves_train(self, player):
        moves_dict = {}
         # get all the positions of the player with the player type
        player_positions = np.argwhere(self.board == player)
        
        for position in player_positions:
            moves_dict[position[0], position[1]] = []
            moves_dict[position[0],position[1]] += self.get_possible_moves(player, (position[0], position[1]))
        # if the player is white, then also add the possible moves for the king
        if player == -1:
            king_positions = np.argwhere(self.board == -2)
            # if the king is on the board
            if (king_positions.size):
                #print(king_positions)
                moves_dict[king_positions[0][0], king_positions[0][1]] = []
                moves_dict[king_positions[0][0],king_positions[0][1]] += self.get_possible_moves(-2, (king_positions[0][0], king_positions[0][1]))
            
                
        return moves_dict
        
    
    def move(self, player, start, end):
        if self.board[start[0], start[1]] == -2:
            if (self.board[end[0], end[1]] == 2):
                self.board[start[0], start[1]] = 0
                self.board[end[0], end[1]] = -2
                return True
            
        if (self.board[end[0], end[1]] != 0):
            return False
        if self.board[start[0], start[1]] == player or (self.board[start[0], start[1]] == -2 and player == -1):
            # check if the move with the player in the start position is possible
            if not (end in self.get_possible_moves(player, start)):
                return False
        else:
            return False
        # move the player
        if (self.board[start[0], start[1]] == -2):
            self.board[start[0], start[1]] = 0
            self.board[end[0], end[1]] = -2
            return True
        self.board[start[0], start[1]] = 0
        self.board[end[0], end[1]] = player
        return True
        
    def is_game_over(self):
        return self.check_winner() != 0     
    
    def get_board(self):
        return self.board
    def reset_board(self):
        self.board = np.array([[2,0,0,1,1,1,1,1,0,0,2],
                              [0,0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0],
                              [1,0,0,0,0,-1,0,0,0,0,1],
                              [1,0,0,0,-1,-1,-1,0,0,0,1],
                              [1,1,0,-1,-1,-2,-1,-1,0,1,1],
                              [1,0,0,0,-1,-1,-1,0,0,0,1],
                              [1,0,0,0,0,-1,0,0,0,0,1],
                              [0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0,0],
                              [2,0,0,1,1,1,1,1,0,0,2]])
    def print_amount(self):
        one = len(np.argwhere(self.board != 0))
        two = len(np.argwhere(self.board == 2))
        print("The amount of pieces on the board is", one - two)