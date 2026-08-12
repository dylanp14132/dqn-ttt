import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        self.winner = 0
        self.game_over = False


    def reset(self):
        self.board[:] = 0
        self.current_player = 1
        self.winner = 0
        self.game_over = False


    def get_observation(self):
        return self.board * self.current_player


    def step(self, action):
        if self.game_over:
            raise RuntimeError('Game over')

        r,c = divmod(action, 3)

        if self.board[r,c] != 0:
            raise RuntimeError('Invalid move')

        self.board[r,c] = self.current_player

        if self._check_win(self.current_player):
            self.game_over = True
            self.winner = self.current_player
            reward = 1.0
        elif not np.any(self.board == 0):
            self.game_over = True
            self.winner = 0
            reward = 0.0
        else:
            reward = 0.0

        if not self.game_over:
            self.current_player *= -1

        return reward, self.game_over


    def legal_moves(self):
        return np.flatnonzero(self.board == 0)


    def _check_win(self, player): # TODO : Could be made simpler
        for i in range(3):
            if (self.board[i][0] == self.board[i][1] == self.board[i][2]) and (self.board[i][0] == player):
                return True
            if (self.board[0][i] == self.board[1][i] == self.board[2][i]) and (self.board[0][i] == player):
                return True

        if (self.board[0][0] == self.board[1][1] == self.board[2][2]) and (self.board[0][0] == player):
            return True
        if (self.board[0][2] == self.board[1][1] == self.board[2][0]) and (self.board[0][2] == player):
            return True
        else:
            return False


    def get_input(self, row, col):
        action = np.zeros_like(self.board)
        action[row][col] = self.current_player
        self.step(action)