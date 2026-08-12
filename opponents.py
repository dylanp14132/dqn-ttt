import random
import numpy as np
from functools import lru_cache

def _check_win(board, player):  # TODO : Could be made simpler
    for i in range(3):
        if (board[i][0] == board[i][1] == board[i][2]) and (board[i][0] == player):
            return True
        if (board[0][i] == board[1][i] == board[2][i]) and (board[0][i] == player):
            return True

    if (board[0][0] == board[1][1] == board[2][2]) and (board[0][0] == player):
        return True
    if (board[0][2] == board[1][1] == board[2][0]) and (board[0][2] == player):
        return True
    else:
        return False


def random_bot (observation, legal_actions):
    return int(random.choice(legal_actions))


@lru_cache(maxsize=None)
def _minimax(board_tuple, player=1):
    board = np.array(board_tuple).reshape(3, 3)
    if _check_win(board, 1):
        return 1, None
    if _check_win(board, -1):
        return -1, None

    legal_moves = np.flatnonzero(board == 0)
    if len(legal_moves) == 0:
        return 0,None

    best_score = -2 if player == 1 else 2
    best_moves = [legal_moves[0]]

    for a in legal_moves:
        r, c = divmod(a, 3)
        board[r][c] = player
        score, _ = _minimax(tuple(board.flatten()), -player)
        board[r][c] = 0
        if player == 1 and score > best_score:
            best_score, best_moves = score, [a]
        elif player == -1 and score < best_score:
            best_score, best_moves = score, [a]
        elif best_score == score:
            best_moves.append(a)

    return best_score, random.choice(best_moves)


def minimax_bot(observation, legal_actions):
    if random.random() < 0.1: # Chance for a random move
        return random.choice(legal_actions)
    _, move = _minimax(tuple(observation.flatten().astype(int)))
    return int(move)