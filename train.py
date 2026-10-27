import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from game import TicTacToe
from agent import Agent
from opponents import minimax_bot

plt.ion()


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push_episode(self, transitions):
        self.buffer.extend(transitions)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        observation, action, returns = zip(*batch)
        return np.array(observation), np.array(action), np.array(returns)

    def __len__(self):
        return len(self.buffer)


def plot_results(wins, draws, losses, eps):
    plt.clf()
    plt.gcf().canvas.manager.set_window_title("Training results")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.title("Agent vs minimax bot")
    plt.ylim(0, 1)
    plt.plot(eps, wins, label='win rate')
    plt.plot(eps, draws, label='draw rate')
    plt.plot(eps, losses, label='loss rate')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.pause(.1)


def plot_final_results(wins, draws, losses, eps):
    plt.ioff()
    plt.clf()
    plt.gcf().canvas.manager.set_window_title("Training results")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.title("Agent vs minimax bot")
    plt.ylim(0, 1)
    plt.plot(eps, wins, label='win rate')
    plt.plot(eps, draws, label='draw rate')
    plt.plot(eps, losses, label='loss rate')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def play_episode(game, agent, epsilon, opponent=None):
    game.reset()
    agent_moves = []

    while not game.game_over:
        observation = game.get_observation()
        legal_moves = game.legal_moves()
        is_agent_turn = opponent is None or game.current_player == 1
        if is_agent_turn:
            action = agent.act(observation, legal_moves, epsilon)
            agent_moves.append((observation.copy(), action, game.current_player))
        else:
            action = opponent(observation, legal_moves)
        game.step(action)

    winner = game.winner
    transitions = []
    for observation, action, player in agent_moves:
        if winner == 0:
            reward = 0
        elif winner == player:
            reward = 1.0
        else:
            reward = -1.0
        transitions.append((observation.flatten(), action, reward))
    return transitions, winner


def evaluate(game, agent, opponent, n=200):
    wins = losses = draws = 0

    for i in range(n):
        game.reset()

        while not game.game_over:
            observation = game.get_observation()
            legal_moves = game.legal_moves()

            if game.current_player == 1:
                action = agent.act(observation, legal_moves, epsilon=0.0)
            else:
                action = opponent(observation, legal_moves)

            game.step(action)

        if game.winner == 1:
            wins += 1
        elif game.winner == 0:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses


def train(num_episodes=10000, batch_size=64, start_training_after=500, eval_every=50, self_play_after=5000):
    print("Starting...")

    game = TicTacToe()
    agent = Agent()
    buffer = ReplayBuffer()

    wins, draws, losses, eps = [], [], [], []

    epsilon_max, epsilon_min, epsilon_decay = 1.0, 0.05, 5000

    for ep in range(num_episodes):
        epsilon = epsilon_min + (epsilon_max - epsilon_min) * max(0, 1 - ep / epsilon_decay)
        #opponent = None if ep >= self_play_after else random_bot
        opponent = minimax_bot

        transitions, _ = play_episode(game, agent, epsilon, opponent)
        buffer.push_episode(transitions)

        if len(buffer) >= start_training_after:
            observation, actions, returns = buffer.sample(batch_size)
            agent.train_step(observation, actions, returns)

        if ep % eval_every == 0 and ep > 0:
            win, draw, loss = evaluate(game, agent, minimax_bot, n=200)
            wins.append(win / 200), draws.append(draw / 200), losses.append(loss / 200), eps.append(ep)
            plot_results(wins, draws, losses, eps)
            print(f"ep {ep:6d} epsilon={epsilon:.3f} vs minimax -> "
                  f"win {win / 200:.2f} draw {draw / 200:.2f} loss {loss / 200:.2f}")

    return wins, draws, losses, eps


if __name__ == '__main__':
    w, d, l, e = train()
    plot_final_results(w, d, l, e)
