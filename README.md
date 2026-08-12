# Tic-Tac-Toe DQN

A small deep Q-learning agent that learns to play tic-tac-toe from scratch 
and trains against a minimax bot with a 10% chance of a random move.

## Files
 
| File            | Purpose                                                          |
|-----------------|-------------------------------------------------------------------|
| `game.py`       | Tic-tac-toe environment (board, legal moves, win/draw detection) |
| `agent.py`      | Q-network and the agent's act / train step                       |
| `opponents.py`  | `random_bot` and `minimax_bot` (used for the curriculum and eval) |
| `train.py`      | Training loop, replay buffer, evaluation, and result plotting     |

## Running
 
```bash
uv run train.py
```
 
This runs training episodes, printing win/draw/loss rates against the 
minimax bot, and plots the results.

## Configuration
 
Training parameters can be adjusted via `train()`'s arguments in `train.py`, 
e.g.:
 
```python
train(num_episodes=30000, batch_size=64, start_training_after=500, eval_every=2000)
```
