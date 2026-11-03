# Deep Q-Learning - Atari Breakout

This project implements Deep Q-Learning using `keras-rl2`, `keras`, and `gymnasium` to train an agent to play Atari's Breakout.

## Requirements
- Ubuntu 20.04 LTS
- Python 3.9
- Gymnasium 0.29.1
- Keras 2.15.0
- Keras-rl2 1.0.4
- NumPy 1.25.2

## Files
- `train.py`: Trains the DQN agent on Breakout and saves weights to `policy.h5`.
- `play.py`: Loads `policy.h5` and visualizes the trained agent playing Breakout.
