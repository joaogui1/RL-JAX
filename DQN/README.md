# DQN

Deep Q Networks is one of the first truly functional value-based Deep Reinforcement Learning Algorithms.
Its main ideas are:

* Q-learning loss
* Deep Neural Network for value function approximation ("Neural Q-learning")
* Use a Replay Buffer to sample from past experiences and avoid overfitting
* Use a target network that is updated exporadically to stabilize training

In this folder we have 3 implementations:

* [Pure Flax DQN](dqn_flax.ipynb): Contains an implementation using only flax and JAX
* [Rlax + Flax DQN](dqn_flax_rlax.ipynb): Contains an implementation using flax and rlax for DRL utils
* [Rlax + Haiky DQN](dqn_haiku_rlax_.ipynb): Contains an implementation using haiku and rlax for DRL utils
