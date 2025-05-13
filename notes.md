# Half Cheetah 

 Ill be using this repo to keep track of all of the different implementations and their pros and cons as well as my notes and research 

## What is the actor_impl doing
This code implements an Actor neural network for algorithms like SAC (Soft Actor-Critic) designed for continuous control environments like HalfCheetah:

Creates a stochastic policy network with 2 hidden layers (256 neurons each)
Takes environment observations as input
Outputs Gaussian distribution parameters (mean and log standard deviation) for actions
Uses the reparameterization trick for backpropagation through random sampling
Applies tanh to bound actions within the environment's range
Returns:

Sampled actions for exploration
Log probabilities for policy gradient updates
Mean actions for deterministic evaluation



The key feature is generating both actions and their probabilities, essential for modern policy gradient methods that balance exploration and exploitation through entropy regularization.

## Log Prob


