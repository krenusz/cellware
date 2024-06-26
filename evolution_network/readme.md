# Cellware: Evolution Network

## Table of Contents

- [About](#about)
- [Updates](#updates)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [License](#license)
## About

Multi-agential Reinforcement Learning solution aiming to reproduce high-level cellular behaviour. Every single agent is built up by a Proximal Policy Optimization network. In a given environment (n*m matrix)  the agents trying to collect high reward objects representing the metabolic traits. After a certain amount of collected objects the fastest agent gets to multiply. The new agents inherits the fastest agents neural weights. The multiplication only happens if a single agents collect certain amount of objects. The collection represented as choosing the right action to change the state to the object state.

## Updates
Updated solution supporting 3 different Policies:
- Discrete
- Gaussian Continous
- Discrete RNN (LSTM or GRU)

The Update mechanism and the Policies are based on the codes:
https://github.com/Lizhi-sjtu/DRL-code-pytorch
https://github.com/DLR-RM/stable-baselines3

The environment is Custom desinged to resemble in-vitro cell environment in primitive format.

An additional GRU layer was applied as an Encoder in order to enphasise Delayed Gratification.

Number of parallel Agents can be arbitrarly scaled with the limit of resources. Furthermore a --collective_switch parameter is responsible for sharing memory across Agents.

KL Divergence target was applied for early stopping.

In this solution, a higher level network has been introduced. The purpose of the network is to optimize "cellfate". In this context cellfate refers to the policy that the agents are using. There are three policy options [Discrete RNN PPO, Discrate PPO and Continuous PPO] for the cellfate network to determine for the next generation of agents. The cellfate network tries to maximize the collected rewards per generation by choosing the best policy for the agents in that generation.

Some of the functionalities are not available yet, but it is on it's way.
## Initial Setting

2D matrix (10,10) Environment

      [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

The Agents take two position (size=2) and start from (0,0),(1,0). 
Action space is 5 [0=UP,1=LEFT,2=DOWN,3=RIGHT,4=SWAP]
SWAP: change back and forth between Vertical and horizonal axis [(0,0),(1,0)] -> [(0,0),(0,1)] ; [(0,0),(0,1)] -> [(0,0),(1,0)]
The Agents will navigate through a reward map which in our initial setting is a labirinth.

      [[ 1,  1,  1,  1,  2,  2,  3,  3,  0,  0],
       [ 1,  1,  1,  1,  2,  2, -1,  4, -1, -1],
       [ 1,  1,  1,  1, -1, -1, -1,  4, 50,  0],
       [ 1,  1,  1,  1, -1, -1, -1,  0,  0,  0],
       [-1, -1, -1, -1, -1, -1, -1, -1, -1,  0],
       [ 0, 10, 20, 10,  0,  0, 20, -1, 60,  0],
       [10, 20, 30, 20, 10, 20, 30, -1,  6,  0],
       [20, 30, 90, 30, 20, 10, 20, -1,  6, -1],
       [10, 20, 30, 20, 10,  5,  5, -1,  0,  0],
       [ 0, 10, 20, 10,  0,  0,  0, 70,  0,  0]]
       
Positive numbers and 0 represents the discoverable rewards, while -1 is the wall which the Agents cannot go through. In order to collect the most reward they need to select best actions for a certain situation (some places without SWAP they cannot go through). In order to collect higher rewards later on their jurney they are facing situations where there is no reward for the best action (Delayed Gratification).
      
## Getting Started

### Prerequisites

- [Python](https://www.python.org/) (version 3.8.13)

### Installation

1. Create virtual environment
   
   conda create -n cellware python=3.8.13

3. Activate virtual environment
   
   activate cellware
  
5. Clone the repository:

   ```bash
   git clone https://github.com/krenusz/cellware/evolution_network.git
6. Install requirements using pip
     
   pip install -r requirements.txt
   
7. Create two empty folder named: runs and routes
### Usage

Change to the working directory: cd ...

Run the program: python PPO_universal_main_selfreplicate.py

### License

This project is licensed under the MIT License - see the LICENSE file for details.
