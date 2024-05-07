# Cellware: A Multi-Agent Reinforcement Learning Approach for Simulating Primitive Cellular Behaviour

## Table of Contents

- [About](#about)
- [Key Features](#key_features)
- [Initial Environment](#initial_environment)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Contact Information](#contact_information)
- [License](#license)

## About
Cellware is a multi-agent reinforcement learning (MARL) framework designed to model high-level cellular behaviour. Each individual agent is structured based on a Proximal Policy Optimization (PPO) architecture. In a targeted environment represented by an n×m matrix, agents aim to collect high-reward objects associated with various metabolic traits. While collecting specific number of objects, the agents undergo reproduction by passing their neural weights to the new generation. The collection procedure involves selecting optimal action to transitions the agent’s state to the object state using an actor network. The critic network’s goal is to enhance reward prediction accuracy within the observation space. 

 This process occurs in two distinct settings: 

Selective setting: 

Newly generated agents inherit the neural weights of the fastest agent 

Differentiation Setting (Cellfate): 

A higher-level network, referred as Cellfate, aim for enhancing policy differentiation among agents. 

A separate pair of actor-critic networks is responsible of selecting the best action according to policy differentiation across generations of agents. 

Additionally, Cellware introduces a memory sharing option that allows parallel agents to share their action and observation buffers. This scalability efficiently handles varying agent numbers and policy depth, making Cellware suitable for increasingly complex use cases. Due to its robustness in optimizing the best policy for specific task Cellware also demonstrates promising generalizability across multiple environments. 


## Key Features
1. Policy Options: Cellware supports three different policies: 

Discrete 

Gaussian Continuous 

Discrete RNN (LSTM or GRU) 

2. Custom Environment: The environment is custom-designed to resemble an in-vitro cell environment in a primitive format. 

3. Delayed Gratification: An additional GRU layer serves as an encoder to emphasize delayed gratification. 

4. Parallel Agents: The agents are distributed by Ray parallel architecture. The number of parallel agents can be arbitrarily scaled, limited only by available resources. The --collective_switch parameter facilitates memory sharing across agents, making data collection faster by parallelizing the process. In the shared environment, memory sharing also implies knowledge transfer, thereby boosting convergence. 

5. GPU support: The architecture allows computation device switch for the neural networks, with mixed precision training. 

6. Cellfate Optimization: A higher-level network optimizes the right policy to use by the agents. The cellfate network aims to maximize collected rewards per generation by selecting the best policy for the agents.
   
## Initial Environment: 

2D matrix (10×10) 

Agents start at positions (0,0) and (1,0) 

Action space: 5 (0=UP, 1=LEFT, 2=DOWN, 3=RIGHT, 4=SWAP) 

SWAP action toggles between vertical and horizontal axes 

Reward map resembles a labyrinth with positive rewards, walls (denoted by -1), and delayed gratification challenges. 

## Getting Started
### Up to Date
The Up to date version of the solution can be found in folder: PPO_universal
### Prerequisites

- [Python](https://www.python.org/) (version 3.8.13)

### Installation

1. Create virtual environment
   
   conda create -n cellware python>=3.8

3. Activate virtual environment
   
   activate cellware
  
5. Clone the repository:

   ```bash
   git clone https://github.com/krenusz/cellware/[foldername].git
   
6. Install requirements using pip
     
   pip install -r requirements.txt
   
### Usage

Change to the working directory: cd ...

Run the program: python PPO_[version]_main.py

version: continous; universal; evolution network (up to date) 

## Contact Information
bence.krenusz@gmail.com
https://www.linkedin.com/in/kr%C3%A9nusz-bence-85ba6512a/

## License

This project is licensed under the MIT License - see the LICENSE file for details.
