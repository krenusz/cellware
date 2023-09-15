# 
Repository for the multi agent ppo for primitive simulation of cellular competition.

PPO model source: https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char07%20PPO/PPO_MountainCar-v0.py

# Cellware

Primitive cell behaviour simulation.

## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About

Multi-agental Reinforcement Learning solution aiming to reproduce highlevel cellular behaviour. Every single agent is built up by a Proximal Policy Optimization network. In a given environment (n*m matrix)  the agents trying to collect high reward objects representing the metabolic traits. After a certain amount of collected objects the fastest agent gets to multiply. The new agents inherits the fastest agents neural weights. The multiplication only happens if a single agents collect certain amount of objects. The collection represented as choosing the right action to change the state to the object state.

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
   git clone https://github.com/krenusz/cellware.git
6. Install requirements using pip
     
   pip install -r requirements.txt
### Usage

Change to the working directory: cd ...

Run the program: python ray_parallel_PPO.py
