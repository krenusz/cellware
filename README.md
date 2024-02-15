# Cellware

Primitive cell behaviour simulation.

## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## About

Multi-agential Reinforcement Learning solution aiming to reproduce high-level cellular behaviour. Every single agent is built up by a Proximal Policy Optimization network. In a given environment (n*m matrix)  the agents trying to collect high reward objects representing the metabolic traits. After a certain amount of collected objects the fastest agent gets to multiply. The new agents inherits the fastest agents neural weights. The multiplication only happens if a single agents collect certain amount of objects. The collection represented as choosing the right action to change the state to the object state.

## Getting Started
### Up to Date
The Up to date version of the solution can be found in folder: PPO_universal
### Prerequisites

- [Python](https://www.python.org/) (version 3.8.13)

### Installation

1. Create virtual environment
   
   conda create -n cellware python=3.8.13

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
version: continous or universal
### License

This project is licensed under the MIT License - see the LICENSE file for details.
