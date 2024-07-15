import numpy as np

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: positions to be encoded, size (B, M) where B is batch size and M is number of positions
    returns: (B, M, D)
    """
    
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)
    
    out = np.einsum('bm,d->bmd', pos, omega)  # (B, M, D/2), outer product

    emb_sin = np.sin(out)  # (B, M, D/2)
    emb_cos = np.cos(out)  # (B, M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=2)  # (B, M, D)
    return emb

def get_1d_sincos_reward_embed_from_value(embed_dim, rewards, max_reward):
    """
    embed_dim: output dimension for each reward
    rewards: reward values to be encoded, size (B, M) where B is batch size and M is minibatch size
    max_reward: the maximum possible reward
    returns: (B, M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)
    
    pos = rewards / max_reward  # Normalize rewards to [0, 1]
    out = np.einsum('bm,d->bmd', pos, omega)  # (B, M, D/2), outer product

    emb_sin = np.sin(out)  # (B, M, D/2)
    emb_cos = np.cos(out)  # (B, M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=2)  # (B, M, D)
    return emb

def get_1d_sincos_action_embed_from_index(embed_dim, actions, num_actions):
    """
    embed_dim: output dimension for each action
    actions: indices of actions to be encoded, size (B, M) where B is batch size and M is minibatch size
    num_actions: total number of actions
    returns: (B, M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)
    
    pos = actions / num_actions  
    
    out = np.einsum('bm,d->bmd', pos, omega)  # (B, M, D/2), outer product
    
    emb_sin = np.sin(out)  # (B, M, D/2)
    emb_cos = np.cos(out)  # (B, M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=2)  # (B, M, D)
    return emb

def create_energy_matrix_with_positional_embedding(state, size, embed_dim):
    """
    state: the state input with shape (B, M, 2*pos_len_1d)
    size: size of the grid
    embed_dim: embedding dimension
    returns: positional embedding with shape (B, M, embed_dim*pos_len_1d)
    """
    B, M, _ = state.shape
    pos_len_1d = state.shape[-1] // 2
    
    max_distance = 2 * (size - 1)  # Maximum Manhattan distance in the matrix
    pos_embed = []
    
    for i in range(pos_len_1d):
        x, y = state[:, :, i], state[:, :, i+pos_len_1d]
        manhattan_distance = x + y
        normalized_distance = manhattan_distance / max_distance
        pos_embed.append(get_1d_sincos_pos_embed_from_grid(embed_dim, normalized_distance))
    
    out = np.concatenate(pos_embed, axis=-1)  # (B, M, embed_dim*pos_len_1d)
    return out

def state_action_reward_embedding(states, actions, rewards, current_states, current_actions=None, current_rewards=None, size=10, embed_dim=8, num_action=5, max_reward=100):
    """
    states: the initial states with shape (B, M, 2*pos_len_1d)
    actions: the actions taken with shape (B, M)
    rewards: the rewards received with shape (B, M)
    current_states: the current states with shape (B, M, 2*pos_len_1d)
    size: size of the grid
    embed_dim: embedding dimension
    num_action: total number of actions
    max_reward: maximum possible reward
    returns: combined embedding with shape (B, M, total_embed_dim)
    """
    pos_emb = create_energy_matrix_with_positional_embedding(states, size, embed_dim)  # (B, M, embed_dim*pos_len_1d)
    action_emb = get_1d_sincos_action_embed_from_index(embed_dim, actions, num_action)  # (B, M, embed_dim)
    reward_emb = get_1d_sincos_reward_embed_from_value(embed_dim, rewards, max_reward)  # (B, M, embed_dim)
    current_state_pos_emb = create_energy_matrix_with_positional_embedding(current_states, size, embed_dim)  # (B, M, embed_dim*pos_len_1d)
    if current_actions is None:
        current_action_emb = np.zeros_like(action_emb)
        current_reward_emb = np.zeros_like(reward_emb)
    else:
        current_action_emb = get_1d_sincos_action_embed_from_index(embed_dim, current_actions, num_action)  # (B, M, embed_dim)
        current_reward_emb = get_1d_sincos_reward_embed_from_value(embed_dim, current_rewards, max_reward)  # (B, M, embed_dim)

    return np.concatenate([pos_emb, action_emb, reward_emb, current_state_pos_emb,current_action_emb,current_reward_emb], axis=2)  # (B, M, total_embed_dim)

def embedding_to_energy(matrix):
    size = matrix.shape[0]
    energy_matrix = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            energy_matrix[i, j] = matrix[i, j].sum() # Using summary of all embedding values as the energy level
    # Normalize the energy matrix to be between 0 and 1
    energy_matrix = (energy_matrix - energy_matrix.min())/ (energy_matrix.max() - energy_matrix.min())
    return energy_matrix
