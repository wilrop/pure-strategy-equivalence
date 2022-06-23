import numpy as np


def identity_game(player_actions):
    """Generate an identity game.

    Args:
        player_actions (Tuple[int]): A tuple of actions indexed by player.

    Returns:
        List[ndarray]: A list of payoff matrices representing the identity game.

    """
    payoffs = []
    joint_strat_length = np.sum(player_actions)  # Description length of a joint strategy.
    payoffs_shape = player_actions + tuple([joint_strat_length])  # Shape of the payoff matrices.
    payoff_matrix = np.zeros(payoffs_shape)  # Make the same payoff matrix for every player.

    for joint_strat in np.ndindex(player_actions):  # Loop over joint strategies.
        identity_vec = []  # Initialise the identity payoff. One hot encode joint strategies in this variable.
        for player, action in enumerate(joint_strat):  # One hot encode each player's strategy.
            strat_vec = np.zeros(player_actions[player])
            strat_vec[action] = 1
            identity_vec.extend(list(strat_vec))
        payoff_matrix[joint_strat] = np.array(identity_vec)

    payoffs.append(payoff_matrix)

    for _ in range(len(player_actions) - 1):  # We already have the first payoff matrix, so copy the rest now.
        payoff_copy = np.copy(payoff_matrix)
        payoffs.append(payoff_copy)

    return payoffs