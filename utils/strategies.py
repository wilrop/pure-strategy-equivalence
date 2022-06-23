from itertools import chain, combinations

import numpy as np


def make_strat_from_action(action, num_actions):
    """Turn an action into a strategy representation.

    Args:
        action (int): An action.
        num_actions (int): The number of possible actions.

    Returns:
        ndarray: A pure strategy as a numpy array.

    """
    strat = np.zeros(num_actions)
    strat[action] = 1
    return strat


def make_joint_strat(player_id, player_strat, opp_strat):
    """Make joint strategy from the opponent strategy and player strategy.

    Args:
        player_id (int): The id of the player.
        player_strat (ndarray): The strategy of the player.
        opp_strat (List[ndarray]): A list of the strategies of all other players.

    Returns:
        List[ndarray]: A list of strategies with the player's strategy at the correct index.

    """
    opp_strat.insert(player_id, player_strat)
    return opp_strat


def make_joint_strat_from_flat(flat_strat, player_actions):
    """Make a joint strategy from a flat joint strategy.

    Args:
        flat_strat (ndarray): A joint strategy as a flat array.
        player_actions (Tuple[int]): A tuple with the number of actions per player.

    Returns:
        List[ndarray]: A list of individual strategies.
    """
    curr = 0
    joint_strat = []

    for num_actions in player_actions:
        start = curr
        curr = curr + num_actions
        strat = flat_strat[start:curr]
        joint_strat.append(strat)

    return joint_strat


def normalise_strat(strat):
    """Normalise a strategy to sum to one.

    Args:
        strat (ndarray): A strategy array.

    Returns:
        ndarray: The same strategy as a probability vector.
    """
    total = np.sum(strat)
    if total > 0:
        norm_strat = strat / np.sum(strat)
    else:
        num_actions = len(strat)
        norm_strat = np.full(num_actions, 1 / num_actions)
    return norm_strat


def normalise_joint_strat(joint_strat):
    """Normalise all individual strategies in a joint strategy.

    Args:
        joint_strat (List[ndarray]): A list of individual strategies.

    Returns:
        List[ndarray]: A joint strategy with each individual strategy normalised.
    """
    norm_joint_strat = []

    for strat in joint_strat:
        norm_joint_strat.append(normalise_strat(strat))

    return norm_joint_strat


def get_support(strat, tol=1e-15):
    """Get the actions which are in the support of a strategy.

    Args:
        strat (ndarray): A strategy array.
        tol: The tolerance to count action probabilities still as zero.

    Returns:
        ndarray: An array of actions that are in the support.

    """
    return np.flatnonzero(strat > tol)


def get_non_support(strat, tol=1e-15):
    """Get the actions which are not in the support.

    Args:
        strat (ndarray): A strategy array.
        tol (float, optional): The tolerance to count action probabilities still as zero. (Default value = 1e-15)

    Returns:
        List[int]: A support of the actions which were not in the input support.
    """
    return np.nonzero(strat < tol)


def supports_diff(support1, support2):
    """Take the difference of two supports, defined as support1 \ support2.

    Args:
        support1 (List[int]): The first support.
        support2 (List[int]): The second support.

    Returns:
        List[int]: A support of the actions which were not in the input support.
    """
    return tuple(set(support1) - set(support2))


def enumerate_supports(num_actions, min_size=1, max_size=None):
    """Enumerate the set of all supports, with a minimum and maximum size, for a number of actions.

    Args:
        num_actions (int): The number of possible actions for a player.
        min_size (int, optional): The minimum size of each support. (Default = 1)
        max_size (int, optional): The maximum size of each support. (Default = None)

    Returns:
        List[Tuple[int]]: A list of supports.
    """
    actions = np.arange(num_actions)
    if max_size is None:
        max_size = num_actions
    return chain.from_iterable(combinations(actions, sup_size) for sup_size in range(min_size, max_size + 1))


def expand_support(support, num_actions):
    """Return all possible supports which also include a base support.

    Args:
        support (ndarray): A base support.
        num_actions (int): The number of actions available to choose from.

    Returns:
        List[ndarray]: A list of expanded supports.
    """
    non_support = np.delete(np.arange(num_actions), support)
    expanded_supports = []

    for num_extra in range(1, len(non_support) + 1):
        expansions = list(combinations(non_support, num_extra))

        for expansion in expansions:
            expansion = np.array(expansion)
            insert_idx = np.searchsorted(support, expansion)
            expanded_support = np.insert(support, insert_idx, expansion)
            expanded_supports.append(expanded_support)

    return expanded_supports


def expand_support_non_support(sup_non_sup):
    """Return all possible support-non support tuples which also include a base support.

    Args:
        sup_non_sup (Tuple[Tuple[int]]): A base support-non support.

    Returns:
        List[Tuple[Tuple[int]]]: A list of expanded support and non-supports.
    """
    support, non_support = sup_non_sup
    expanded_sup_non_sups = []

    for num_extra in range(1, len(non_support) + 1):
        expansions = list(combinations(non_support, num_extra))

        for expansion in expansions:
            expansion = np.array(expansion)
            insert_idx = np.searchsorted(support, expansion)
            expanded_support = tuple(np.insert(support, insert_idx, expansion))
            reduced_non_support = tuple(np.delete(expansion))
            expanded_sup_non_sups.append((expanded_support, reduced_non_support))

    return expanded_sup_non_sups


def totally_mixed_supports(player_actions):
    """Generate the totally mixed supports.

    Note:
        Totally mixed supports are supports which assign a positive probability to each action. In this case, it means
        that a dictionary is returned with for every player a tuple of all their actions.

    Args:
        player_actions (Tuple[int]): A tuple of actions in the support.

    Returns:
        Dict{int: Tuple[int]}: A dictionary of the totally mixed supports for each player.
    """
    full_suports = {}

    for player, num_actions in enumerate(player_actions):
        full_suports[player] = tuple(range(num_actions))

    return full_suports
