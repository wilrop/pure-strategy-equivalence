import os

import numpy as np


def create_game_path(content, experiment, game, parent_dir=None, mkdir=True):
    """Create a new directory based on the given parameters.

    Args:
        content (str): The type of content this directory will hold. Most often this is either 'data' or 'plots'.
        experiment (str): The name of the experiment that is being performed.
        game (str): The game that is experimented on.
        parent_dir (str, optional): Parent directory for data and plots. (Default value = None)
        mkdir (bool, optional): Whether to create the directory. (Default value = True)

    Returns:
        str: The path that was created.

    """
    if parent_dir is None:
        parent_dir = '.'
    path = f'{parent_dir}/{content}/{experiment}/{game}'
    if mkdir:
        os.makedirs(path, exist_ok=True)

    return path


def get_payoffs(actions, payoff_matrices):
    """Get the payoffs from the payoff matrices from the selected actions.

    Args:
        actions (List[int]): The actions taken by each player.
        payoff_matrices (List[ndarray]): A list of payoff matrices.

    Returns:
        List[ndarray]: A list of received payoffs.

    """
    actions = tuple(actions)
    return list(map(lambda x: x[actions], payoff_matrices))


def calc_returns(payoffs_dict, agents, rollouts):
    """Calculate the scalarised expected returns for each agent.

    Args:
        payoffs_dict (Dict[int]: List[ndarray]): The vectorial payoffs obtained by the agents.
        agents (List[Agent]): A list of agents.
        rollouts (int): The amount of rollouts that were performed.

    Returns:
        List[float]: A list of scalarised expected returns.

    """
    returns = {}
    for ag, payoff_hist in payoffs_dict.items():
        payoff_sum = np.sum(payoff_hist, axis=0)
        avg_payoff = payoff_sum / rollouts
        ser = agents[ag].u(avg_payoff)
        returns[ag] = ser
    return returns


def calc_action_probs(actions_dict, player_actions, rollouts):
    """Calculate empirical action probabilities.

    Args:
        actions_dict (Dict[int]: List[int]): The actions performed by each agent over the rollout period.
        player_actions (Tuple[int]): The number of actions per agent.
        rollouts (int): The number of rollouts.

    Returns:
        List[List[float]]: The action probabilities for each agent.

    """
    all_probs = {}

    for (ag, action_hist), num_actions in zip(actions_dict.items(), player_actions):
        counts = np.bincount(action_hist, minlength=num_actions)
        probs = counts / rollouts
        all_probs[ag] = probs

    return all_probs


def calc_com_probs(commitments, rollouts):
    """Calculate the empirical commitment probabilities.

    Args:
        commitments (List[int | ndarray]): A list of commitments.
        rollouts (int): The number of rollouts.

    Returns:
        List[float]: The commitment probabilities for each agent.

    """
    com = sum(commitment is not None for commitment in commitments)
    no_com = (rollouts - com)
    return [com / rollouts, no_com / rollouts]
