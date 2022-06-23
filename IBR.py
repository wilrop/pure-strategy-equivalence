import copy

import numpy as np
from Player import IBRPlayer
from best_response import verify_nash


def iterated_best_response(monfg, u_tpl, epsilon=0., max_iter=1000, init_joint_strategy=None, variant='alternating',
                           global_opt=False, verify=True, seed=None):
    """Execute the iterated best response algorithm on a given MONFG and utility functions.

    There are two variants of the iterated best response algorithm implemented, a simultaneous and alternating variant.
    These are not equivalent in general. In the simultaneous variant, all players calculate their best-response
    strategy simultaneously. The alternating variant does it by alternating.

    Note:
        At this point in time, the algorithm does not find cycles and will continue to execute until the maximum number
        of iterations is reached.

    Args:
        monfg (List[ndarray]): A list of payoff matrices representing the MONFG.
        u_tpl (Tuple[callable]): A tuple of utility functions.
        epsilon (float, optional): An optional parameter to allow for approximate Nash equilibria. (Default value = 0)
        max_iter (int, optional): The maximum amount of iterations to run IBR for. (Default value = 1000)
        init_joint_strategy (List[ndarray], optional): Initial guess for the joint strategy. (Default value = None)
        variant (str, optional): The variant to use, which is either simultaneous or alternating. (Default value =
            'alternating')
        global_opt (bool, optional): Whether to use a global optimiser or a local one. (Default value = False)
        verify (bool, optional): Verify if a converged joint strategy is a Nash equilibrium. When set to true, this uses
            a global optimiser and might be computationally expensive. (Default value = True)
        seed (int, optional): The initial seed for the random number generator. (Default value = None)

    Returns:
        Tuple[bool, List[ndarray]]: Whether or not we reached a Nash equilibrium and the final joint strategy.

    """
    rng = np.random.default_rng(seed=seed)

    player_actions = monfg[0].shape[:-1]  # Get the number of actions available to each player.
    players = []  # A list to hold all the players.
    joint_strategy = []  # A list to hold the current joint strategy.

    for player, num_actions in enumerate(player_actions):  # Loop over all players to create a new IBRAgent object.
        u = u_tpl[player]
        payoff_matrix = monfg[player]
        init_strategy = None
        if init_joint_strategy is not None:
            init_strategy = init_joint_strategy[player]
        player = IBRPlayer(player, u, num_actions, payoff_matrix, init_strategy=init_strategy, rng=rng)
        players.append(player)
        joint_strategy.append(player.strategy)

    nash_equilibrium = False  # The current joint strategy is not known to be a Nash equilibrium at this point.
    new_joint_strategy = copy.deepcopy(joint_strategy)

    if variant == 'simultaneous':
        def update_strategy():
            """Hide the strategy updates of other players until everyone is finished for the simultaneous update."""
            return joint_strategy
    else:
        def update_strategy():
            """Show the strategy updates of other players for the alternating update."""
            return new_joint_strategy

    for i in range(max_iter):
        converged = True

        for pid, player in enumerate(players):
            done, br = player.update_strategy(update_strategy(), epsilon=epsilon, global_opt=global_opt)
            new_joint_strategy[pid] = br  # Update the joint strategy.
            if not done:
                converged = False

        if converged:  # If IBR converged, check if we can guarantee a Nash equilibrium.
            if global_opt:  # If we used a global optimiser, it is guaranteed to be a Nash equilibrium.
                nash_equilibrium = True
            elif verify:  # Otherwise check if the user wanted to verify.
                nash_equilibrium = verify_nash(monfg, u_tpl, joint_strategy, epsilon=epsilon)
            break
        else:
            joint_strategy = copy.deepcopy(new_joint_strategy)  # Update the joint strategy.

    return nash_equilibrium, joint_strategy
