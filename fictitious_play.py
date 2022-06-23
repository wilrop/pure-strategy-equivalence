import numpy as np

from Player import FPPlayer
from best_response import verify_nash


def fictitious_play(monfg, u_tpl, epsilon=0, max_iter=1000, init_joint_strategy=None, variant='alternating',
                    global_opt=False, verify=True, early_stop=None, seed=None):
    """Execute the fictitious play algorithm on a given MONFG and utility functions.

    There are two variants of the fictitious play algorithm implemented, simultaneous and alternating fictitious play.
    These variants are not equivalent in general. In the simultaneous variant, all players calculate their best-response
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
        variant (str, optional): The variant to use, which is either simultaneous or alternating.
            (Default value = 'alternating')
        global_opt (bool, optional): Whether to use a global optimiser or a local one. (Default value = False)
        verify (bool, optional): Verify if a converged joint strategy is a Nash equilibrium. When set to true, this uses
            a global optimiser and might be computationally expensive. (Default value = True)
        early_stop (int, optional): The number of iterations the joint strategy has to be the same to allow an early
            stop. (Default value = None)
        seed (int, optional): The initial seed for the random number generator. (Default value = None)

    Returns:
        Tuple[bool, List[ndarray]]: Whether or not we reached a Nash equilibrium and the final joint strategy.

    """
    rng = np.random.default_rng(seed=seed)

    def simultaneous_variant(joint_strategy):
        """Execute one iteration of the simultaneous fictitious play variant.

        Args:
            joint_strategy (List[ndarray]): The current joint strategy.

        Returns:
            Tuple[bool, List[ndarray]]: Whether the policies have converged and the new joint strategy.

        """
        converged = True
        actions = []

        for player in players:  # Collect actions.
            actions.append(player.select_action())

        for player in players:  # Update the empirical state distributions.
            player.update_empirical_strategies(actions)

        for player_id, player in enumerate(players):  # Update the policies simultaneously.
            done, br = player.update_strategy(epsilon=epsilon, global_opt=global_opt)
            joint_strategy[player_id] = br  # Update the joint strategy.
            converged = False

        return converged, joint_strategy

    def alternating_variant(joint_strategy):
        """Execute one iteration of the alternating fictitious play variant.

        Args:
            joint_strategy (List[ndarray]): The current joint strategy.

        Returns:
            Tuple[bool, List[ndarray]]: Whether the policies have converged and the new joint strategy.

        """
        converged = True

        for player_id, player in enumerate(players):  # Loop once over each player to update with alternating.
            actions = []

            for action_player in players:  # Collect actions.
                actions.append(action_player.select_action())

            for update_player in players:  # Update the empirical state distributions.
                update_player.update_empirical_strategies(actions)

            done, br = player.update_strategy(epsilon=epsilon,
                                              global_opt=global_opt)  # Update the current player's policy.
            joint_strategy[player_id] = br  # Update the joint strategy.
            converged = False

        return converged, joint_strategy

    player_actions = monfg[0].shape[:-1]  # Get the number of actions available to each player.
    players = []  # A list to hold all the players.
    joint_strategy = []  # A list to hold the current joint strategy.

    for player_id, u in enumerate(u_tpl):  # Loop over all players to create a new FPAgent object.
        payoff_matrix = monfg[player_id]
        init_strategy = None
        if init_joint_strategy is not None:
            init_strategy = init_joint_strategy[player_id]
        player = FPPlayer(player_id, u, player_actions, payoff_matrix, init_strategy=init_strategy, rng=rng)
        players.append(player)
        joint_strategy.append(player.strategy)

    nash_equilibrium = False  # The current joint strategy is not known to be a Nash equilibrium at this point.
    if early_stop is None:
        early_stop = max_iter
    num_same = 0
    log = [[0] + [item for strat in joint_strategy for item in strat]]

    if variant == 'simultaneous':
        execute_iteration = simultaneous_variant
    else:
        execute_iteration = alternating_variant

    for i in range(max_iter):
        if num_same >= early_stop:
            break

        converged, joint_strategy = execute_iteration(joint_strategy)
        record = [i + 1] + [item for strat in joint_strategy for item in strat]
        log.append(record)

        if converged:  # If FP converged, check if we can guarantee a Nash equilibrium.
            num_same += 1

    if verify:  # Check if the user wanted to verify.
        nash_equilibrium = verify_nash(monfg, u_tpl, joint_strategy, epsilon=epsilon)

    return nash_equilibrium, joint_strategy, log
