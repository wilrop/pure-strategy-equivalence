import numpy as np
import scipy.optimize as scopt

from utils.strategies import normalise_strat


def objective(strategy, expected_returns, u):
    """The objective function in an MONFG under SER.

    Implements the objective function for a player in an MONFG under SER. In a best-response calculation, players aim to
    maximise their utility.

    Args:
        strategy (ndarray) The current estimate for the best response strategy.
        expected_returns (ndarray): The expected returns given all other players' strategies.
        u (callable): The utility function of this agent.

    Returns:
        float: The value on the objective with the provided arguments.

    """
    strategy = normalise_strat(strategy)
    expected_vec = strategy @ expected_returns  # The expected vector of the strategy applied to the expected returns.
    utility = u(expected_vec)
    return utility


def optimise_policy(expected_returns, u, epsilon=0, global_opt=False, init_strat=None, guesses=1):
    """Optimise a policy given a utility function.

    When setting ``global_opt=True``, this will optimise the function using the SHGO algorithm. The algorithm is proven
    to converge to the global optimum for the general case where :math:`f(x)` is non-continuous, non-convex and
    non-smooth, when using the default simplicial sampling method [1]. We currently use the non-default sobol sampling
    method as there is a bug in the default method and sobol has shown more reliable in practice.

    When using a local optimiser, the function is only guaranteed to find a local optimum. By default it will use
    Sequential Least Squares Programming (SLSQP).

    References:
        .. [1] Endres, SC, Sandrock, C, Focke, WW (2018) "A simplicial homology algorithm for lipschitz optimization",
            Journal of Global Optimization.

    Args:
        expected_returns (ndarray): The expected returns from the player's actions.
        u (callable): The player's utility function.
        epsilon (float, optional): Allow epsilon approximate solutions. (Default value = 0)
        global_opt (bool, optional): Whether to use a global optimiser or a local one. We use the sampling method
         'sobol' by default as we found better empirical results with it than with 'simplicial'. The drawback is that
         simplicial has much better theoretical convergence guarantees. (Default value = False)
        init_strat (ndarray, optional): An initial guess for the optimal policy. (Default value = None)
        guesses (int, optional): The amount of starting guesses to try. (Default value = 1)

    Returns:
        Tuple[bool, ndarray, float]: Whether the optimisation was successful, the optimised strategy and utility from
        this strategy.

    """
    num_actions = len(expected_returns)
    bounds = [(0, 1)] * num_actions  # Constrain probabilities to 0 and 1.
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Equality constraint is equal to zero by default.

    if global_opt:
        options = {}
        if epsilon > 0:
            # Set a tolerance for the global optimizer. Note that we don't set a tolerance for each local minimization.
            # We do this because we want to be in the region of the global best-response and by doing it for each local
            # optimization we risk missing this strategy.
            options['f_tol'] = epsilon
        best_res = scopt.shgo(lambda x: - objective(x, expected_returns, u), bounds=bounds, constraints=constraints,
                              sampling_method='sobol', options=options)
    else:
        guesses = max(1, guesses)  # Perform at least one guess.
        init_guesses = []
        if init_strat is not None:
            init_guesses.append(init_strat)
            guesses -= 1

        for i in range(guesses):
            guess = np.random.dirichlet(np.ones(num_actions))  # Random distribution summing to one.
            init_guesses.append(guess)

        best_res = None
        for guess in init_guesses:  # Attempt a local minimization for the amount of initial guesses.
            tol = None
            if epsilon > 0:
                tol = epsilon  # Set a specified tolerance.
            res = scopt.minimize(lambda x: - objective(x, expected_returns, u), guess, bounds=bounds,
                                 constraints=constraints, tol=tol)

            if best_res is None or res['fun'] < best_res['fun']:  # If this local minimization was better, then use it.
                best_res = res

    success = best_res['success']
    br_strategy = best_res['x'] / np.sum(best_res['x'])  # In case of floating point errors.
    br_utility = objective(br_strategy, expected_returns, u)  # Calculate the utility to force the same precision.
    return success, br_strategy, br_utility


def calc_expected_returns(player, payoff_matrix, joint_strategy):
    """Calculate the expected return for a player's actions with a given joint strategy.

    Args:
        player (int): The player to caculate expected returns for.
        payoff_matrix (ndarray): The payoff matrix for the given player.
        joint_strategy (List[ndarray]): A list of each player's individual strategy.

    Returns:
        ndarray: The expected returns for the given player's actions.

    """
    num_objectives = payoff_matrix.shape[-1]
    num_actions = len(joint_strategy[player])
    num_players = len(joint_strategy)
    opponents = np.delete(np.arange(num_players), player)
    expected_returns = payoff_matrix

    for opponent in opponents:  # Loop over all opponent strategies.
        strategy = joint_strategy[opponent]  # Get this opponent's strategy.

        # We reshape this strategy to be able to multiply along the correct axis for weighting expected returns.
        # For example if you end up in [1, 2] or [2, 3] with 50% probability.
        # We calculate the individual expected returns first: [0.5, 1] or [1, 1.5]
        dim_array = np.ones((1, expected_returns.ndim), int).ravel()
        dim_array[opponent] = -1
        strategy_reshaped = strategy.reshape(dim_array)

        expected_returns = expected_returns * strategy_reshaped  # Calculate the probability of a joint state occurring.
        # We now take the sum of the weighted returns to get the expected returns.
        # We need keepdims=True to make sure that the opponent still exists at the correct axis, their action space is
        # just reduced to one action resulting in the expected return now.
        expected_returns = np.sum(expected_returns, axis=opponent, keepdims=True)

    expected_returns = expected_returns.reshape(num_actions, num_objectives)  # Cast the result to a correct shape.

    return expected_returns


def calc_utility_from_joint_strat(u, player, payoff_matrix, joint_strategy):
    """Calculate the utility from a given joint strategy.

    Args:
        u (callable): The utility function for this player.
        player (int): The player to calculate expected returns for.
        payoff_matrix (ndarray): The payoff matrix for the given player.
        joint_strategy (List[ndarray]): A list of each player's individual strategy.

    Returns:
        float: The utility from the joint strategy for this player.
    """
    expected_returns = calc_expected_returns(player, payoff_matrix, joint_strategy)
    strategy = joint_strategy[player]
    utility = objective(strategy, expected_returns, u)
    return utility


def calc_best_response(u, player, payoff_matrix, joint_strategy, epsilon=0, global_opt=False, init_strat=None):
    """Calculate a best response for a given player to a joint strategy.

    Args:
        u (callable): The utility function for this player.
        player (int): The player to calculate expected returns for.
        payoff_matrix (ndarray): The payoff matrix for the given player.
        joint_strategy (List[ndarray]): A list of each player's individual strategy.
        epsilon (float, optional): Tolerance parameter to calculate an epsilon best-response strategy.
            (Default value = 0)
        global_opt (bool, optional): Whether to use a global optimiser or a local one. (Default value = False)
        init_strat (ndarray, optional): The initial guess for the best response. (Default value = None)

    Returns:
        ndarray: A best response strategy.

    """
    expected_returns = calc_expected_returns(player, payoff_matrix, joint_strategy)
    _, br_strategy, _ = optimise_policy(expected_returns, u, epsilon=epsilon, global_opt=global_opt,
                                        init_strat=init_strat)
    return br_strategy


def verify_nash(monfg, u_tpl, joint_strat, epsilon=0, tol=1e-12, strict=False):
    """Verify whether the joint strategy is a Nash equilibrium

    Args:
        monfg (List[ndarray]): A list of payoff matrices.
        u_tpl (Tuple[callable]): A utility function per player.
        joint_strat (List[ndarray]): The joint strategy to verify.
        epsilon (float, optional): An optional parameter to allow for approximate Nash equilibria. (Default value = 0)
        tol (float, optional): The tolerance in the utility calculation. The default is set to the shgo default from
        SciPy. (Default value = 1e-12)
        strict (bool, optional): Whether to count unsuccessful optimisations as unverified and thus returning False.
        (Default value = False)

    Note:
        A Nash equilibrium occurs whenever all strategies are best-responses to each other. We specifically use a global
        optimiser in this function to ensure all strategies are really best-responses and not local optima. Be aware
        that finding a global optimum for a function is computationally expensive, so this function might take longer
        than expected.

    Returns:
        bool: Whether the given joint strategy is a Nash equilibrium.
    """
    for player, (payoffs, u, strat) in enumerate(zip(monfg, u_tpl, joint_strat)):
        expected_returns = calc_expected_returns(player, payoffs, joint_strat)
        utility_from_strat = objective(strat, expected_returns, u)
        success, br_strat, br_utility = optimise_policy(expected_returns, u, global_opt=True)
        if (not strict or success) and utility_from_strat + epsilon + tol < br_utility:
            return False
    return True


def verify_all_nash(monfg, u_tpl, joint_strats, epsilon=0, tol=1e-12):
    """Globally verify if each joint strategy in a list is a Nash equilibrium.

    Args:
        monfg (List[ndarray]): An MONFG as a list of payoff matrices.
        u_tpl (Tuple[callable]): A tuple of utility functions.
        joint_strats (List[ndarray]): A list of joint strategies to check.
        epsilon (float, optional): An optional parameter to allow for approximate Nash equilibria. (Default value = 0)
        tol (float, optional): The tolerance in the utility calculation. The default is set to the shgo default from
        SciPy. (Default value = 1e-12)

    Returns:
        bool: Whether the joint_strategies in the list are actually Nash equilibria.
    """
    for joint_strat in joint_strats:
        if not verify_nash(monfg, u_tpl, joint_strat, epsilon=epsilon, tol=tol):
            return False
    return True
