import numpy as np

from best_response import calc_best_response, calc_utility_from_joint_strat


class Player:
    """A best-response player"""
    def __init__(self, pid, u, num_actions, payoff_matrix, init_strategy=None, rng=None):
        self.pid = pid
        self.u = u
        self.num_actions = num_actions
        self.payoff_matrix = payoff_matrix
        self.rng = rng if rng is not None else np.random.default_rng()
        if init_strategy is None:
            self.strategy = np.full(self.num_actions, 1 / self.num_actions)
        else:
            self.strategy = init_strategy

    def update(self, joint_strategy, epsilon=0, global_opt=False):
        """Update the strategy by calculating a best response to the other players' strategies.

        Args:
            joint_strategy (List[ndarray]): A list of each player's individual strategy.
            epsilon (float, optional): An optional parameter to allow for approximate Nash equilibria.
                (Default value = 0)
            global_opt (bool, optional): Whether to use a global optimiser or a local one. (Default value = False)

        Returns:
            Tuple[bool, ndarray]: Whether the strategy has converged and the best response strategy.

        """
        br = calc_best_response(self.u, self.pid, self.payoff_matrix, joint_strategy, epsilon=epsilon,
                                global_opt=global_opt, init_strat=self.strategy)

        converged = False
        if self.check_converged(br, joint_strategy, epsilon=epsilon):
            converged = True
        else:
            self.strategy = br
        return converged, br

    def check_converged(self, new_strat, joint_strat, epsilon=0):
        """Check whether a given player has converged in their best-response dynamics.

        This works by comparing the performance of the old strategy and the new strategy to the current opponent
        strategies. If the old strategy performed as good (or better) in response, we don't have to change the strategy
        and this player has (temporarily) converged.

        Args:
            new_strat: The new best-response strategy.
            joint_strat: The old joint strategy.
            epsilon (float, optional): An optional parameter to allow for approximate Nash equilibria.
                (Default value = 0)

        Returns:
            bool: Whether the player's strategy has converged.

        """
        old_strat_utility = calc_utility_from_joint_strat(self.u, self.pid, self.payoff_matrix, joint_strat)
        joint_strat[self.pid] = new_strat
        new_strat_utility = calc_utility_from_joint_strat(self.u, self.pid, self.payoff_matrix, joint_strat)
        return old_strat_utility + epsilon >= new_strat_utility


class IBRPlayer(Player):
    """A player that learns a strategy using best-response iteration."""
    def __init__(self, pid, u, num_actions, payoff_matrix, init_strategy=None, rng=None):
        super().__init__(pid, u, num_actions, payoff_matrix, init_strategy=init_strategy, rng=rng)

    def update_strategy(self, joint_strat, epsilon=0, global_opt=False):
        """Update the strategy by using the super class implementation.

        Args:
            joint_strat (List[ndarray]): A list of each player's individual strategy.
            epsilon (float, optional): An optional parameter to allow for approximate Nash equilibria.
                (Default value = 0)
            global_opt (bool, optional): Whether to use a global optimiser or a local one. (Default value = False)

        Returns:
            Tuple[bool, ndarray]: Whether the strategy has converged and the best response strategy.

        """
        return super().update(joint_strat, epsilon=epsilon, global_opt=global_opt)


class FPPlayer(Player):
    """A player that learns a strategy using the fictitious play algorithm."""

    def __init__(self, pid, u, player_actions, payoff_matrix, init_strategy=None, rng=None):
        self.pid = pid
        self.player_actions = player_actions
        self.num_actions = player_actions[pid]
        self.empirical_strategies = [np.zeros(num_actions) for num_actions in player_actions]
        super().__init__(pid, u, self.num_actions, payoff_matrix, init_strategy=init_strategy, rng=rng)

    def select_action(self):
        """Select an action using the current strategy.

        Returns:
            int: The selected action.

        """
        return self.rng.choice(range(self.num_actions), p=self.strategy)

    def calc_joint_strategy(self):
        """Calculates the empirical joint strategy.

        Returns:
            List[ndarray]: The joint strategy.

        """
        joint_strategy = []

        for player_actions in self.empirical_strategies:
            strategy = player_actions / np.sum(player_actions)
            joint_strategy.append(strategy)

        joint_strategy[self.pid] = self.strategy
        return joint_strategy

    def update_strategy(self, epsilon=0, global_opt=False):
        """Updates the strategy of the player by calculating a best response to the empirical joint strategy.

        Args:
            epsilon (float, optional): An optional parameter to allow for approximate Nash equilibria.
                (Default value = 0)
            global_opt (bool, optional): Whether to use a global optimiser or a local one. (Default value = False)

        Returns:
            Tuple[bool, ndarray]: Whether the strategy has converged and the best response strategy.

        """
        joint_strat = self.calc_joint_strategy()
        return super().update(joint_strat, epsilon=epsilon, global_opt=global_opt)

    def update_empirical_strategies(self, actions):
        """Update the empirical strategy of all players.

        Args:
            actions (List[int]): The actions that were taken by the players.

        """
        for player, action in enumerate(actions):
            self.empirical_strategies[player][action] += 1
