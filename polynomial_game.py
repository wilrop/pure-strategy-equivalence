from identity_game import identity_game
from strategy_bijections import one_simplex_coord_to_point


def u1(x, y):
    """The utility function for player 1 in the polynomial game.

    Args:
        x (float): The strategy of player 1.
        y (float): The strategy of player 2.

    Returns:
        float: The utility for player 1.
    """
    return 2 * x * (y ** 2) - x ** 2 - y


def u2(x, y):
    """The utility function for player 2 in the polynomial game.

    Args:
        x (float): The strategy of player 1.
        y (float): The strategy of player 2.

    Returns:
        float: The utility for player 2.
    """
    return - u1(x, y)


def setup_polynomial_game(min_x, max_x):
    """Set up a polynomial game.

    Args:
        min_x (float): The minimum value in the strategy interval.
        max_x (float): The maximum value in the strategy interval.

    Returns:
        List[ndarray], Tuple[callable]: The MONFG and a tuple of utility functions.
    """
    player_actions = (2, 2)
    monfg = identity_game(player_actions)

    def um1(payoff):
        x = one_simplex_coord_to_point(payoff[0:2], min_x, max_x)
        y = one_simplex_coord_to_point(payoff[2:4], min_x, max_x)
        return u1(x, y)

    def um2(payoff):
        x = one_simplex_coord_to_point(payoff[0:2], min_x, max_x)
        y = one_simplex_coord_to_point(payoff[2:4], min_x, max_x)
        return u2(x, y)

    u_tpl = (um1, um2)
    return monfg, u_tpl
