from identity_game import identity_game
from strategy_bijections import one_simplex_coord_to_point


def u1(x, y):
    return 2 * x * (y ** 2) - x ** 2 - y


def u2(x, y):
    return - u1(x, y)


def setup_polynomial_game(min_x, max_x):
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
