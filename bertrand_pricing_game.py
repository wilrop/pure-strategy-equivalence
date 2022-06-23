from identity_game import identity_game
from strategy_bijections import one_simplex_coord_to_point


def demand_x1(price, a):
    return a - price


def demand_x2(price_x, price_y, sigma, gamma, n):
    return n * (price_x ** (-sigma)) * (price_x ** (1 - sigma) + price_y ** (1 - sigma)) ** (
                (gamma - sigma) / (-1 + sigma))


def demand_x3():
    return 0


def demand_y1():
    return 0


def demand_y2(price_x, price_y, sigma, gamma, n):
    return n * (price_y ** (-sigma)) * (price_x ** (1 - sigma) + price_y ** (1 - sigma)) ** (
                (gamma - sigma) / (-1 + sigma))


def demand_y3(price, a):
    return a - price


def total_demand_x(price_x, price_y, sigma, gamma, n, a):
    return demand_x1(price_x, a) + demand_x2(price_x, price_y, sigma, gamma, n) + demand_x3()


def total_demand_y(price_x, price_y, sigma, gamma, n, a):
    return demand_y1() + demand_y2(price_x, price_y, sigma, gamma, n) + demand_y3(price_y, a)


def profit_x(price_x, price_y, sigma, gamma, n, m, a):
    return (price_x - m) * total_demand_x(price_x, price_y, sigma, gamma, n, a)


def profit_y(price_x, price_y, sigma, gamma, n, m, a):
    return (price_y - m) * total_demand_y(price_x, price_y, sigma, gamma, n, a)


def marginal_profit_x(price_x, price_y, sigma, gamma, n, m, a):
    return (price_x - m) * total_demand_x(price_x, price_y, sigma, gamma, n, a)


def marginal_profit_y(price_x, price_y, sigma, gamma, n, m, a):
    return (price_y - m) * total_demand_y(price_x, price_y, sigma, gamma, n, a)


def setup_bertrand_pricing_game(min_price, max_price, sigma, gamma, n, m, a):
    player_actions = (2, 2)
    monfg = identity_game(player_actions)

    def u1(payoff):
        price_x = one_simplex_coord_to_point(payoff[0:2], min_price, max_price)
        price_y = one_simplex_coord_to_point(payoff[2:4], min_price, max_price)
        return marginal_profit_x(price_x, price_y, sigma, gamma, n, m, a)

    def u2(payoff):
        price_x = one_simplex_coord_to_point(payoff[0:2], min_price, max_price)
        price_y = one_simplex_coord_to_point(payoff[2:4], min_price, max_price)
        return marginal_profit_y(price_x, price_y, sigma, gamma, n, m, a)

    u_tpl = (u1, u2)
    return monfg, u_tpl
