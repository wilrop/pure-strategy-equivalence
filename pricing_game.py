from identity_game import identity_game
from best_response.execute_algorithm import execute_algorithm
from best_response.best_response import calc_best_response
import utils.printing as pt
import numpy as np


def strat_from_price(price, min_price, max_price):
    """Compute the strategy from a given price.

    Args:
        price (float): The input price.
        min_price (float): The minimum price in the game. (Default value = 0)
        max_price (float): The maximum price in the game. (Default value = 1000)

    Returns:
        ndarray: The corresponding strategy.
    """
    x_strat = (price - min_price)/(max_price - min_price)
    y_strat = 1 - x_strat
    strat = np.array([x_strat, y_strat])
    return strat


def price_from_strat(strat, min_price, max_price):
    """Compute the price from a given strategy.

    Args:
        strat (ndarray): The input strategy.
        min_price (float): The minimum price in the game. (Default value = 0)
        max_price (float): The maximum price in the game. (Default value = 1000)

    Returns:
        float: The corresponding price.
    """
    return min_price + strat[0] * (max_price - min_price)


def pricing_u(price_i, price_j, a, b, c, m):
    """The utility function in a pricing game.

    Args:
        price_i (float): The price of the manufacturers' product.
        price_j (float): The price of the competing product.
        a (float, optional): All factors affecting demand other than price. (Default value = 3000)
        b (float, optional): Slope of the demand curve. (Default value = 4)
        c (float, optional): Substitute parameter. When c > 0, the products are subtitutes. When c < 0, the products are
            complements. (Default value = 2)
        m (float, optional): Marginal cost to produce the product. (Default value = 200)

    Returns:
        float: The marginal profit of the player.
    """
    demand = a - b * price_i + c * price_j
    margin = (price_i - m) * demand
    return margin


def pricing_game(min_price=0, max_price=1000, a=3000, b=4, c=2, m=200):
    """Constructs a pricing game.

    Args:
        min_price (float, optional): The minimum price in the game. (Default value = 0)
        max_price (float, optional): The maximum price in the game. (Default value = 1000)
        a (float, optional): All factors affecting price other than price. (Default value = 3000)
        b (float, optional): Slope of the demand curve. (Default value = 4)
        c (float, optional): Substitute parameter. When c > 0, the products are subtitutes. When c < 0, the products are
            complements. (Default value = 2)
        m (float, optional): Marginal cost to produce the product. (Default value = 200)

    Returns:
        List[ndarray], Tuple[callable]: A list of payoff matrices and the utility functions.
    """
    player_actions = (2, 2)
    game = identity_game(player_actions)

    def u1(payoff):
        price_i = price_from_strat(payoff[0:2], min_price, max_price)
        price_j = price_from_strat(payoff[2:4], min_price, max_price)
        return pricing_u(price_i, price_j, a, b, c, m)

    def u2(payoff):
        price_i = price_from_strat(payoff[2:4], min_price, max_price)
        price_j = price_from_strat(payoff[0:2], min_price, max_price)
        return pricing_u(price_i, price_j, a, b, c, m)

    u_tpl = (u1, u2)
    return game, u_tpl


def calc_optimal_price(other_price, min_price=0, max_price=1000, a=3000, b=4, c=2, m=200):
    """Calculate the optimal price by applying a multi-objective best-response computation.

    Args:
        other_price (float): The price of the opponent.
        min_price (float, optional): The minimum price in the game. (Default value = 0)
        max_price (float, optional): The maximum price in the game. (Default value = 1000)
        a (float, optional): All factors affecting price other than price. (Default value = 3000)
        b (float, optional): Slope of the demand curve. (Default value = 4)
        c (float, optional): Substitute parameter. When c > 0, the products are subtitutes. When c < 0, the products are
            complements. (Default value = 2)
        m (float, optional): Marginal cost to produce the product. (Default value = 200)

    Returns:
        float: The best response price.
    """
    price_strat = strat_from_price(other_price, min_price, max_price)
    joint_strat = [np.array([1, 0]), price_strat]
    monfg, u_tpl = pricing_game(min_price, max_price, a, b, c, m)
    br = calc_best_response(u_tpl[0], 0, monfg[0], joint_strat, global_opt=True)
    price = price_from_strat(br, min_price, max_price)
    return price


def find_equilibrium(min_price=0, max_price=1000, a=3000, b=4, c=2, m=200):
    """Compute an equilibrium in a pricing game.

    Args:
        min_price (float, optional): The minimum price in the game. (Default value = 0)
        max_price (float, optional): The maximum price in the game. (Default value = 1000)
        a (float, optional): All factors affecting price other than price. (Default value = 3000)
        b (float, optional): Slope of the demand curve. (Default value = 4)
        c (float, optional): Substitute parameter. When c > 0, the products are subtitutes. When c < 0, the products are
            complements. (Default value = 2)
        m (float, optional): Marginal cost to produce the product. (Default value = 200)

    Returns:
        bool, List[ndarray]: Whether the final joint strategy is a Nash equilibrium and the final joint strategy.
    """
    monfg, u_tpl = pricing_game(min_price, max_price, a, b, c, m)
    print(monfg)
    options = {'variant': 'alternating', 'global_opt': True}
    res = execute_algorithm(monfg, u_tpl, algorithm='IBR', options=options)
    return res


if __name__ == '__main__':
    other_price = 633
    opt_price = calc_optimal_price(other_price)
    print(f'The best response price against {other_price}: {opt_price}')
    ne, final_strategy = find_equilibrium()
    pt.print_ne(ne, final_strategy)
