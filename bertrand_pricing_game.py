from identity_game import identity_game
from strategy_bijections import one_simplex_coord_to_point


def demand_x1(price, a):
    """Compute the demand for product x by customer type 1.

    Args:
        price (float): The price for product x.
        a (float): All factors affecting price other than demand.

    Returns:
        float: The demand for product x by customer type 1.
    """
    return a - price


def demand_x2(price_x, price_y, sigma, gamma, n):
    """Compute the demand for product x by customer type 2.

    Args:
        price_x (float): The price for product x.
        price_y (float): The price for product y.
        sigma (float): The elasticity of substitution between x and y.
        gamma (float): The elasticity of demand for the composite good.
        n (float): The number of type two customers.

    Returns:
        float: The demand for product x by customer type 2.
    """
    return n * (price_x ** (-sigma)) * (price_x ** (1 - sigma) + price_y ** (1 - sigma)) ** (
                (gamma - sigma) / (-1 + sigma))


def demand_x3():
    """Compute the demand for product x by customer type 3.

    Returns:
        float: The demand for product x by customer type 3.
    """
    return 0


def demand_y1():
    """Compute the demand for product y by customer type 1.

    Returns:
        float: The demand for product y by customer type 1.
    """
    return 0


def demand_y2(price_x, price_y, sigma, gamma, n):
    """Compute the demand for product y by customer type 2.

    Args:
        price_x (float): The price for product x.
        price_y (float): The price for product y.
        sigma (float): The elasticity of substitution between x and y.
        gamma (float): The elasticity of demand for the composite good.
        n (float): The number of type two customers.

    Returns:
        float: The demand for product y by customer type 2.
    """
    return n * (price_y ** (-sigma)) * (price_x ** (1 - sigma) + price_y ** (1 - sigma)) ** (
                (gamma - sigma) / (-1 + sigma))


def demand_y3(price, a):
    """Compute the demand for product y by customer type 3.

    Args:
        price (float): The price for product y.
        a (float): All factors affecting price other than demand.

    Returns:
        float: The demand for product y by customer type 3.
    """
    return a - price


def total_demand_x(price_x, price_y, sigma, gamma, n, a):
    """Compute the total demand for product x.

    Args:
        price_x (float): The price for product x.
        price_y (float): The price for product y.
        sigma (float): The elasticity of substitution between x and y.
        gamma (float): The elasticity of demand for the composite good.
        n (float): The number of type two customers.
        a (float): All factors affecting price other than demand.

    Returns:
        float: The total demand for product x.
    """
    return demand_x1(price_x, a) + demand_x2(price_x, price_y, sigma, gamma, n) + demand_x3()


def total_demand_y(price_x, price_y, sigma, gamma, n, a):
    """Compute the total demand for product y.

    Args:
        price_x (float): The price for product x.
        price_y (float): The price for product y.
        sigma (float): The elasticity of substitution between x and y.
        gamma (float): The elasticity of demand for the composite good.
        n (float): The number of type two customers.
        a (float): All factors affecting price other than demand.

    Returns:
        float: The total demand for product y.
    """
    return demand_y1() + demand_y2(price_x, price_y, sigma, gamma, n) + demand_y3(price_y, a)


def profit_x(price_x, price_y, sigma, gamma, n, m, a):
    """Compute the total profit for product x.

    Args:
        price_x (float): The price for product x.
        price_y (float): The price for product y.
        sigma (float): The elasticity of substitution between x and y.
        gamma (float): The elasticity of demand for the composite good.
        n (float): The number of type two customers.
        m (float): The unit cost of production for each firm.
        a (float): All factors affecting price other than demand.

    Returns:
        float: The profit for product x.
    """
    return (price_x - m) * total_demand_x(price_x, price_y, sigma, gamma, n, a)


def profit_y(price_x, price_y, sigma, gamma, n, m, a):
    """Compute the total profit for product y.

    Args:
        price_x (float): The price for product x.
        price_y (float): The price for product y.
        sigma (float): The elasticity of substitution between x and y.
        gamma (float): The elasticity of demand for the composite good.
        n (float): The number of type two customers.
        m (float): The unit cost of production for each firm.
        a (float): All factors affecting price other than demand.

    Returns:
        float: The profit for product y.
    """
    return (price_y - m) * total_demand_y(price_x, price_y, sigma, gamma, n, a)


def marginal_profit_x(price_x, price_y, sigma, gamma, n, m, a):
    """Compute the marginal profit for product x.

    Args:
        price_x (float): The price for product x.
        price_y (float): The price for product y.
        sigma (float): The elasticity of substitution between x and y.
        gamma (float): The elasticity of demand for the composite good.
        n (float): The number of type two customers.
        m (float): The unit cost of production for each firm.
        a (float): All factors affecting price other than demand.

    Returns:
        float: The marginal profit for product x.
    """
    return (price_x - m) * total_demand_x(price_x, price_y, sigma, gamma, n, a)


def marginal_profit_y(price_x, price_y, sigma, gamma, n, m, a):
    """Compute the marginal profit for product y.

    Args:
        price_x (float): The price for product x.
        price_y (float): The price for product y.
        sigma (float): The elasticity of substitution between x and y.
        gamma (float): The elasticity of demand for the composite good.
        n (float): The number of type two customers.
        m (float): The unit cost of production for each firm.
        a (float): All factors affecting price other than demand.

    Returns:
        float: The marginal profit for product y.
    """
    return (price_y - m) * total_demand_y(price_x, price_y, sigma, gamma, n, a)


def setup_bertrand_pricing_game(min_price, max_price, sigma, gamma, n, m, a):
    """Set up a Bertrand price game.

    Args:
        min_price (float): The minimum price in the game.
        max_price (float): The maximum price in the game.
        sigma (float): The elasticity of substitution between x and y.
        gamma (float): The elasticity of demand for the composite good.
        n (int): The number of type two customers.
        m (float): The unit cost of production for each firm.
        a (float): All factors affecting price other than demand.

    Returns:
        List[ndarray], Tuple[callable]: The MONFG and a tuple of utility functions.
    """
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
