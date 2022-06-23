import numpy as np
import pandas as pd

from IBR import iterated_best_response
from bertrand_pricing_game import setup_bertrand_pricing_game
from best_response import calc_best_response
from fictitious_play import fictitious_play
from polynomial_game import setup_polynomial_game
from strategy_bijections import one_simplex_coord_to_point, one_simplex_point_to_coord


def continuous_br(monfg, u_tpl, player, opp_x, min_x, max_x):
    """Compute the best-response to a specific continuous strategy.

    Args:
        monfg (ndarray): A list of payoff matrices.
        u_tpl (Tuple[callable]): A tuple of utility functions.
        player (int): The player to compute a best-response for.
        opp_x (float): The opponent strategy in the continuous game.
        min_x (float): The minimum value in the continuous game.
        max_x (float): The maximum value in the continuous game.

    Returns:
        float: A best-response in the continuous game.
    """
    player_strat = np.array([1, 0])
    opp_strat = one_simplex_point_to_coord(opp_x, min_x, max_x)
    if player == 0:
        joint_strat = [player_strat, opp_strat]
    else:
        joint_strat = [opp_strat, player_strat]
    br_strat = calc_best_response(u_tpl[player], player, monfg[player], joint_strat, global_opt=True)
    br_x = one_simplex_coord_to_point(br_strat, min_x, max_x)
    return br_x


def run_polynomial_game(min_x=-1, max_x=1, max_iter=1000):
    monfg, u_tpl = setup_polynomial_game(min_x, max_x)
    ne, joint_strat, log = run_experiment(monfg, u_tpl, max_iter=max_iter)
    return ne, joint_strat, log


def run_bertrand_pricing_game(min_price=1, max_price=100, sigma=3, gamma=2, n=2700, m=1, a=50, max_iter=100):
    monfg, u_tpl = setup_bertrand_pricing_game(min_price, max_price, sigma, gamma, n, m, a)
    ne, joint_strat, log = run_experiment(monfg, u_tpl, max_iter=max_iter)
    return ne, joint_strat, log


def run_experiment(monfg, u_tpl, algorithm='FP', max_iter=1000, variant='alternating', global_opt=True):
    if algorithm == 'FP':
        return fictitious_play(monfg, u_tpl, max_iter=max_iter, variant=variant, global_opt=global_opt)
    elif algorithm == 'IBR':
        return iterated_best_response(monfg, u_tpl, max_iter=max_iter, variant=variant, global_opt=global_opt)
    else:
        raise NotImplementedError('Algorithm {}')


def transform_log(run, log, min_x, max_x):
    transformed_log = []
    for record in log:
        i = record[0]
        strat1 = record[1:3]
        strat2 = record[3:5]
        point1 = one_simplex_coord_to_point(strat1, min_x, max_x)
        point2 = one_simplex_coord_to_point(strat2, min_x, max_x)
        transformed_record = [run, i, point1, point2]
        transformed_log.append(transformed_record)
    return transformed_log


def save_logs(logs, name):
    columns = ['run', 'iteration', 'player1', 'player2']
    filename = f'{name}.csv'
    df = pd.DataFrame(logs, columns=columns)
    df.to_csv(filename)


def run_experiments(runs=100):
    poly_min_x = -1
    poly_max_x = 1
    poly_iters = 1000

    bertrand_min_x = 10
    bertrand_max_x = 25
    sigma = 3
    gamma = 2
    n = 2700
    m = 1
    a = 50
    price_iters = 10

    poly_logs = []
    bertrand_logs = []

    for run in range(runs):
        print(f"[{run + 1}/{runs}] Executing run")
        ne, final_strat, poly_log = run_polynomial_game(min_x=poly_min_x, max_x=poly_max_x, max_iter=poly_iters)
        poly_logs.extend(transform_log(run, poly_log, poly_min_x, poly_max_x))
        ne, final_strat, bertrand_log = run_bertrand_pricing_game(min_price=bertrand_min_x, max_price=bertrand_max_x,
                                                                  sigma=sigma, gamma=gamma, n=n, m=m, a=a,
                                                                  max_iter=price_iters)
        bertrand_logs.extend(transform_log(run, bertrand_log, bertrand_min_x, bertrand_max_x))

    save_logs(poly_logs, "polynomial_game")
    save_logs(bertrand_logs, "bertrand_price_game")


if __name__ == '__main__':
    run_experiments(runs=100)
