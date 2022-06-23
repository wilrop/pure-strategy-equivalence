import numpy as np

from rich import print, box
from rich.console import Console
from rich.table import Table


def print_two_player_monfg(game, name='MONFG', highlight_cells=None):
    """Visualise a two-player MONFG as a single payoff matrix.

    Args:
        game (List[ndarray]): The list of payoff matrices.
        name (str, optional): The name of the game. (Default value = 'MONFG')
        highlight_cells (List[array_like], optional): Cell coordinates to highlight. (Default value = None)

    Returns:

    """
    if highlight_cells is None:
        highlight_cells = []

    console = Console()

    if len(game) == 2:
        table = Table(title=name, show_header=False, show_lines=True, box=box.HEAVY)

        player_actions = game[0].shape[:-1]
        top_row = ['Action'] + [f'[bold light_slate_blue]{action}.' for action in range(player_actions[1])]
        table.add_row(*top_row)

        for p1_action in range(player_actions[0]):
            row_data = [f'[bold light_slate_blue]{p1_action}.']

            for p2_action in range(player_actions[1]):
                data = f'{tuple(game[0][p1_action, p2_action])}; {tuple(game[1][p1_action, p2_action])}'

                for highlight_strat in highlight_cells:
                    if ([p1_action, p2_action] == highlight_strat).all():
                        data = f'[black on green3]{data}'
                        break

                row_data.append(data)

            table.add_row(*row_data)
        console.print(table)
    else:
        print(f'This visualisation is currently not supported for games with more than two players')


def print_payoff_matrices(game, name='MONFG', highlight_cells=None):
    """Visualise an MONFG as a list of payoff matrices.

    Args:
        game (List[ndarray]): The list of payoff matrices.
        name (str, optional): The name of the game. (Default value = 'MONFG')
        highlight_cells (List[array_like], optional): Cell coordinates to highlight. (Default value = None)

    Returns:

    """
    if highlight_cells is None:
        highlight_cells = []

    player_actions = game[0].shape[:-1]

    console = Console()

    for player, (payoff_matrix, num_actions) in enumerate(zip(game, player_actions)):
        table = Table(title=f'{name}: Player {player}', show_header=False, show_lines=True, box=box.HEAVY)

        opp_actions = np.delete(player_actions, player)
        opp_joint_strats = np.unravel_index(np.arange(np.prod(opp_actions)), opp_actions)
        opp_joint_strats = list(zip(*[opp_joint_strats[i] for i in range(len(opp_joint_strats))]))

        top_row = ['Action'] + [f'[bold light_slate_blue]{opp_strat}' for opp_strat in opp_joint_strats]
        table.add_row(*top_row)

        for action in range(num_actions):
            row_data = [f'[bold light_slate_blue]{action}.']

            for opp_strat in opp_joint_strats:
                joint_strat = np.insert(opp_strat, player, action)
                payoff_vec = tuple(payoff_matrix[tuple(joint_strat)])
                data = f'{payoff_vec}'

                for highlight_strat in highlight_cells:
                    if (joint_strat == highlight_strat).all():
                        data = f'[black on green3]{payoff_vec}'
                        break

                row_data.append(f'{data}')

            table.add_row(*row_data)
        console.print(table)


def print_monfg(game, name='MONFG', highlight_cells=None):
    """Visualise an MONFG as a matrix for two-player games or list of payoff matrices for n-player games.

    Args:
        game (List[ndarray]): The list of payoff matrices.
        name (str, optional): The name of the game. (Default value = 'MONFG')
        highlight_cells (List[array_like], optional): Cell coordinates to highlight. (Default value = None)

    Returns:

    """
    if len(game) == 2:
        print_two_player_monfg(game, name, highlight_cells=highlight_cells)
    else:
        print_payoff_matrices(game, name, highlight_cells=highlight_cells)


def print_psne(game, psne_lst, name='MONFG'):
    """Pretty print a list of PSNE.

    Args:
        game (ndarray): A payoff matrix.
        psne_lst (List[ndarray]): A list of PSNE.
        name (str, optional): The name of the game. (Default value = 'MONFG')

    Returns:

    """
    print('There are a total of ' + repr(len(psne_lst)) + ' pure strategy Nash equilibria')

    for idx, psne in enumerate(psne_lst):
        print(f'PSNE {idx} indexes: {psne}')

    print_monfg(game, name, highlight_cells=psne_lst)


def print_ne(ne, joint_strategy, decimals=None):
    """Pretty print a Nash equilibrium

    Args:
        ne (bool): Whether the joint strategy is a Nash equilibrium.
        joint_strategy (List[ndarray]: The joint strategy that is a Nash equilibrium.
        decimals (int, optional): Round the mixed strategies to a given number of decimals. (Default value = None)

    Returns:

    """
    if decimals is not None:
        joint_strategy = [np.around(strat, decimals=decimals) for strat in joint_strategy]

    if ne:
        print(f'The Nash equilibrium that was found is the joint strategy: {joint_strategy}')
    else:
        print(f'No Nash equilibrium was found, but the final joint strategy was: {joint_strategy}.')


def print_all_ne(ne_lst):
    """Pretty print a list of Nash equilibria.

    Args:
        ne_lst (List[List[ndarray]]): A list of Nash equilibria.

    Returns:

    """
    print(f'There are a total of {len(ne_lst)} Nash equilibria')
    for ne in ne_lst:
        print(repr(ne))


def print_start(algorithm):
    """Pretty print the introduction to an algorithm.

    Args:
        algorithm (str): The name of the algorithm.

    Returns:

    """
    print(f'Executing the {algorithm} algorithm')
    print(f'-----------------------------------------------------')
