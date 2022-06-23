def get_player_actions(monfg):
    """Get the number of actions per player for a given MONFG.

    Args:
        monfg (List[ndarray]): An MONFG as a list of payoff matrices.

    Returns:
        Tuple[int]: A tuple with the number of actions per player.
    """
    return monfg[0].shape[:-1]


def get_num_objectives(monfg, player=0):
    """

    Args:
        monfg (List[ndarray]): An MONFG as a list of payoff matrices.
        player (int, optional): The player to get the number of objectives for. (Default value = 0)

    Returns:
        int: The number of objectives in the game for the given player.
    """
    return monfg[player].shape[-1]
