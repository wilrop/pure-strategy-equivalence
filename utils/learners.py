import numpy as np
from jax.nn import softmax


def softmax_policy(theta):
    """Take a softmax over an array of parameters.

    Args:
        theta (ndarray): An array of policy parameters.

    Returns:
        ndarray: A probability distribution over actions as a policy.

    """
    policy = np.asarray(softmax(theta), dtype=float)
    policy = policy / np.sum(policy)
    return policy
