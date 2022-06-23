from ramo.commitment.best_response_agent import BestResponseAgent
from ramo.commitment.comp_action_agent import CompActionAgent
from ramo.commitment.coop_action_agent import CoopActionAgent
from ramo.commitment.coop_policy_agent import CoopPolicyAgent
from ramo.commitment.non_stationary_agent import NonStationaryAgent
from ramo.commitment.optional_com_agent import OptionalComAgent
from ramo.utility_functions.examples import get_u
from ramo.learners.indep_actor_critic import IndependentActorCriticAgent
from ramo.learners.indep_q import IndependentQAgent
from ramo.learners.ja_actor_critic import JointActionActorCriticAgent
from ramo.learners.ja_q import JointActionQAgent


def create_agents(experiment, u_tpl, num_agents, player_actions, num_objectives, alpha_q=0.01, alpha_theta=0.01,
                  alpha_fq=0.01, alpha_ftheta=0.01, alpha_cq=0.01, alpha_ctheta=0.01, alpha_q_decay=1,
                  alpha_theta_decay=1, alpha_com_decay=1, epsilon=1, epsilon_decay=0.995, min_epsilon=0.1, rng=None):
    """Create a list of agents.

    Args:
        experiment (str): The type of experiment that is run. This is used to determine which agents to create.
        u_tpl (Tuple[str]): A tuple of utility functions.
        num_agents (int): The number of agents to create.
        player_actions (Tuple[int]): The number of actions per player.
        num_objectives (int): The number of objectives.
        alpha_q (float, optional): The learning rate for Q-values. (Default value = 0.01)
        alpha_theta (float, optional): The learning rate for policy parameters. (Default value = 0.01)
        alpha_fq (float, optional): The learning rate for follower Q-values. (Default value = 0.01)
        alpha_ftheta (float, optional): The learning rate for follower policy parameters. (Default value = 0.01)
        alpha_cq (float, optional): The learning rate for optional commitment Q-values. (Default value = 0.01)
        alpha_ctheta (float, optional): The learning rate for optional commitment policy parameters.
            (Default value = 0.01)
        alpha_q_decay (float, optional): The decay for the Q-values learning rate. (Default value = 1)
        alpha_theta_decay (float, optional): The decay for the policy parameters learning rate. (Default value = 1)
        alpha_com_decay (float, optional): The decay for the optional commitment strategy learning rate.
            (Default value = 1)
        epsilon (float, optional): The exploration rate for a Q-learner agent. (Default value = 1)
        epsilon_decay (float, optional): The decay for the exploration rate. (Default value = 0.995)
        min_epsilon (float, optional): The minimum value for the exploration rate. (Default value = 0.1)
        rng (Generator, optional): A random number generator. (Default value = None)

    Returns:
        List[Agent]: A list of agents.

    Raises:
        Exception: When the requested agent is unknown in the context of the experiment.

    """
    agents = []
    for ag, u_str, num_actions in zip(range(num_agents), u_tpl, player_actions):
        u = get_u(u_str)
        if experiment == 'indep_ac':
            new_agent = IndependentActorCriticAgent(u, num_actions, num_objectives, alpha_q=alpha_q,
                                                    alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                                    alpha_theta_decay=alpha_theta_decay)
        elif experiment == 'indep_q':
            new_agent = IndependentQAgent(u, num_actions, num_objectives, alpha_q=alpha_q, alpha_q_decay=alpha_q_decay,
                                          epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
        elif experiment == 'ja_ac':
            new_agent = JointActionActorCriticAgent(ag, u, num_actions, num_objectives, player_actions, alpha_q=alpha_q,
                                                    alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                                    alpha_theta_decay=alpha_theta_decay)
        elif experiment == 'ja_q':
            new_agent = JointActionQAgent(ag, u, num_actions, num_objectives, player_actions, alpha_q=alpha_q,
                                          alpha_q_decay=alpha_q_decay, epsilon=epsilon, epsilon_decay=epsilon_decay,
                                          min_epsilon=min_epsilon)
        elif experiment == 'coop_action':
            new_agent = CoopActionAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay, rng=rng)
        elif experiment == 'comp_action':
            new_agent = CompActionAgent(ag, u, num_actions, num_objectives, alpha_lq=alpha_q,
                                        alpha_ltheta=alpha_theta, alpha_fq=alpha_fq, alpha_ftheta=alpha_ftheta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay, rng=rng)
        elif experiment == 'coop_policy':
            new_agent = CoopPolicyAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay, rng=rng)
        elif experiment == 'best_response':
            new_agent = BestResponseAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q,
                                          alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                          alpha_theta_decay=alpha_theta_decay, rng=rng)
            new_agent.set_leader_utility(get_u(u_tpl[0]))
        elif experiment == 'non_stationary':
            new_agent = NonStationaryAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q,
                                           alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                           alpha_theta_decay=alpha_theta_decay, rng=rng)
            new_agent.set_opponent_actions(player_actions[abs(1 - ag)])
        elif experiment == 'opt_coop_action':
            no_com_agent = IndependentActorCriticAgent(u, num_actions, num_objectives, alpha_q=alpha_q,
                                                       alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                                       alpha_theta_decay=alpha_theta_decay, rng=rng)
            com_agent = CoopActionAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay, rng=rng)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, num_actions, num_objectives, alpha_q=alpha_cq,
                                         alpha_theta=alpha_ctheta, alpha_q_decay=alpha_q_decay,
                                         alpha_theta_decay=alpha_com_decay, rng=rng)
        elif experiment == 'opt_comp_action':
            no_com_agent = IndependentActorCriticAgent(u, num_actions, num_objectives, alpha_q=alpha_q,
                                                       alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                                       alpha_theta_decay=alpha_theta_decay, rng=rng)
            com_agent = CompActionAgent(ag, u, num_actions, num_objectives, alpha_lq=alpha_q,
                                        alpha_ltheta=alpha_theta, alpha_fq=alpha_fq, alpha_ftheta=alpha_ftheta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay, rng=rng)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, num_actions, num_objectives, alpha_q=alpha_cq,
                                         alpha_theta=alpha_ctheta, alpha_q_decay=alpha_q_decay,
                                         alpha_theta_decay=alpha_com_decay, rng=rng)
        elif experiment == 'opt_coop_policy':
            no_com_agent = IndependentActorCriticAgent(u, num_actions, num_objectives, alpha_q=alpha_q,
                                                       alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                                       alpha_theta_decay=alpha_theta_decay, rng=rng)
            com_agent = CoopPolicyAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay, rng=rng)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, num_actions, num_objectives, alpha_q=alpha_cq,
                                         alpha_theta=alpha_ctheta, alpha_q_decay=alpha_q_decay,
                                         alpha_theta_decay=alpha_com_decay, rng=rng)
        else:
            raise Exception(f'No agent of type {experiment} exists')
        agents.append(new_agent)
    return agents
