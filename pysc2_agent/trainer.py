import argparse
from functools import partial

from pysc2.agents.random_agent import RandomAgent
from s2clientprotocol import common_pb2 as sc_common

from . import environment as env
from .environment import SubprocVecEnv

parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')

parser.add_argument('--map', type=str, default='MoveToBeacon',
                    help='name of SC2 map')
parser.add_argument('--res', type=int, default=32,
                    help='screen and minimap resolution')
parser.add_argument('--envs', type=int, default=1,
                    help='number of environments simulated in parallel')
parser.add_argument('--step_mul', type=int, default=8,
                    help='number of game steps per agent step')

args = parser.parse_args()


def train():
    env_args = dict(
        map_name=args.map,
        step_mul=args.step_mul,
        game_steps_per_episode=0,
        agent_interface_format=env.make_agent_interface_format(args.res),
        players=[env.make_agent(sc_common.NoRace)],
        visualize=False,
    )

    env_fns = [partial(env.make_sc2env, **env_args)] * args.envs
    envs = SubprocVecEnv(env_fns)

    random_agent = RandomAgent()
    random_agent.setup(envs.observation_spec()[0][0], envs.action_spec()[0][0])

    timesteps = envs.reset()
    while True:
        action = random_agent.step(timesteps[0])
        print(action)
        timesteps = envs.step([action])
