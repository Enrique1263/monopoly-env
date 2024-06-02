import os
import sys
import importlib.util
import inspect
from typing import Type
import monopoly_env
from player import Player
import gym

def find_subclasses(path: str, cls: Type[Player]):
    """Find all subclasses of a given class in a given path.

    :param path: The path to the directory where the modules are.
    :param cls: The class to find the subclasses of.
    :return: A list with all the subclasses of the given class in the
        given path.
    """
    player_classes = []

    sys.path.append(path)  # So we can import the modules from the path

    for filename in os.listdir(path):
        if filename.endswith('.py'):
            # Get the full module path name
            mod_name = filename[:-3]
            mod_path = os.path.join(path, filename)

            # Load the module
            spec = importlib.util.spec_from_file_location(mod_name, mod_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, o in inspect.getmembers(module):
                if inspect.isclass(o) and issubclass(o, cls) and o is not cls:
                    player_classes.append(o)

    return player_classes

if __name__ == '__main__':
    players_classes = find_subclasses('agents', Player)
    players = [clazz() for clazz in players_classes]
    env = gym.make('MonopolyEnv-v0',players=players, render_mode='Human', max_steps=1000, board_names_path='cards/f1_board_names.txt', community_chest_path='cards/community_chest.txt', chance_path='cards/chance.txt')
    observation, _ = env.reset()
    done = False
    while not done:
        action = env.players[env.player_on_turn].action(observation)
        observation, reward, Terminated, Truncated, _ = env.step(action)
        done = Terminated or Truncated
    env.close()