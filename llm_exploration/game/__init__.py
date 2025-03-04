try:
    from llm_exploration.game.game import GameSimulator
except:
    print("Could not load the GameSimulator, so cannot use it!")

from llm_exploration.game.game_environment_template import GameEnvironmentTemplate
from llm_exploration.game.game_environment import GameEnvironment


def get_game_environment(environment_name: str) -> GameEnvironmentTemplate:
    """
    Helper function to load different environments

    Input:
        environment_name (str):
            The name of the environment to load
            eg., twenty_questions

    Output:
        game_environment (GameEnvironment):
            Loaded game environment object
    """
    return GameEnvironment(env_name=environment_name)
