import os
import time
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import *


def render_level(layout_name):
    _, _, env = setup_env(layout_name)
    print("size: ", (env.mdp.width, env.mdp.height))
    env.render()
    time.sleep(60)


def setup_env(layout_name):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)
    agent1 = RandomAgent(all_actions=True)
    agent2 = RandomAgent(all_actions=True)
    agent1.set_agent_index(0)
    agent2.set_agent_index(1)
    agent1.set_mdp(mdp)
    agent2.set_mdp(mdp)
    return agent1, agent2, env


def extract_environment_info(env, next_state):

    counter_locations = env.mdp_generator_fn().get_counter_locations()
    pot_locations = env.mdp_generator_fn().get_pot_locations()
    serving_locations = env.mdp_generator_fn().get_serving_locations()
    tomato_dispenser_locations = env.mdp_generator_fn().get_tomato_dispenser_locations()
    onion_dispenser_locations = env.mdp_generator_fn().get_onion_dispenser_locations()
    dish_dispenser_locations = env.mdp_generator_fn().get_dish_dispenser_locations()

    num_players = len(next_state.players)
    player_state = {}
    for player_num in range(num_players):
        player_state[player_num] = {}

        ps = next_state.players[player_num]
        player_state[player_num]["PlayerState"] = [ps.position, ps.orientation, ps.held_object, ps.num_ingre_held, ps.num_plate_held, ps.num_served]

        player_pose = ps.position

        player_state[player_num]["CounterLocations"] = []
        for cl in counter_locations:
            player_state[player_num]["CounterLocations"].append([cl[0]-player_pose[0], cl[1]-player_pose[1]])

        player_state[player_num]["PotLocations"] = []
        for pl in pot_locations:
            player_state[player_num]["PotLocations"].append([pl[0]-player_pose[0], pl[1]-player_pose[1]])

        player_state[player_num]["ServingLocations"] = []
        for sl in serving_locations:
            player_state[player_num]["ServingLocations"].append([sl[0]-player_pose[0], sl[1]-player_pose[1]])

        player_state[player_num]["TomatoDispenserLocations"] = []
        for tdl in tomato_dispenser_locations:
            player_state[player_num]["TomatoDispenserLocations"].append([tdl[0]-player_pose[0], tdl[1]-player_pose[1]])

        player_state[player_num]["OnionDispenserLocations"] = []
        for odl in onion_dispenser_locations:
            player_state[player_num]["OnionDispenserLocations"].append([odl[0]-player_pose[0], odl[1]-player_pose[1]])

        player_state[player_num]["DishDispenserLocations"] = []
        for ddl in onion_dispenser_locations:
            player_state[player_num]["DishDispenserLocations"].append([ddl[0]-player_pose[0], ddl[1]-player_pose[1]])

    return player_state


agent1, agent2, env = setup_env("train_gan_small/gen2_basic_6-6-4")
done = False
while not done:
    env.render()
    joint_action = (agent1.action(env.state)[0], agent2.action(env.state)[0])
    next_state, timestep_sparse_reward, done, info = env.step(joint_action)

    player_state = extract_environment_info(env, next_state)

    print("PLAYER STATE")
    print(player_state)    

    time.sleep(0.1)