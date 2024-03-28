import os
import re
import time
import numpy as np
import json

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import *

from overcooked_ai_py.mdp.actions import Action, Direction

from openai import OpenAI

from d_star import Map, Dstar



'''
Notes based on our discussion (Mayank | Siddharth | Aditya):-

For Centralized LLM :
 - JSON description and text description for positions of each entity
 - Evaluations:- 
	- Success rate (Avg Number of dishes served/Avg Number of dishes that can be served) 
	- Transport rate (fraction of subgoals completed) 
	- Time elapsed/ Timesteps for each subgoal
	- Coverage wrt objects (right objects eg: interact with 3 ingredients for making a soup)

 - Provide feedback to LLM via prompts --> 1 dish served
 - Vary the prompting methods:-
	- REACT
	- CoT
	- Vanilla prompting
'''



'''
Map Layout:-

Top Left corner is (0,0)
Cardinal Directions:
North: (0, -1), South: (0, 1), East: (1, 0), West: (-1, 0)

For D-Star algorithm, the Map is transposed

X -> counter locations
P -> pot locations
S -> serving locations
O -> onion dispenser locations
T -> tomato dispenser locations
D -> dish dispenser locations

'''



client = OpenAI(
	api_key="INSERT_KEY",
	)


# ENVIRONMENT SETUP
def setup_env(layout_name, horizon):
	mdp = OvercookedGridworld.from_layout_name(layout_name)
	env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=horizon)
	return env


# LOW LEVEL MOTION & PATH PLANNING POLICY

def create_map_for_d_star(env):
	height, width = len(env.mdp_generator_fn().terrain_mtx), len(env.mdp_generator_fn().terrain_mtx[0])
	m = Map(width, height)
	obstacle_list = list(set(env.mdp_generator_fn().terrain_pos_dict['X'] + env.mdp_generator_fn().terrain_pos_dict['O'] + \
							env.mdp_generator_fn().terrain_pos_dict['P'] + env.mdp_generator_fn().terrain_pos_dict['S'] + \
							env.mdp_generator_fn().terrain_pos_dict['D']))
	m.set_obstacle(obstacle_list)

	dstar = Dstar(m)

	# DRAW MAP
	# for i in range(m.row):
	# 	for j in range(m.col):
	# 		print(m.map[i][j].state, end=" ")
	# 	print()

	return m, dstar, obstacle_list


def extract_start_goal_loc(env, player_state, high_level_action, obstacle_list):

	if "wait" in high_level_action: # wait
		return player_state.position, player_state.position
	
	if "OD" in high_level_action:
		search_key = "O"
		search_index = int(re.findall(r'\d+', high_level_action.split("OD-")[-1])[0])
	elif "DD" in high_level_action:
		search_key = "D"
		search_index = int(re.findall(r'\d+', high_level_action.split("DD-")[-1])[0])
	elif "C" in high_level_action:
		search_key = "X"
		search_index = int(re.findall(r'\d+', high_level_action.split("C-")[-1])[0])
	elif "P" in high_level_action:
		search_key = "P"
		search_index = int(re.findall(r'\d+', high_level_action.split("P-")[-1])[0])
	elif "SL" in high_level_action:
		search_key = "S"
		search_index = int(re.findall(r'\d+', high_level_action.split("SL-")[-1])[0])

	if search_key == "X":
		return player_state.position, env.mdp_generator_fn().get_accessible_counter_locations()[search_index]
	else:
		return player_state.position, env.mdp_generator_fn().terrain_pos_dict[search_key][search_index]



# Global Variables For LLM High Level Planning

LLM_MODEL = "gpt-4" # "gpt-3.5-turbo"

HIGH_LEVEL_ACTIONS = [
"move to OD-X and pick up an onion",
"move to DD-X and pick up a plate",
"move to C-X and pick up the onion",
"move to C-X and pick up the plate",
"move to P-X and put onion in it",
"move to P-X and put soup on the plate",
"move to SL-X to deliver the soup",
"move to C-X to place the onion",
"move to C-X to place the plate",
"wait for other chefs", 
]


TASK_DESCRIPTION = """
Imagine two chefs in a kitchen aiming to efficiently prepare and serve onion soups (minimizing time to prepare and deliver) cooked with 3 onions. Given their current tasks and obstacles, like counters, they must decide on the best next action. 
Each chef can carry one item only and they aim to assist each other. Based on the kitchen layout and the locations of ingredients, pots, and serving stations, advise on their next moves from the provided actions list {}. 
Each action targets an entity, indicated by X. X must be replaced with an appropriate entity number. Offer a rationale and specific next step for each chef, using this format:
<Template Start> 
Explanation for Chef-1: <Rationale>. Action for Chef-1: <Specific action>. 
Explanation for Chef-2: <Rationale>. Action for Chef-2: <Specific action>. 
<Template End>
Choose only one action per chef from the action list provided, focusing on concise reasoning and actions. Understand?
""".format(HIGH_LEVEL_ACTIONS)

# 1 episode length
HORIZON = 500
NUM_EPISODES = 100
RENDER = False

# We start with the waiting action
ACTION_HISTORY_LEN = 3
ACTION_HISTORY = {
	1: ["wait for other chefs", "wait for other chefs", "wait for other chefs"],
	2: ["wait for other chefs", "wait for other chefs", "wait for other chefs"]
}
ACTION_HISTORY_COUNTER = {
	1 : 0, 
	2: 0
}


# HELPER FUNCTIONS FOR LLM HIGH LEVEL PLANNING
def relative_pose_to_NSEW(relative_pose):
	direction = ""
	if relative_pose[0]>0:
		direction += str(abs(relative_pose[0])) + " steps EAST and "
	else:
		direction += str(abs(relative_pose[0])) + " steps WEST and "

	if relative_pose[1]>0:
		direction += str(abs(relative_pose[1])) + " steps SOUTH"
	else:
		direction += str(abs(relative_pose[1])) + " steps NORTH"

	return direction



def create_counter_object_map(env):
	counters = env.mdp_generator_fn().terrain_pos_dict['X']
	counter_object_map = {}
	for i in range(len(counters)):
		counter_object_map[i] = "Empty"

	return counter_object_map



def extract_environment_info(env, next_state, actions):
	counter_locations = env.mdp_generator_fn().get_accessible_counter_locations() #env.mdp_generator_fn().get_counter_locations()
	pot_locations = env.mdp_generator_fn().get_pot_locations()
	serving_locations = env.mdp_generator_fn().get_serving_locations()
	tomato_dispenser_locations = env.mdp_generator_fn().get_tomato_dispenser_locations()
	onion_dispenser_locations = env.mdp_generator_fn().get_onion_dispenser_locations()
	dish_dispenser_locations = env.mdp_generator_fn().get_dish_dispenser_locations()
	pot_state_dict = env.mdp_generator_fn().get_pot_states(next_state)
	
	LLM_player_prompts = []

	num_players = len(next_state.players)
	player_state = {}
	for player_num in range(num_players):
		player_state[player_num] = {}

		prompts = ""

		ps = next_state.players[player_num]
		player_state[player_num]["PlayerState"] = [ps.position, ps.orientation, ps.held_object, ps.num_ingre_held, ps.num_plate_held, ps.num_served]

		orientation = ps.orientation
		for direction_delta in Direction.ALL_DIRECTIONS:
			if ps.orientation == direction_delta:
				facing = Direction.DIRECTION_TO_STRING[direction_delta]
				break

		# Basic details about each chef
		prompts += "Chef-"+str(player_num+1)+"'s Status: Located at "+str(ps.position)+", facing "+facing+", holding "+str(ps.held_object)+" with "+str(ps.num_served)+" dish served. Nearby Entities (relative to Chef-"+str(player_num)+"):"

		player_pose = ps.position

		prompts = " Counters (C): ["
		player_state[player_num]["CounterLocations"] = []
		for i, cl in enumerate(counter_locations):
			player_state[player_num]["CounterLocations"].append([cl[0]-player_pose[0], cl[1]-player_pose[1]])
			prompts += "(C-"+str(i)+", " + str(player_state[player_num]["CounterLocations"][-1]) + ", " + COUNTER_OBJECT_MAP[i] + ") ,"
		
		prompts += "]. Pots (P): ["

		player_state[player_num]["PotLocations"] = []
		for i, pl in enumerate(pot_locations):
			
			for key in ["empty", "onion", "tomato"]:
				if key == "empty" and (pl[0], pl[1]) in pot_state_dict[key]:
					# pot_prompt = "It is empty"
					pot_prompt = "Empty"
				elif key != "empty":
					for k in pot_state_dict[key].keys():
						if "1" in k and (pl[0], pl[1]) in pot_state_dict[key][k]:
							pot_prompt = "It has 1 " + key + " and needs 2 more " + key + "s"
						elif "2" in k and (pl[0], pl[1]) in pot_state_dict[key][k]:
							pot_prompt = "It has 2 " + key + "s and needs 1 more " + key + ""
						elif "ready" in k and (pl[0], pl[1]) in pot_state_dict[key][k]:
							pot_prompt = key + " soup is ready. It can be served"
						elif "cooking" in k and (pl[0], pl[1]) in pot_state_dict[key][k]:
							pot_prompt = key + " soup is cooking. It will be ready soon"


			player_state[player_num]["PotLocations"].append([pl[0]-player_pose[0], pl[1]-player_pose[1]])
			
			prompts += "(P-"+str(i)+", " + str(player_state[player_num]["PotLocations"][-1]) + ", " + pot_prompt + "), "
		
		prompts += "]. Serving Locations (SL): ["

		player_state[player_num]["ServingLocations"] = []
		for i, sl in enumerate(serving_locations):
			player_state[player_num]["ServingLocations"].append([sl[0]-player_pose[0], sl[1]-player_pose[1]])
			prompts += "(SL-"+str(i)+": " + str(player_state[player_num]["ServingLocations"][-1]) + "), "
		prompts += "]. Onion Dispenser (OD): ["

		player_state[player_num]["OnionDispenserLocations"] = []
		for i, odl in enumerate(onion_dispenser_locations):
			
			player_state[player_num]["OnionDispenserLocations"].append([odl[0]-player_pose[0], odl[1]-player_pose[1]])
			prompts += "(OD-"+str(i)+": " + str(player_state[player_num]["OnionDispenserLocations"][-1]) + "), "
		
		prompts += "]. Dish Dispenser (DD): ["

		player_state[player_num]["DishDispenserLocations"] = []
		for i, ddl in enumerate(onion_dispenser_locations):
			
			player_state[player_num]["DishDispenserLocations"].append([ddl[0]-player_pose[0], ddl[1]-player_pose[1]])
			prompts += "(DD-"+str(i)+": " + str(player_state[player_num]["DishDispenserLocations"][-1]) + "), "
		
		prompts += "]. Action History of Chef-"+str(player_num+1)+" in the last "+str(len(ACTION_HISTORY[player_num+1]))+" planning steps in descending order: "+str(ACTION_HISTORY[player_num+1])+". "

		LLM_player_prompts.append(prompts)

	return player_state, LLM_player_prompts


ENV = setup_env("train_gan_small/gen2_basic_6-6-4", horizon=HORIZON)

COUNTER_OBJECT_MAP = create_counter_object_map(ENV)

# Create low level actions from high level actions
def create_action_dict(path_xy_coordinates, high_level_action):
	action_list = []
	for i in range(len(path_xy_coordinates)-1):
		x_start, y_start = path_xy_coordinates[i]
		x_end, y_end = path_xy_coordinates[i+1]

		if x_end-x_start > 0:
			x_delta = 1
		elif x_end-x_start < 0:
			x_delta = -1
		else:
			x_delta = 0

		if y_end-y_start > 0:
			y_delta = 1
		elif y_end-y_start < 0:
			y_delta = -1
		else:
			y_delta = 0

		action_list.append((x_delta, y_delta))


	if "pick" in high_level_action or "put" in high_level_action or "place" in high_level_action or "deliver" in high_level_action:
		action_list.append("interact")

	return action_list


# DATA TO RECORD
DATA_TO_RECORD = {
"CHEF-1":
{
	"onion_pickup": [],
	"useful_onion_pickup": [],
	"onion_drop": [],
	"useful_onion_drop": [],
	"potting_onion": [],
	"dish_pickup": [],
	"useful_dish_pickup": [],
	"dish_drop": [],
	"useful_dish_drop": [],
	"soup_pickup": [],
	"soup_delivery": [],
	"soup_drop": [],
},

"CHEF-2":
{
	"onion_pickup": [],
	"useful_onion_pickup": [],
	"onion_drop": [],
	"useful_onion_drop": [],
	"potting_onion": [],
	"dish_pickup": [],
	"useful_dish_pickup": [],
	"dish_drop": [],
	"useful_dish_drop": [],
	"soup_pickup": [],
	"soup_delivery": [],
	"soup_drop": [],
},

"LLM_RESPONSE": [],

"LLM_INPUT": [],

"NUM_LLM_CALLS": [],

"AVG_HIGH_LEVEL_ACTION_EXEC_TIME": [],

}


for episode in range(NUM_EPISODES):

	ENV.reset()

	done = False
	high_level_actions = {0: HIGH_LEVEL_ACTIONS[-1], 1: HIGH_LEVEL_ACTIONS[-1]}
	joint_action = ((0, 0), (0, 0))
	next_state, timestep_sparse_reward, done, info = ENV.step(joint_action)
	HIGH_LEVEL_ACTIONS_ = []

	# episodic data to record 
	num_llm_calls = 0
	avg_high_level_action_exec_time = []
	llm_response = []
	llm_input = []

	while not done: # Horizon length handled internally by the environment

		if RENDER:
			ENV.render()
			# time.sleep(0.1)
		
		num_players = len(next_state.players)

		_, LLM_player_prompts = extract_environment_info(ENV, next_state, high_level_actions)

		completion = client.chat.completions.create(
		  model=LLM_MODEL,
		  messages=[
			{"role": "system", "content": TASK_DESCRIPTION},
			{"role": "user", "content": LLM_player_prompts[0] + LLM_player_prompts[1]}
		  ]
		)

		# record llm input and response
		num_llm_calls += 1
		llm_input.append(LLM_player_prompts[0] + ". " + LLM_player_prompts[1])
		llm_response.append(completion.choices[0].message.content)


		high_level_actions = []
		for player_num in range(num_players):

			if player_num == 0:
				high_level_actions.append((completion.choices[0].message.content).split("Action for Chef-1: ")[-1].split("\nExplanation for Chef-2: ")[0])
			else:
				high_level_actions.append((completion.choices[0].message.content).split("Action for Chef-2: ")[-1].split("\n<Template End>")[0])

			
			ACTION_HISTORY[player_num+1][ACTION_HISTORY_COUNTER[player_num+1]] = high_level_actions[player_num]

			if ACTION_HISTORY_COUNTER[player_num+1] == ACTION_HISTORY_LEN-1:
				ACTION_HISTORY_COUNTER[player_num+1] = 0
			else:
				ACTION_HISTORY_COUNTER[player_num+1] += 1

		
		HIGH_LEVEL_ACTIONS_.append((high_level_actions[0], high_level_actions[1]))


		path_x, path_y = [], []
		for player_num in range(num_players):
			map_for_dstar, dstar, obstacle_list = create_map_for_d_star(ENV)
			s, g = extract_start_goal_loc(ENV, next_state.players[player_num], high_level_actions[player_num], obstacle_list)
			if s[0] == g[0] and s[1] == g[1]:
				path_x.append([])
				path_y.append([])
				continue
			start = map_for_dstar.map[s[0]][s[1]]
			end = map_for_dstar.map[g[0]][g[1]]
			rx, ry = dstar.run(start, end)
			path_x.append(rx)
			path_y.append(ry)


		# Create Action Dict based on planner's path
		action_dict = {}
		max_action_list_len = 0
		for player_num in range(num_players):

			# if no path to be followed
			if len(path_x[player_num]) == 0:
				action_dict[player_num] = []
				continue

			path_xy = [(x, y) for x, y in zip(path_x[player_num], path_y[player_num])]
			action_dict[player_num] = create_action_dict(path_xy, high_level_actions[player_num])

			if len(action_dict[player_num]) > max_action_list_len:
				max_action_list_len = len(action_dict[player_num])
		
		for player_num, actions in action_dict.items():
			if len(actions) < max_action_list_len:
				action_dict[player_num] = actions + [(0, 0)]*(max_action_list_len-len(actions))

		# record action_exec time
		avg_high_level_action_exec_time.append(max_action_list_len)

		
		for i in range(max_action_list_len):
			joint_action = (action_dict[0][i], action_dict[1][i])
			next_state, timestep_sparse_reward, done, info = ENV.step(joint_action)	

			if RENDER:
				ENV.render()
				# time.sleep(0.1)

			# premature break
			if done:
				break


		# update counter_object_map
		search_string = "move to C-"
		for abstract_action in high_level_actions:	
			if search_string in abstract_action:
				# extract index of the counter
				match = re.search(r"C-(\d+)", abstract_action)
				if match:
					counter_index = match.group(1)
					if "place" in abstract_action:
						if "onion" in abstract_action:
							COUNTER_OBJECT_MAP[counter_index] = "Onion"
						elif "tomato" in abstract_action:
							COUNTER_OBJECT_MAP[counter_index] = "Tomato"
						elif "plate" in abstract_action:
							COUNTER_OBJECT_MAP[counter_index] = "Plate"
					elif "pick up" in abstract_action:
						COUNTER_OBJECT_MAP[counter_index] = "Empty"

	# On episode completion record data
	DATA_TO_RECORD["LLM_RESPONSE"].append(llm_response)
	DATA_TO_RECORD["LLM_INPUT"].append(llm_input)
	DATA_TO_RECORD["NUM_LLM_CALLS"].append(num_llm_calls)
	DATA_TO_RECORD["AVG_HIGH_LEVEL_ACTION_EXEC_TIME"].append(avg_high_level_action_exec_time)

	DATA_TO_RECORD["CHEF-1"]["onion_pickup"].append(info["episode"]["ep_game_stats"]["onion_pickup"][0])
	DATA_TO_RECORD["CHEF-2"]["onion_pickup"].append(info["episode"]["ep_game_stats"]["onion_pickup"][1])
	DATA_TO_RECORD["CHEF-1"]["useful_onion_pickup"].append(info["episode"]["ep_game_stats"]["useful_onion_pickup"][0])
	DATA_TO_RECORD["CHEF-2"]["useful_onion_pickup"].append(info["episode"]["ep_game_stats"]["useful_onion_pickup"][1])
	DATA_TO_RECORD["CHEF-1"]["onion_drop"].append(info["episode"]["ep_game_stats"]["onion_drop"][0])
	DATA_TO_RECORD["CHEF-2"]["onion_drop"].append(info["episode"]["ep_game_stats"]["onion_drop"][1])
	DATA_TO_RECORD["CHEF-1"]["useful_onion_drop"].append(info["episode"]["ep_game_stats"]["useful_onion_drop"][0])
	DATA_TO_RECORD["CHEF-2"]["useful_onion_drop"].append(info["episode"]["ep_game_stats"]["useful_onion_drop"][1])
	DATA_TO_RECORD["CHEF-1"]["potting_onion"].append(info["episode"]["ep_game_stats"]["potting_onion"][0])
	DATA_TO_RECORD["CHEF-2"]["potting_onion"].append(info["episode"]["ep_game_stats"]["potting_onion"][1])
	DATA_TO_RECORD["CHEF-1"]["dish_pickup"].append(info["episode"]["ep_game_stats"]["dish_pickup"][0])
	DATA_TO_RECORD["CHEF-2"]["dish_pickup"].append(info["episode"]["ep_game_stats"]["dish_pickup"][1])
	DATA_TO_RECORD["CHEF-1"]["useful_dish_pickup"].append(info["episode"]["ep_game_stats"]["useful_dish_pickup"][0])
	DATA_TO_RECORD["CHEF-2"]["useful_dish_pickup"].append(info["episode"]["ep_game_stats"]["useful_dish_pickup"][1])
	DATA_TO_RECORD["CHEF-1"]["dish_drop"].append(info["episode"]["ep_game_stats"]["dish_drop"][0])
	DATA_TO_RECORD["CHEF-2"]["dish_drop"].append(info["episode"]["ep_game_stats"]["dish_drop"][1])
	DATA_TO_RECORD["CHEF-1"]["useful_dish_drop"].append(info["episode"]["ep_game_stats"]["useful_dish_drop"][0])
	DATA_TO_RECORD["CHEF-2"]["useful_dish_drop"].append(info["episode"]["ep_game_stats"]["useful_dish_drop"][1])
	DATA_TO_RECORD["CHEF-1"]["soup_pickup"].append(info["episode"]["ep_game_stats"]["soup_pickup"][0])
	DATA_TO_RECORD["CHEF-2"]["soup_pickup"].append(info["episode"]["ep_game_stats"]["soup_pickup"][1])
	DATA_TO_RECORD["CHEF-1"]["soup_delivery"].append(info["episode"]["ep_game_stats"]["soup_delivery"][0])
	DATA_TO_RECORD["CHEF-2"]["soup_delivery"].append(info["episode"]["ep_game_stats"]["soup_delivery"][1])
	DATA_TO_RECORD["CHEF-1"]["soup_drop"].append(info["episode"]["ep_game_stats"]["soup_drop"][0])
	DATA_TO_RECORD["CHEF-2"]["soup_drop"].append(info["episode"]["ep_game_stats"]["soup_drop"][1])


with open('./record_data/FULL_DICTIONARY.json', 'w') as f:
	json.dump(DATA_TO_RECORD, f)

# np.save(os.path.join("./record_data/Chef_1_onion_pickup"), np.array(DATA_TO_RECORD["CHEF-1"]["onion_pickup"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_2_onion_pickup"), np.array(DATA_TO_RECORD["CHEF-2"]["onion_pickup"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_1_useful_onion_pickup"), np.array(DATA_TO_RECORD["CHEF-1"]["useful_onion_pickup"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_2_useful_onion_pickup"), np.array(DATA_TO_RECORD["CHEF-2"]["useful_onion_pickup"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_1_onion_drop"), np.array(DATA_TO_RECORD["CHEF-1"]["onion_drop"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_2_onion_drop"), np.array(DATA_TO_RECORD["CHEF-2"]["onion_drop"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_1_useful_onion_drop"), np.array(DATA_TO_RECORD["CHEF-1"]["useful_onion_drop"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_2_useful_onion_drop"), np.array(DATA_TO_RECORD["CHEF-2"]["useful_onion_drop"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_1_potting_onion"), np.array(DATA_TO_RECORD["CHEF-1"]["potting_onion"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_2_potting_onion"), np.array(DATA_TO_RECORD["CHEF-2"]["potting_onion"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_1_dish_pickup"), np.array(DATA_TO_RECORD["CHEF-1"]["dish_pickup"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_2_dish_pickup"), np.array(DATA_TO_RECORD["CHEF-2"]["dish_pickup"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_1_useful_dish_pickup"), np.array(DATA_TO_RECORD["CHEF-1"]["useful_dish_pickup"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_2_useful_dish_pickup"), np.array(DATA_TO_RECORD["CHEF-2"]["useful_dish_pickup"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_1_dish_drop"), np.array(DATA_TO_RECORD["CHEF-1"]["dish_drop"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_2_dish_drop"), np.array(DATA_TO_RECORD["CHEF-2"]["dish_drop"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_1_useful_dish_drop"), np.array(DATA_TO_RECORD["CHEF-1"]["useful_dish_drop"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_2_useful_dish_drop"), np.array(DATA_TO_RECORD["CHEF-2"]["useful_dish_drop"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_1_soup_pickup"), np.array(DATA_TO_RECORD["CHEF-1"]["soup_pickup"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_2_soup_pickup"), np.array(DATA_TO_RECORD["CHEF-2"]["soup_pickup"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_1_soup_delivery"), np.array(DATA_TO_RECORD["CHEF-1"]["soup_delivery"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_2_soup_delivery"), np.array(DATA_TO_RECORD["CHEF-2"]["soup_delivery"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_1_soup_drop"), np.array(DATA_TO_RECORD["CHEF-1"]["soup_drop"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Chef_2_soup_drop"), np.array(DATA_TO_RECORD["CHEF-2"]["soup_drop"]), allow_pickle=True, fix_imports=True)

# np.save(os.path.join("./record_data/Num_LLM_calls"), np.array(DATA_TO_RECORD["NUM_LLM_CALLS"]), allow_pickle=True, fix_imports=True)
# np.save(os.path.join("./record_data/Avg_high_level_action_exec_time"), np.array(DATA_TO_RECORD["AVG_HIGH_LEVEL_ACTION_EXEC_TIME"]), allow_pickle=True, fix_imports=True)

# with open('./record_data/LLM_inputs.json', 'w') as f:
# 	json.dump(DATA_TO_RECORD["LLM_INPUT"], f)

# with open('./record_data/LLM_response.json', 'w') as f:
# 	json.dump(DATA_TO_RECORD["LLM_RESPONSE"], f)