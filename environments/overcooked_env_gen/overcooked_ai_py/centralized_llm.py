import os
import re
import time
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import *

from overcooked_ai_py.mdp.actions import Action, Direction

from openai import OpenAI

from d_star import Map, Dstar


'''
Centralized LLM 
Decentralized LLM -- 2 rounds of discussion

JSON description and text description for positions of each entity
Evaluate success rate (Avg Number of dishes served/Avg Number of dishes that can be served); Transport rate (fraction of subgoals completed); time elapsed/ timesteps for each subgoal; coverage wrt objects (right objects eg: interact with 3 ingredients for making a soup))
Provide feedback to LLM via prompts --> 1 dish served
Vary the prompting methods --> REACT, CoT, Vanilla prompting
'''


client = OpenAI(
	api_key="INSERT_KEY",
	)

# DIRECTIONS = {
# 	"NORTH": (0, -1),
#     "SOUTH": (0, 1),
#     "EAST": (1, 0),
#     "WEST": (-1, 0)
#     }

# def render_level(layout_name):
# 	_, _, env = setup_env(layout_name)
# 	print("size: ", (env.mdp.width, env.mdp.height))
# 	env.render()
# 	time.sleep(60)


def setup_env(layout_name, horizon):
	mdp = OvercookedGridworld.from_layout_name(layout_name)
	env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=horizon)
	agent1 = RandomAgent(all_actions=True)
	agent2 = RandomAgent(all_actions=True)
	agent1.set_agent_index(0)
	agent2.set_agent_index(1)
	agent1.set_mdp(mdp)
	agent2.set_mdp(mdp)
	return agent1, agent2, env


'''
X -> counter locations
P -> pot locations
S -> serving locations
O -> onion dispenser locations
T -> tomato dispenser locations
D -> dish dispenser locations
'''

def create_map_for_d_star(env):
	height, width = len(env.mdp_generator_fn().terrain_mtx), len(env.mdp_generator_fn().terrain_mtx[0])
	m = Map(width, height)
	obstacle_list = list(set(env.mdp_generator_fn().terrain_pos_dict['X'] + env.mdp_generator_fn().terrain_pos_dict['O'] + \
							env.mdp_generator_fn().terrain_pos_dict['P'] + env.mdp_generator_fn().terrain_pos_dict['S'] + \
							env.mdp_generator_fn().terrain_pos_dict['D']))
	m.set_obstacle(obstacle_list)

	dstar = Dstar(m)

	# DRAW MAP
	for i in range(m.row):
		for j in range(m.col):
			print(m.map[i][j].state, end=" ")
		print()

	return m, dstar, obstacle_list


def create_counter_object_map(env):
	counters = env.mdp_generator_fn().terrain_pos_dict['X']
	counter_object_map = {}
	for i in range(len(counters)):
		counter_object_map[i] = "Empty"

	return counter_object_map


def extract_start_goal_loc(env, player_state, high_level_action, obstacle_list):

	if "wait" in high_level_action: # wait
		return player_state.position, player_state.position
	# elif HIGH_LEVEL_ACTIONS[-1] == high_level_action: # move away
	# 	start_loc = player_state.position
	# 	for delta in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
	# 		if (start_loc[0]+delta[0], start_loc[1]+delta[1]) in obstacle_list:
	# 			continue
	# 		goal_loc = (start_loc[0]+delta[0], start_loc[1]+delta[1])
	# 	return start_loc, goal_loc

	print("high_level_action: ", high_level_action)
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

	# print("search_key, search_index")
	# print(search_key, search_index)
	if search_key == "X":
		# print("SEARCH KEY LIST", env.mdp_generator_fn().get_accessible_counter_locations())
		return player_state.position, env.mdp_generator_fn().get_accessible_counter_locations()[search_index]
	else:
		# print("SEARCH KEY LIST", env.mdp_generator_fn().terrain_pos_dict[search_key])
		return player_state.position, env.mdp_generator_fn().terrain_pos_dict[search_key][search_index]




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
"wait for other chefs", # wait until other agent executes its high-leve action
# DO NOT NEED IT SINCE CHEFS CAN GO PAST EACH OTHER
# "move to give way to other chefs" # one step action
]


def agent_level_task_description():
	
	# TASK_DESCRIPTION = """
	# There are 2 agent chefs, Chef-1 and Chef-2, in the kitchen. They must coordinate to make onion soups with 3 onions. 
	# Once a soup is cooked it needs to be placed on a plate and delivered to the serving location. 
	# Each chef can only carry one item at a time. The goal is to maximize the number of deliveries. 
	# The chefs want to be efficient and prepare for the next soup while the current soup is cooking. 
	# They will provide thei action history, current state, and the possible actions that they can take. 
	# The chefs prefer helping each other with their cooking and delivery if the situation arises. 
	# Help them select the best action from the list {}. X must be replaced with the entity number in order to perform an action on it. 
	# Format your response as this template: 
	# <Template Start>
	# Explanation for Chef-1: <Brief explanation for next action not stating the action as is.>. Action for Chef-1: <action>. 
	# Explanation for Chef-2: <Brief explanation for next action not stating the action as is.>. Action for Chef-2: <action>. 
	# <Template End>
	# Only select one action from the action list for both the chefs. It is compulsory to only provide a brief explanation and action for each chef and not say anything else. Got it?
	# """.format(HIGH_LEVEL_ACTIONS)

	# TASK_DESCRIPTION = """
	# There are 2 agent chefs, Chef-1 and Chef-2, in the kitchen who need to prepare, cook and serve onion soups which requires 3 onions each.
	# They both need to cooperate to chop onions, cook soup and serve it avoiding obstacles like counters. Each chef can only carry one item at a time. 
	# How would you collaborate effectively to maximize efficiency and serve the most onion soups quickly? Consider the layout, where ingredients, pots, and serving stations are located.
	# The action history, current state, and the possible actions of the chefs would be provided. 
	# The chefs prefer helping each other with their cooking and delivery if the situation arises. 
	# Help them select the best action from the list {}. X must be replaced with the entity number in order to perform an action on it. 
	# Format your response as this template: 
	# <Template Start>
	# Explanation for Chef-1: <Brief explanation for next action not stating the action as is.>. Action for Chef-1: <action>. 
	# Explanation for Chef-2: <Brief explanation for next action not stating the action as is.>. Action for Chef-2: <action>. 
	# <Template End>
	# Only select one action from the action list for both the chefs. It is compulsory to only provide a brief explanation and action for each chef and not say anything else. Got it?
	# """.format(HIGH_LEVEL_ACTIONS)

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

	return TASK_DESCRIPTION


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

def extract_environment_info(env, next_state, actions):

	counter_locations = env.mdp_generator_fn().get_accessible_counter_locations() #env.mdp_generator_fn().get_counter_locations()
	pot_locations = env.mdp_generator_fn().get_pot_locations()
	serving_locations = env.mdp_generator_fn().get_serving_locations()
	tomato_dispenser_locations = env.mdp_generator_fn().get_tomato_dispenser_locations()
	onion_dispenser_locations = env.mdp_generator_fn().get_onion_dispenser_locations()
	dish_dispenser_locations = env.mdp_generator_fn().get_dish_dispenser_locations()
	pot_state_dict = env.mdp_generator_fn().get_pot_states(next_state)
	
	# print("POT STATE", pot_state_dict)

	# for y in range(len(env.mdp_generator_fn().terrain_mtx)):
	# 	for x in range(len(env.mdp_generator_fn().terrain_mtx[0])):
	# 		print(env.mdp_generator_fn().terrain_mtx[y][x], end=", ")
	# 	print()


	LLM_player_prompts = []

	num_players = len(next_state.players)
	player_state = {}
	for player_num in range(num_players):
		player_state[player_num] = {}

		prompts = ""

		ps = next_state.players[player_num]
		player_state[player_num]["PlayerState"] = [ps.position, ps.orientation, ps.held_object, ps.num_ingre_held, ps.num_plate_held, ps.num_served]

		# prompts += "This is Chef-"+str(player_num)+". I am currently at position "+str(ps.position)+", facing "+str(ps.orientation)+" direction. I am currently holding "+str(ps.held_object)+". I am holding "+str(ps.num_ingre_held)+" ingredients and "+str(ps.num_plate_held)+" plates. I have completed serving "+str(ps.num_served)+" dishes."

		# Relative poses: -
		# NORTH = (0, 1)
		# SOUTH = (0, -1)
		# EAST = (-1, 0)
		# WEST = (1, 0)
		orientation = ps.orientation
		for direction_delta in Direction.ALL_DIRECTIONS:
			if ps.orientation == direction_delta:
				facing = Direction.DIRECTION_TO_STRING[direction_delta]
				break

		# if orientation == (0, 1):
		# 	facing = "NORTH"
		# elif orientation == (0, -1):
		# 	facing = "SOUTH"
		# elif orientation == (1, 0):
		# 	facing = "WEST"
		# elif orientation == (-1, 0):
		# 	facing = "EAST"


		# prompts += "This is Chef-"+str(player_num+1)+". I am currently at position "+str(ps.position)+", facing the "+facing+" direction. I am currently holding "+str(ps.held_object)+". I have completed serving "+str(ps.num_served)+" dishes."

		prompts += "Chef-"+str(player_num+1)+"'s Status: Located at "+str(ps.position)+", facing "+facing+", holding "+str(ps.held_object)+" with "+str(ps.num_served)+" dish served."

		player_pose = ps.position

		# prompts += " Each entity's position is relative to my position, Chef-"+str(player_num)+"."

		prompts += " Nearby Entities (relative to Chef-"+str(player_num)+"):"


		sub_prompt = " Counters (C): ["
		player_state[player_num]["CounterLocations"] = []
		for i, cl in enumerate(counter_locations):
			# direction = relative_pose_to_NSEW([cl[0]-player_pose[0], cl[1]-player_pose[1]])

			# sub_prompt += "C-"+str(i)+": " + direction + ", "
			
			# Direction in numerals
			player_state[player_num]["CounterLocations"].append([cl[0]-player_pose[0], cl[1]-player_pose[1]])
			# sub_prompt += ". Counter-"+str(i)+" is at " + str(player_state[player_num]["CounterLocations"][-1])
			# sub_prompt += "C-"+str(i)+": " + str(player_state[player_num]["CounterLocations"][-1]) + ", "

			sub_prompt += "(C-"+str(i)+", " + str(player_state[player_num]["CounterLocations"][-1]) + ", " + counter_object_map[i] + ") ,"
		
		sub_prompt += "]"

		# sub_prompt += ". Each pot's relative position is with respect to myself, Player-"+str(player_num)+". ["
		sub_prompt += ". Pots (P): ["
		player_state[player_num]["PotLocations"] = []
		for i, pl in enumerate(pot_locations):
			# direction = relative_pose_to_NSEW([pl[0]-player_pose[0], pl[1]-player_pose[1]])

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


			# sub_prompt += "P-"+str(i)+": " + direction + "; " + pot_prompt + ", "

			# Direction in numerals
			player_state[player_num]["PotLocations"].append([pl[0]-player_pose[0], pl[1]-player_pose[1]])
			# sub_prompt += ". Pot-"+str(i)+" is at " + str(player_state[player_num]["PotLocations"][-1])
			# sub_prompt += "P-"+str(i)+": " + str(player_state[player_num]["PotLocations"][-1])+", "
			sub_prompt += "(P-"+str(i)+", " + str(player_state[player_num]["PotLocations"][-1]) + ", " + pot_prompt + "), "
		
		sub_prompt += "]"

		# sub_prompt += ". Each serving location's relative position is with respect to myself, Player-"+str(player_num)+". ["
		sub_prompt += ". Serving Locations (SL): ["
		player_state[player_num]["ServingLocations"] = []
		for i, sl in enumerate(serving_locations):
			# direction = relative_pose_to_NSEW([sl[0]-player_pose[0], sl[1]-player_pose[1]])

			# sub_prompt += "SL-"+str(i)+": " + direction + ", "

			# Direction in numerals
			player_state[player_num]["ServingLocations"].append([sl[0]-player_pose[0], sl[1]-player_pose[1]])
			# sub_prompt += ". Serving Locations-"+str(i)+" is at " + str(player_state[player_num]["ServingLocations"][-1])
			# sub_prompt += "SL-"+str(i)+": " + str(player_state[player_num]["ServingLocations"][-1]) + ", "
			sub_prompt += "(SL-"+str(i)+": " + str(player_state[player_num]["ServingLocations"][-1]) + "), "
		sub_prompt += "]"

		# sub_prompt += ". Each tomato dispenser's relative position is with respect to myself, Player-"+str(player_num)+". ["
		# sub_prompt += ". Tomato Dispenser (TD): ["
		# player_state[player_num]["TomatoDispenserLocations"] = []
		# for i, tdl in enumerate(tomato_dispenser_locations):
		# 	player_state[player_num]["TomatoDispenserLocations"].append([tdl[0]-player_pose[0], tdl[1]-player_pose[1]])
		# 	# sub_prompt += ". Tomato Dispenser-"+str(i)+" is at " + str(player_state[player_num]["TomatoDispenserLocations"][-1])
		# 	sub_prompt += "TD-"+str(i)+": " + str(player_state[player_num]["TomatoDispenserLocations"][-1]) +", "
		# sub_prompt += "]"

		# sub_prompt += "Each onion dispenser's relative position is with respect to myself, Player-"+str(player_num)+". ["
		sub_prompt += ". Onion Dispenser (OD): ["
		player_state[player_num]["OnionDispenserLocations"] = []
		for i, odl in enumerate(onion_dispenser_locations):
			# direction = relative_pose_to_NSEW([odl[0]-player_pose[0], odl[1]-player_pose[1]])

			# sub_prompt += "OD-"+str(i)+": " + direction + ", "

			# Direction in numerals
			player_state[player_num]["OnionDispenserLocations"].append([odl[0]-player_pose[0], odl[1]-player_pose[1]])
			# sub_prompt += ". Onion Dispenser-"+str(i)+" is " + str(player_state[player_num]["OnionDispenserLocations"][-1])
			# sub_prompt += "OD-"+str(i)+": " + str(player_state[player_num]["OnionDispenserLocations"][-1]) + ", "
			sub_prompt += "(OD-"+str(i)+": " + str(player_state[player_num]["OnionDispenserLocations"][-1]) + "), "
		sub_prompt += "]"

		# sub_prompt += "Each dish dispenser's relative position is with respect to myself, Player-"+str(player_num)+". ["
		sub_prompt += ". Dish Dispenser (DD): ["
		player_state[player_num]["DishDispenserLocations"] = []
		for i, ddl in enumerate(onion_dispenser_locations):
			# direction = relative_pose_to_NSEW([ddl[0]-player_pose[0], ddl[1]-player_pose[1]])

			# sub_prompt += "DD-"+str(i)+": " + direction + ", "

			# Direction in numerals
			player_state[player_num]["DishDispenserLocations"].append([ddl[0]-player_pose[0], ddl[1]-player_pose[1]])
			# sub_prompt += ". Dish Dispenser-"+str(i)+" is " + str(player_state[player_num]["DishDispenserLocations"][-1])
			# sub_prompt += "DD-"+str(i)+": " + str(player_state[player_num]["DishDispenserLocations"][-1]) + ", "
			sub_prompt += "(DD-"+str(i)+": " + str(player_state[player_num]["DishDispenserLocations"][-1]) + "), "
		sub_prompt += "]. "

		# providing past action history
		sub_prompt += "Action History of Chef-"+str(player_num+1)+" in the last "+str(len(ACTION_HISTORY[player_num+1]))+" planning steps in descending order: "+str(ACTION_HISTORY[player_num+1])+". "

		prompts += sub_prompt

		LLM_player_prompts.append(prompts)

	return player_state, LLM_player_prompts

HORIZON = 500

task_description = agent_level_task_description()

agent1, agent2, env = setup_env("train_gan_small/gen2_basic_6-6-4", horizon=HORIZON)

counter_object_map = create_counter_object_map(env)

def create_action_dict(path_xy_coordinates, high_level_action):
	# print("PATH XY COORDINATION")
	# print(path_xy_coordinates)
	action_list = [] # 0:NORTH, 1:SOUTH, 2:EAST, 3:WEST, 4:STAY, 5:INTERACT
	for i in range(len(path_xy_coordinates)-1):
		x_start, y_start = path_xy_coordinates[i]
		x_end, y_end = path_xy_coordinates[i+1]

		# print(x_end-x_start, y_end-y_start)

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


done = False
high_level_actions = {0: HIGH_LEVEL_ACTIONS[-1], 1: HIGH_LEVEL_ACTIONS[-1]}
joint_action = ((0, 0), (0, 0))
next_state, timestep_sparse_reward, done, info = env.step(joint_action)
HIGH_LEVEL_ACTIONS_ = []
# while not done:
for i in range(HORIZON):
	env.render()
	# take a random action to get state from the environment
	# joint_action = (agent1.action(env.state)[0], agent2.action(env.state)[0])
	# print("JOINT ACTION")
	# print(joint_action)
	# joint_action = ((0, 0), (0, 0))
	# next_state, timestep_sparse_reward, done, info = env.step(joint_action)

	num_players = len(next_state.players)

	player_state, LLM_player_prompts = extract_environment_info(env, next_state, high_level_actions)

	# print("CHEF-1 INPUT STATE PROMPT")
	# print(LLM_player_prompts[0])

	# print("CHEF-2 INPUT STATE PROMPT")
	# print(LLM_player_prompts[1])


	completion = client.chat.completions.create(
	  # model="gpt-3.5-turbo",
	  model="gpt-4",
	  messages=[
		{"role": "system", "content": task_description},
		{"role": "user", "content": LLM_player_prompts[0] + LLM_player_prompts[1]}
	  ]
	)


	print("*"*20)
	print("LLM RESPONSE:-")
	print(completion.choices[0].message.content)
	print("*"*20)

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


	# print("HIGH_LEVEL_ACTIONS")
	# print(high_level_actions)

	# path returned by planning algo (includes start position)
	path_x, path_y = [], []
	for player_num in range(num_players):
		map_for_dstar, dstar, obstacle_list = create_map_for_d_star(env)
		s, g = extract_start_goal_loc(env, next_state.players[player_num], high_level_actions[player_num], obstacle_list)
		# print("START AND GOAL POSITION EXTRACTED")
		# print("start", s)
		# print("end", g)
		if s[0] == g[0] and s[1] == g[1]:
			path_x.append([])
			path_y.append([])
			continue
		start = map_for_dstar.map[s[0]][s[1]]
		end = map_for_dstar.map[g[0]][g[1]]
		# print("PLANNING PATH")
		rx, ry = dstar.run(start, end)
		path_x.append(rx)
		path_y.append(ry)

		# print("PLANNED PATH")
		# print("path_x", rx)
		# print("path_y", ry)



	# print(next_state.players[0].position, next_state.players[1].position)
	# print("path x")
	# print(path_x)
	# print("path y")
	# print(path_y)


	action_dict = {}
	max_action_list_len = 0
	for player_num in range(num_players):
		path_xy = [(x, y) for x, y in zip(path_x[player_num], path_y[player_num])]
		action_dict[player_num] = create_action_dict(path_xy, high_level_actions[player_num])

		if len(action_dict[player_num]) > max_action_list_len:
			max_action_list_len = len(action_dict[player_num])
	
	for player_num, actions in action_dict.items():
		if len(actions) < max_action_list_len:
			action_dict[player_num] = actions + [(0, 0)]*(max_action_list_len-len(actions))

	# print("ACTION DICTIONARY")
	# print(action_dict)

	# print("HIGH LEVEL ACTIONS")
	# print(HIGH_LEVEL_ACTIONS_)

	for i in range(max_action_list_len):
		joint_action = (action_dict[0][i], action_dict[1][i])
		next_state, timestep_sparse_reward, done, info = env.step(joint_action)	

		print("timestep_sparse_reward", timestep_sparse_reward)
		print("done", done)
		print("info", info)

		env.render()

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
						counter_object_map[counter_index] = "Onion"
					elif "tomato" in abstract_action:
						counter_object_map[counter_index] = "Tomato"
					elif "plate" in abstract_action:
						counter_object_map[counter_index] = "Plate"
				elif "pick up" in abstract_action:
					counter_object_map[counter_index] = "Empty"
	

# 	time.sleep(0.1)
