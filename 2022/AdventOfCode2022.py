#%%
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import re

#%%
#Day 1: Calorie Counting
def day1():
    with open ('input1.txt') as f:
        food = f.readlines()
        
    elves_calories = []
    first_index = 0
    for ind in range(len(food)):
        if food[ind] == '\n':
            second_index = ind
            elves_calories.append(sum([int(x) for x in food[first_index:second_index]]))
            first_index = ind+1
            
    max_calories = max(elves_calories)
    max_three_calories = sum(sorted(elves_calories, reverse = True)[:3])
    
    return max_calories, max_three_calories

#%%
#Day 2: Rock Paper Scissors
def day2():
    with open ('input2.txt') as f:
        strat = f.readlines()
        
    strat = [s.split() for s in strat]
    
    score = 0
    for round in range(len(strat)):
        game = strat[round]
        if game[1] == "X":
            if game[0] == "A":
                score += 4
            elif game[0] == "B":
                score += 1
            else:
                score += 7
        elif game[1] == "Y":
            if game[0] == "A":
                score += 8
            elif game[0] == "B":
                score += 5
            else:
                score += 2
        else:
            if game[0] == "A":
                score += 3
            elif game[0] == "B":
                score += 9
            else:
                score += 6
                
    scen1_score = score
    
    score = 0
    for roundnum in range(len(strat)):
        game = strat[roundnum]
        if game[1] == "X":
            if game[0] == "A":
                score += 3
            elif game[0] == "B":
                score += 1
            else:
                score += 2
        elif game[1] == "Y":
            if game[0] == "A":
                score += 4
            elif game[0] == "B":
                score += 5
            else:
                score += 6
        else:
            if game[0] == "A":
                score += 8
            elif game[0] == "B":
                score += 9
            else:
                score += 7
                
    scen2_score = score
    
    return scen1_score, scen2_score

#%%
#Day 3: Rucksack Reorganization

def day3():
    with open("input3.txt") as f:
        packs = f.readlines()
    
    packs = [x.replace("\n","") for x in packs]
    packs = [[x[:round(len(x)/2)],x[round(len(x)/2):]] for x in packs]
    
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    numbers = range(1,53)
    scoring = {k:v for (k,v) in zip(letters,numbers)}
    
    total_score = 0
    for pack in packs:
        for item in pack[0]:
            if item in pack[1]:
                total_score += scoring[item]
                break
    
    groups = []      
    for i in range(0,len(packs)-1,3):
        next_group = [packs[i][0] + packs[i][1], packs[i+1][0] + packs[i+1][1], packs[i+2][0] + packs[i+2][1]]
        groups.append(next_group)
    
    shared_score = 0    
    for group in groups:
        for item in group[0]:
            if item in group[1] and item in group[2]:
                shared_score += scoring[item]
                break
            
    return total_score, shared_score

#%%
# Day 4: Camp Cleanup

def day4():
    with open("input4.txt") as f:
        sections = f.readlines()
        
    sections = [x.replace("\n","") for x in sections]
    sections = [x.split(",") for x in sections]
    sections = [[[int(z) for z in y.split("-")] for y in x] for x in sections]
    
    contained_count = 0
    overlap_count = 0
    for section in sections:
        if (section[0][0] <= section[1][0] and section[0][1] >= section[1][1]) or (section[0][0] >= section[1][0] and section[0][1] <= section[1][1]):
            contained_count += 1
            overlap_count += 1 
        elif (section[0][0] >= section[1][0] and section[0][0] <= section[1][1]) or (section[0][1] >= section[1][0] and section[0][1] <= section[1][1]):
            overlap_count += 1
            
    return contained_count, overlap_count

#%%
# Day 5: Supply Stacks

def day5():
    with open ("input5.txt") as f:
        data = f.readlines()
        
    stacks_og = data[:9]
    stacks_og.reverse()
    steps = data[10:]
    
    stacks = []
    for i in range(0, len(stacks_og[0]), 4):
        new_stack = []
        for old_stack in stacks_og:
            new_item = old_stack[i:i+4].strip()
            if new_item == "":
                continue
            else: 
                new_stack.append(new_item)
        stacks.append(new_stack)
        stacks_beginning = deepcopy(stacks)
        
    steps = [x.replace("move","") for x in steps]
    steps = [x.replace("from","") for x in steps]
    steps = [x.replace("to","") for x in steps]
    steps = [x.split() for x in steps]
    
    for step in steps:
        num = int(step[0])
        frm = int(step[1])
        to = int(step[2])
        
        moving = stacks[frm-1][-num:]
        moving.reverse()
        stacks[frm-1] = stacks[frm-1][:-num]
        _ = [stacks[to-1].append(x) for x in moving]
        
    tops_1 = "".join([x[-1][1:2] for x in stacks])
    
    
    stacks = deepcopy(stacks_beginning)
    for step in steps:
        num = int(step[0])
        frm = int(step[1])
        to = int(step[2])
        
        moving = stacks[frm-1][-num:]
        stacks[frm-1] = stacks[frm-1][:-num]
        _ = [stacks[to-1].append(x) for x in moving]
        
    tops_2 = "".join([x[-1][1:2] for x in stacks])
    
    return tops_1, tops_2

#%%
# Day 6: Tuning Trouble

def day6():
    with open("input6.txt") as f:
        data = f.readline()
        
    for i in range(4,len(data)):
        if len(set(data[i-4:i])) == 4:
            break
        
    start_packet = i
    
    for i in range(14,len(data)):
        if len(set(data[i-14:i])) == 14:
            break
        
    start_message = i

    return start_packet, start_message

#%%
# Day 7: No Space Left On Device

def day7():
    with open("input7.txt") as f:
        data = f.readlines()
        
    data = [x.replace("\n","") for x in data]
        
    folder = []
    contents = {}
    for line in data:
        if line[:5] == "$ cd ":
            if ".." in line:
                folder.pop(-1)
            else:
                folder.append(line[5:])
                
            if "/".join(folder) not in contents.keys():
                contents["/".join(folder)] = 0
                
        elif line[0] != "$" and line[:3] != "dir":
            file_size = int(line.split()[0])
            for i in range(len(folder)):
                contents["/".join(folder[:i+1])] += file_size
            
    folder_sizes = np.asarray(list(contents.values()))
    total_small_folders = sum(folder_sizes[folder_sizes <= 100000])
    
    space_taken = contents["/"]
    unused_space = 70000000-space_taken
    need_to_free_up = 30000000-unused_space
    
    sorted_folders = np.sort(folder_sizes)
    deletable = sorted_folders[sorted_folders >= need_to_free_up]
    to_delete = deletable[0]
    
    return total_small_folders, to_delete

#%%
# Day 8: Treetop Tree House

def day8():
    with open("input8.txt") as f:
        raw_forest = f.readlines()
        
    raw_forest = [x.replace("\n","") for x in raw_forest]
        
    forest = []
    for row in raw_forest:
        forest.append([int(x) for x in row])
        
    forest = np.asarray(forest)
    
    visible = 0
    for row in range(forest.shape[0]):
        for col in range(forest.shape[1]):
            
            if row == 0 or row == forest.shape[0]-1 or col == 0 or col == forest.shape[1]-1:
                visible += 1 
                continue
            
            tree = forest[row, col]
            
            top = forest[:row,col]
            bottom = forest[row+1:,col]
            left = forest[row,:col]
            right = forest[row,col+1:]
            
            max_top = max(top)
            max_bottom = max(bottom)
            max_left = max(left)
            max_right = max(right)
            
            if tree > max_top or tree > max_bottom or tree > max_left or tree > max_right:
                visible += 1
    
    views = np.zeros(forest.shape)
    for row in range(forest.shape[0]):
        for col in range(forest.shape[1]):
            tree = forest[row, col]
            
            if row == 0:
                top_view = 0
            else:
                top = list(reversed([x >= tree for x in forest[:row,col]]))
                if True not in top:
                    top_view = len(top)
                else:
                    top_view = top.index(True)+1
    
            if row == forest.shape[0]-1:
                bottom_view = 0
            else:
                bottom = [x >= tree for x in forest[row+1:,col]]
                if True not in bottom:
                    bottom_view = len(bottom)
                else:
                    bottom_view = bottom.index(True)+1
    
            if col == 0:
                left_view = 0
            else:
                left = list(reversed([x >= tree for x in forest[row,:col]]))
                if True not in left:
                    left_view = len(left)
                else:
                    left_view = left.index(True)+1
            
            if col == forest.shape[1]-1:
                right_view = 0
            else:
                right = [x >= tree for x in forest[row,col+1:]]  
                if True not in right:
                    right_view = len(right)
                else:
                    right_view = right.index(True)+1
                    
            views[row,col] = top_view * bottom_view * left_view * right_view
            
    best_view = int(np.max(views))
    
    return visible, best_view
        
#%%
# Day 9: Rope Bridge

def day9():
    with open("input9.txt") as f:
        raw_directions = f.readlines()
        
    directions = [x.replace("\n","") for x in raw_directions]
    
    directions = [x.split() for x in directions]
    directions = [[x[0], int(x[1])] for x in directions]
    
    left_needed = 0
    right_needed = 0
    top_needed = 0
    bottom_needed = 0
    
    pos = [0,0]
    for step in directions:
        if step[0] == 'R':
            pos[1] += step[1]
        elif step[0] == "L":
            pos[1] -= step[1]
        elif step[0] == "U":
            pos[0] -= step[1]
        else:
            pos[0] += step[1]
        
        left_needed = min(left_needed, pos[1])
        right_needed = max(right_needed, pos[1])
        top_needed = min(top_needed, pos[0])
        bottom_needed = max(bottom_needed, pos[0])
    
    def move_tail(leader,follower):     
        if leader[0] == follower[0] - 2:
            if leader[1] < follower[1]:
                follower[0] = follower[0] - 1
                follower[1] = follower[1] - 1
            elif leader[1] == follower[1]:
                follower[0] = follower[0] - 1
            elif leader[1] > follower[1]:
                follower[0] = follower[0] - 1
                follower[1] = follower[1] + 1  
                
        elif leader[1] == follower[1] - 2:
            if leader[0] < follower[0]:
                follower[1] = follower[1] - 1
                follower[0] = follower[0] - 1
            elif leader[0] == follower[0]:
                follower[1] = follower[1] - 1
            elif leader[0] > follower[0]:
                follower[1] = follower[1] - 1
                follower[0] = follower[0] + 1
    
        elif leader[0] == follower[0] + 2:
            if leader[1] < follower[1]:
                follower[0] = follower[0] + 1
                follower[1] = follower[1] - 1
            elif leader[1] == follower[1]:
                follower[0] = follower[0] + 1
            elif leader[1] > follower[1]:
                follower[0] = follower[0] + 1
                follower[1] = follower[1] + 1   
                
        elif leader[1] == follower[1] + 2:
            if leader[0] < follower[0]:
                follower[1] = follower[1] + 1
                follower[0] = follower[0] - 1
            elif leader[0] == follower[0]:
                follower[1] = follower[1] + 1
            elif leader[0] > follower[0]:
                follower[1] = follower[1] + 1
                follower[0] = follower[0] + 1
                
        return follower
    
    
    visited = np.zeros([bottom_needed-top_needed+1, right_needed-left_needed+1])
    start = [-top_needed, -left_needed]
    
    head = start.copy()
    tail = start.copy()
    visited[tuple(start)] = 1  
    
    for step in directions:
        to = step[0]
        num = step[1]
        for _ in range(num):
            if to == "U":
                head[0] -= 1
            elif to == "L":
                head[1] -= 1
            elif to == "D":
                head[0] += 1
            else:
                head[1] += 1  
            tail = move_tail(head,tail)
            visited[tuple(tail)] = 1       
    
    total_visited = int(np.sum(visited))
    
    
    visited = np.zeros([bottom_needed-top_needed+1, right_needed-left_needed+1])
    start = [-top_needed, -left_needed]
    
    head = start.copy()
    tails = [start.copy() for x in range(9)]
    visited[tuple(start)] = 1
    
    for step in directions:
        to = step[0]
        num = step[1]
        for _ in range(num):
            if to == "U":
                head[0] -= 1
            elif to == "L":
                head[1] -= 1
            elif to == "D":
                head[0] += 1
            else:
                head[1] += 1  
            tails[0] = move_tail(head,tails[0])
            for i in range(1,9):
                tails[i] = move_tail(tails[i-1], tails[i])
                
            visited[tuple(tails[8])] = 1
            
    total_visited_tail9 = int(np.sum(visited))
    
    return total_visited, total_visited_tail9

#%%
# Day 10: Cathode-Ray Tube

def day10():
    with open("input10.txt") as f:
        commands_raw = f.readlines()
    
    commands = [x.replace("\n","") for x in commands_raw]
    commands = [x.split() for x in commands]
    commands = [[x[0], int(x[1])] if len(x) ==2 else x[0] for x in commands]
    
    
    x = [1]
    for com in commands:
        if type(com) == str:
            x.append(x[-1])
        else:
            x.append(x[-1])
            x.append(x[-1] + com[1])
    x.pop(-1)
            
    signal_strength = 0
    for cyc in [20, 60, 100, 140, 180, 220]:
        signal_strength += cyc*x[cyc-1]
        
    crt = [x%40 for x in list(range(240))]
    
    x = np.asarray(x)
    crt = np.asarray(crt)
    
    disp = abs(x-crt) <= 1
    disp = disp.reshape(6,40)
    
    plt.imshow(disp)
    
    return signal_strength, disp

#%%
# Day 11: Monkey in the Middle

def day11():
    with open("input11.txt") as f:
        obs_raw = f.readlines()
        
    obs_raw = [x.replace("\n","").strip() for x in obs_raw]
    
    obs = [[]]
    for line in obs_raw:
        if line != "":
            obs[-1].append(line)
        else:
            obs.append([])
            
    monkeys = deepcopy(obs)
    for monkey in monkeys:
        monkey[1] = [int(x) for x in monkey[1][16:].split(", ")]
        monkey[4] = int(monkey[4].split()[-1])
        monkey[5] = int(monkey[5].split()[-1])
        monkey.append(0)
    
    for rnd in range(20):
        for monkey in monkeys:
            for _ in range(len(monkey[1])):
                item = monkey[1].pop(0)
                item = eval(monkey[2][17:], {}, {"old": item})
                item = item//3
                if item%int(monkey[3].split()[-1]) == 0:
                    monkeys[monkey[4]][1].append(item)
                else:
                    monkeys[monkey[5]][1].append(item)
                monkey[6] += 1
                
    passes = [x[6] for x in monkeys]
    sorted_passes = sorted(passes, reverse = True)
    monkey_business_1 = sorted_passes[0] * sorted_passes[1]
    
    monkeys = deepcopy(obs)
    for monkey in monkeys:
        monkey[1] = [int(x) for x in monkey[1][16:].split(", ")]
        monkey[4] = int(monkey[4].split()[-1])
        monkey[5] = int(monkey[5].split()[-1])
        monkey.append(0)
    
    all_divs = [int(x[3].split()[-1]) for x in monkeys]
    all_divs_prod = int(np.prod(all_divs))
    
    for rnd in range(10000):
        for monkey in monkeys:
            for _ in range(len(monkey[1])):
                item = monkey[1].pop(0)
                item = eval(monkey[2][17:], {}, {"old": item})
                item = item % all_divs_prod
                if item%int(monkey[3].split()[-1]) == 0:
                    monkeys[monkey[4]][1].append(item)
                else:
                    monkeys[monkey[5]][1].append(item)
                monkey[6] += 1
                
    passes = [x[6] for x in monkeys]
    sorted_passes = sorted(passes, reverse = True)
    monkey_business_2 = sorted_passes[0] * sorted_passes[1]
    
    return monkey_business_1, monkey_business_2

#%%
# Day 12: Hill Climbing Algorithm

def day12(visualize = False):
    with open("input12.txt") as f:
        map_raw = f.readlines()
        
    map_raw = [x.replace("\n","") for x in map_raw]
        
    letters = [x for x in "SabcdefghijklmnopqrstuvwxyzE"]
    numbers = list(range(0,28))
    
    map = []
    for row in map_raw:
        map.append([numbers[letters.index(x)] for x in row])
        
    map = np.asarray(map)
    
    
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[(i,j)] == 27:
                map[(i,j)] = 26
                start = [i,j]
            if map[(i,j)] == 0:
                map[(i,j)] = 1
                finish = [i,j]
    
    def flat_list(nested_lists):
        flattened = []
        for l in nested_lists:
            flattened += l
        return flattened
    
    def remove_dup(dup_list):
        no_dups = []
        for l in dup_list:
            if l not in no_dups:
                no_dups.append(l)
        return no_dups
    
    
    paths = [[start]]
    while True:
        newPaths = []
        for path in paths:
            latest = path[-1]
            latest_height = map[tuple(latest)]
            if map[tuple([latest[0] + 1, latest[1]])] >= latest_height - 1 and [latest[0] + 1, latest[1]] not in flat_list(paths) and latest[0] < map.shape[0] - 2:
                newPaths.append(path + [[latest[0] + 1, latest[1]]])
            if map[tuple([latest[0] - 1, latest[1]])] >= latest_height - 1 and [latest[0] - 1, latest[1]] not in flat_list(paths) and latest[0] > 0:
                newPaths.append(path + [[latest[0] - 1, latest[1]]])
            if map[tuple([latest[0], latest[1] + 1])] >= latest_height - 1 and [latest[0], latest[1] + 1] not in flat_list(paths) and latest[1] < map.shape[1] - 2:
                newPaths.append(path + [[latest[0], latest[1] + 1]])            
            if map[tuple([latest[0], latest[1] - 1])] >= latest_height - 1 and [latest[0], latest[1] - 1] not in flat_list(paths) and latest[1] > 0:
                newPaths.append(path + [[latest[0], latest[1] - 1]])
                
        last_locs = [x[-1] for x in newPaths]  
                
        paths = []
        for last_loc in remove_dup(last_locs):
            paths.append(newPaths[last_locs.index(last_loc)])
    
        last_locs = [x[-1] for x in paths] 
        
        if finish in last_locs:
            shortest_path = paths[last_locs.index(finish)]
            break
        
    shortest_path_length = len(shortest_path) - 1
    
    
    paths = [[start]]
    while True:
        newPaths = []
        for path in paths:
            latest = path[-1]
            latest_height = map[tuple(latest)]
            if map[tuple([latest[0] + 1, latest[1]])] >= latest_height - 1 and [latest[0] + 1, latest[1]] not in flat_list(paths) and latest[0] < map.shape[0] - 2:
                newPaths.append(path + [[latest[0] + 1, latest[1]]])
            if map[tuple([latest[0] - 1, latest[1]])] >= latest_height - 1 and [latest[0] - 1, latest[1]] not in flat_list(paths) and latest[0] > 0:
                newPaths.append(path + [[latest[0] - 1, latest[1]]])
            if map[tuple([latest[0], latest[1] + 1])] >= latest_height - 1 and [latest[0], latest[1] + 1] not in flat_list(paths) and latest[1] < map.shape[1] - 2:
                newPaths.append(path + [[latest[0], latest[1] + 1]])            
            if map[tuple([latest[0], latest[1] - 1])] >= latest_height - 1 and [latest[0], latest[1] - 1] not in flat_list(paths) and latest[1] > 0:
                newPaths.append(path + [[latest[0], latest[1] - 1]])
                
        last_locs = [x[-1] for x in newPaths]  
                
        paths = []
        for last_loc in remove_dup(last_locs):
            paths.append(newPaths[last_locs.index(last_loc)])
    
        last_heights = [map[tuple(x[-1])] for x in paths] 
        
        if 1 in last_heights:
            shortest_scenic_path = paths[last_heights.index(1)]
            break
        
    shortest_scenic_path_length = len(shortest_scenic_path) - 1
    
    if visualize:
        fig, axes = plt.subplots(1,2)
        axes = axes.ravel()
        
        axes[0].imshow(map)
        axes[0].plot([x[1] for x in shortest_path], [x[0] for x in shortest_path], c = "r")
        axes[0].set_title("Shortest Path")
        
        axes[1].imshow(map)
        axes[1].plot([x[1] for x in shortest_scenic_path], [x[0] for x in shortest_scenic_path], c = "r")
        axes[1].set_title("Shortest Scenic Path")
    
    return shortest_path_length, shortest_scenic_path_length

#%%
# Day 13: Distress Signal

def day13():
    with open("input13.txt") as f:
        raw_packs = f.readlines()
        
    packs = [x.strip() for x in raw_packs]
    packs = [eval(x,{},{}) if x != "" else None for x in packs]
    
    pkgs = [[]]
    for p in packs:
        if p is None:
            pkgs.append([])
        else:
            pkgs[-1].append(p)
    
            
    def comp_packs(pkg1, pkg2):
        left = pkg1
        right = pkg2
        
        if type(left) == int:
            left = [left]
            
        if type(right) == int:
            right = [right]
        
        if len(left) == 0 and len(right) > 0:
            return 1
        elif len(left) > 0 and len(right) == 0:
            return -1
        
        count = 1    
        for l, r in zip(left, right):
            if type(l) == list or type(r) == list:
                ret = comp_packs(l, r)
                if ret == 1:
                    return 1
                elif ret == -1:
                    return -1
                
            else:
                if l < r:
                    return 1
                if l > r:
                    return -1
                
            count += 1
            if len(left) < count and len(right) >= count:
                return 1
            elif len(left) >= count and len(right) < count:
                return -1
            
        return 0
    
    
    right_order_inds = []
    for i, pkg in enumerate(pkgs):
        ans = comp_packs(*pkg)
        if ans == 1:
            right_order_inds.append(i+1)
            
    right_order_inds_sum = sum(right_order_inds)
    
    
    flat_pkgs = [[[2]], [[6]]]
    for pkg in pkgs:
        flat_pkgs.append(pkg[0])
        flat_pkgs.append(pkg[1])
        
    ordered_pkgs = [flat_pkgs[0]]
    for pkg in flat_pkgs[1:]:
        at_the_end = True
        for comp_pkg in ordered_pkgs:
            sort_up = comp_packs(pkg, comp_pkg)
            if sort_up == 1:
                at_the_end = False
                ordered_pkgs.insert(ordered_pkgs.index(comp_pkg), pkg)
                break
        if at_the_end:
            ordered_pkgs.append(pkg)
            
    divider_packets_ind_mul = (ordered_pkgs.index([[2]]) + 1) * (ordered_pkgs.index([[6]]) + 1)
    
    return right_order_inds_sum, divider_packets_ind_mul

#%%
# Day 14: Regolith Reservoir

def day14(visualize = False):
    with open ("input14.txt") as f:
        raw_rock = f.readlines()
        
    rock = [x.strip().split(" -> ") for x in raw_rock]
    rock = [[list(reversed(x.split(","))) for x in y] for y in rock]
    rock = [[[int(x) for x in y] for y in z] for z in rock] 
    
    
    max_x = 0
    max_y = 0
    for row in rock:
        for entry in row:
            max_x = max(max_x, entry[1])
            max_y = max(max_y, entry[0])
            
        
    cave = np.zeros([max_y + 2, max_x + 1])
    for row in rock:
        for i in range(1,len(row)):
            from_rock = row[i-1]
            to_rock = row[i]
            if from_rock[0] == to_rock[0]:
                if from_rock[1] < to_rock[1]:
                    for x in range(from_rock[1],to_rock[1]+1):
                        cave[from_rock[0],x] = 2
                else:
                    for x in range(to_rock[1],from_rock[1]+1):
                        cave[from_rock[0],x] = 2
            else:
                if from_rock[0] < to_rock[0]:
                    for y in range(from_rock[0],to_rock[0]+1):
                        cave[y,from_rock[1]] = 2
                else:
                    for y in range(to_rock[0],from_rock[0]+1):
                        cave[y,from_rock[1]] = 2
            
    def sand_fall(sand, cave):
        sand = sand.copy()
        while cave[(sand[0]+1,sand[1])] == 0:
            if sand[0]+1 == max_y+1:
                return None
            sand[0] += 1 
        return sand
    
    def sand_settle(sand, cave):
        sand = sand.copy()
        if cave[(sand[0]+1,sand[1]-1)] > 0 and cave[(sand[0]+1,sand[1]+1)] > 0:
            sand = sand
        else:
            if sand[0]+1 == max_y+1:
                return None
            elif cave[(sand[0]+1,sand[1]-1)] == 0:
                sand = [sand[0]+1, sand[1]-1]
            elif cave[(sand[0]+1,sand[1]+1)] == 0:
                sand = [sand[0]+1, sand[1]+1]
        return sand
    
    source = [0,500]
    count = 0
    while True:
        sand = source.copy()
            
        while sand_fall(sand, cave) != sand or sand_settle(sand, cave) != sand:
            sand = sand_fall(sand, cave)
            
            if sand == None:
                break
            
            sand = sand_settle(sand, cave)
    
        if sand == None:
            break
        
        count += 1
        cave[tuple(sand)] = 1
        
        subset_cave = cave[:,444:]
    
    cave2 = np.zeros([max_y + 3, round(max_x * 1.5)])
    for row in rock:
        for i in range(1,len(row)):
            from_rock = row[i-1]
            to_rock = row[i]
            if from_rock[0] == to_rock[0]:
                if from_rock[1] < to_rock[1]:
                    for x in range(from_rock[1],to_rock[1]+1):
                        cave2[from_rock[0],x] = 2
                else:
                    for x in range(to_rock[1],from_rock[1]+1):
                        cave2[from_rock[0],x] = 2
            else:
                if from_rock[0] < to_rock[0]:
                    for y in range(from_rock[0],to_rock[0]+1):
                        cave2[y,from_rock[1]] = 2
                else:
                    for y in range(to_rock[0],from_rock[0]+1):
                        cave2[y,from_rock[1]] = 2
                        
    cave2[max_y+2, :] = 2
    
    def sand_fall2(sand, cave):
        sand = sand.copy()
        while cave[(sand[0]+1,sand[1])] == 0:
            sand[0] += 1 
        return sand
    
    def sand_settle2(sand, cave):
        sand = sand.copy()
        if cave[(sand[0]+1,sand[1]-1)] > 0 and cave[(sand[0]+1,sand[1]+1)] > 0:
            sand = sand
        else:
            if cave[(sand[0]+1,sand[1]-1)] == 0:
                sand = [sand[0]+1, sand[1]-1]
            elif cave[(sand[0]+1,sand[1]+1)] == 0:
                sand = [sand[0]+1, sand[1]+1]
        return sand
    
    source = [0,500]
    count2 = 0
    while True:
        sand = source.copy()
        
        if cave2[tuple(sand)] > 0:
            break
            
        while sand_fall2(sand, cave2) != sand or sand_settle2(sand, cave2) != sand:
            sand = sand_fall2(sand, cave2)
            sand = sand_settle2(sand, cave2)
        
        count2 += 1
        cave2[tuple(sand)] = 1
        
    subset_cave2 = cave2[:,300:675]
        
    if visualize:
        fig, axes = plt.subplots(1,2)
        axes = axes.ravel()
        
        axes[0].imshow(subset_cave)
        axes[0].set_title("Infinite Bottom")
        
        axes[1].imshow(subset_cave2)
        axes[1].set_title("Cave Bottom")
        
    return count, count2

#%%
# Day 15: Beacon Exclusion Zone

def day15():
    with open("input15.txt") as f:
        raw_readings = f.readlines()
        
    readings = [x.strip().split() for x in raw_readings]
    readings = [[[x[3],x[2]], [x[9],x[8]]] for x in readings]
    readings = [[[int(re.sub(r"[^0-9-]","",x)) for x in y] for y in z] for z in readings]
    
    beacons = [x[1] for x in readings]
    sens_loc_beac_dist = [[x[0], abs(x[0][0]-x[1][0]) + abs(x[0][1]-x[1][1])] for x in readings]
    
    
    row_in_q = 2000000
    relevant = []
    for beacon_reading in sens_loc_beac_dist:
        min_row = beacon_reading[0][0] - beacon_reading[1]
        max_row = beacon_reading[0][0] + beacon_reading[1]
        if row_in_q >= min_row and row_in_q <= max_row:
            relevant.append(beacon_reading)
    
    not_in_ranges = []
    for beacon_reading in relevant:
        b = beacon_reading[0][1]
        row_diff = abs(beacon_reading[0][0] - row_in_q)
        dist = beacon_reading[1]
        not_in_ranges.append([b-abs(dist-row_diff), b+abs(dist-row_diff)])
        
    not_in = set()
    for eliminated in not_in_ranges:
        for i in range(eliminated[0],eliminated[1]+1):
            not_in.add(i)
    
    final_not_in = set()
    for i in not_in:
        if [row_in_q, i] not in beacons:
            final_not_in.add(i)
            
    row_in_q_not_in = len(final_not_in)
    
    def check_dist(point1, point2):
        return abs(point1[0]-point2[0]) + abs(point1[1]-point2[1])
    
    boundary = 4000000
    for beacon_reading in sens_loc_beac_dist:
        left_edged_out = list(range(beacon_reading[0][1]-beacon_reading[1]-1, beacon_reading[0][1]+1))
        top_edged_out = list(range(beacon_reading[0][0], beacon_reading[0][0]-beacon_reading[1]-2, -1))
        bottom_edged_out = list(range(beacon_reading[0][0],beacon_reading[0][0]+beacon_reading[1]+2))
        right_edged_out = list(range(beacon_reading[0][1]+beacon_reading[1]+1, beacon_reading[0][1]-1, -1))
        
        edged_out_tl = [[top_edged_out[i], left_edged_out[i]] for i in range(len(left_edged_out))]
        edged_out_tr = [[top_edged_out[i], right_edged_out[i]] for i in range(len(left_edged_out))]
        edged_out_br = [[bottom_edged_out[i], right_edged_out[i]] for i in range(len(left_edged_out))]
        edged_out_bl = [[bottom_edged_out[i], left_edged_out[i]] for i in range(len(left_edged_out))]
        edged_out_new = [tuple(x) for x in edged_out_tl + edged_out_tr + edged_out_br + edged_out_bl]
        
        edged_out = set(edged_out_new)
        
        for e in edged_out:
            if e[0] < 0 or e[0] > boundary or e[1] < 0 or e[1] > boundary:
                continue
            this_one = True
            for beacon_reading2 in sens_loc_beac_dist:
                if check_dist(e, beacon_reading2[0]) <= beacon_reading2[1]:
                    this_one = False
                    break
            if this_one:
                the_beacon = e
                break
        if this_one:
            break
                
    tuning_freq = the_beacon[1]*4000000+the_beacon[0]
    
    return row_in_q_not_in, tuning_freq
    
#%%
# Day 16: Proboscidea Volcanium

def day16():
    with open("input16.txt") as f:
        raw_pipe_map = f.readlines()
        
    pipe_map = [x.strip().split() for x in raw_pipe_map]
    pipe_map = [[x[1], int(re.sub(r"[^0-9]","",x[4])), [y[:2] for y in x[9:]]] for x in pipe_map]
    
    pressure_sorted = sorted([[x[1],x[0]] for x in pipe_map], reverse = True)
    pressure_pipes = [x[1] for x in pressure_sorted if x[0] > 0]
    pipes_to_pressure = {x[1]:x[0] for x in pressure_sorted}
    
    pipe_to_pipes = {x[0]:x[2] for x in pipe_map}
    
    def get_shortest(orig, dest, pipe_to_pipes):
        paths  = [[orig]]
        found = False
        while True:
            new_paths = []
            for path in paths:
                for possible in pipe_to_pipes[path[-1]]:
                    if possible in path:
                        continue
                    new_paths.append(path + [possible])                           
                    if possible == dest:
                        shortest = path + [possible]
                        found = True
                        break      
                if found:
                    break
            paths = deepcopy(new_paths)   
            if found:
                break
        return shortest
    
    orig = "AA"
    shortest_paths = {}
    for dest in pressure_pipes:
        shortest_paths[(orig, dest)] = get_shortest(orig, dest, pipe_to_pipes)
    
    for orig in pressure_pipes:
        for dest in pressure_pipes:
            if [orig, dest] in [x[0] for x in shortest_paths] or orig == dest:
                continue
            shortest_paths[(orig, dest)] = get_shortest(orig, dest, pipe_to_pipes)
    
    class path():
        def __init__(self, pos = "AA", pressure_released = 0, time_left = 30, visited = ["AA"], shortest_paths = shortest_paths, pipes_to_pressure = pipes_to_pressure):
            self.pos = pos
            self.pressure_released = pressure_released
            self.time_left = time_left
            self.visited = visited
            self.shortest_paths = shortest_paths
            self.pipes_to_pressure = pipes_to_pressure
            
        def __str__(self):
            return f"{self.pos}, {self.pressure_released}, {self.time_left}, {self.visited}"
        
        def __repr__(self):
            return self.__str__()
            
        def copy(self):
            return path(self.pos, self.pressure_released, self.time_left, self.visited.copy())
        
        def can_move_to(self, pipe):
            from_to = (self.pos, pipe)        
            shortest_path = self.shortest_paths[from_to]
            if len(shortest_path) > self.time_left:
                return False
            else:
                return True
            
        def move_to(self, pipe):
            from_to = (self.pos, pipe)
            self.pos = pipe
            shortest_path = self.shortest_paths[from_to]
            self.time_left -= len(shortest_path)
            self.pressure_released += self.pipes_to_pressure[self.pos]*self.time_left
            self.visited.append(self.pos)
            return self
    
    attempts = [path()]    
    movement = True
    i = 0
    while movement:
        i += 1
        new_attempts = []
        movement = False
        for attempt in attempts:
            for pipe in pressure_pipes:
                if pipe in attempt.visited:
                    continue
                else:
                    if attempt.can_move_to(pipe):
                        movement = True
                        new_attempts.append(attempt.copy().move_to(pipe))
        sorted_new_attempts = sorted(new_attempts, reverse = True, key = lambda x: x.pressure_released)
        if movement:
            attempts = sorted_new_attempts[:40]
            
    max_pressure_released = attempts[0].pressure_released
    
    
    attempts = [[path(time_left = 26), path(time_left = 26)]]    
    movement = True
    i = 0
    while movement:
        i += 1
        new_attempts = []
        movement = False
        for attempt in attempts:
            me = attempt[0]
            eleph = attempt[1]
            for pipe in pressure_pipes:
                if pipe in me.visited + eleph.visited:
                    continue
                else:
                    if me.can_move_to(pipe) and len(shortest_paths[(me.pos, pipe)]) <= len(shortest_paths[(eleph.pos, pipe)]):
                        movement = True
                        new_attempts.append((me.copy().move_to(pipe), eleph))
                    elif eleph.can_move_to(pipe) and len(shortest_paths[(me.pos, pipe)]) > len(shortest_paths[(eleph.pos, pipe)]):
                        movement = True
                        new_attempts.append((me, eleph.copy().move_to(pipe)))
        sorted_new_attempts = sorted(new_attempts, reverse = True, key = lambda x: x[0].pressure_released + x[1].pressure_released)
        if movement:
            attempts = sorted_new_attempts[:1800]
            
    max_pressure_released_2 = attempts[0][0].pressure_released + attempts[0][1].pressure_released
    
    return max_pressure_released, max_pressure_released_2
            

#%%
# Day 17: Pyroclastic Flow

def day17():
    with open("input17.txt") as f:
        raw_gusts = f.readlines()
        
    
    def make_repeater(all_things, include_num = False):
        while True:
            for i, thing in enumerate(all_things):
                if not include_num:
                    yield thing
                else:
                    yield thing, i
    
            
    all_gusts = [x for x in raw_gusts[0].strip()]
    gust = make_repeater(all_gusts)
    
    rock1 = np.zeros([4,7])
    rock1[0,2:6] = 1
    rock2 = np.zeros([6,7])
    rock2[0,3] = 1
    rock2[1,2:5] = 1
    rock2[2,3] = 1
    rock3 = np.zeros([6,7])
    rock3[0,4] = 1
    rock3[1,4] = 1
    rock3[2,2:5] = 1
    rock4 = np.zeros([7,7])
    rock4[0:4,2] = 1
    rock5 = np.zeros([5,7])
    rock5[0:2,2:4] = 1
    rock = make_repeater([rock1, rock2, rock3, rock4, rock5])
    
    def fall(falling, chamber):
        falling = falling.copy()
        falling = np.vstack((np.zeros([1,7]),falling))
        if 2 in falling[:-1,:] + chamber:
            return None
        else:
            return falling[:-1,:]
    
    def gust_push(falling, chamber, gust):
        before_gust = falling.copy()
        falling = falling.copy()
        if gust == "<":
            if 1 in falling[:,0]:
                return before_gust
            else:
                falling = np.hstack((falling,np.zeros([falling.shape[0],1])))
                if 2 in chamber + falling[:,1:]:
                    return before_gust
                else:
                    return falling[:,1:]
        elif gust == ">":
            if 1 in falling[:,-1]:
                return before_gust
            else:
                falling = np.hstack((np.zeros([falling.shape[0],1]),falling))
                if 2 in chamber + falling[:,:-1]:
                    return before_gust
                else:
                    return falling[:,:-1]
        else:
            raise Exception("Invalid gust type")
                
    height = []
    chamber = np.ones([1,7])
    for i in range(2022):
        falling = next(rock)
        rock_type = falling.shape
        falling = np.vstack((falling,np.zeros(chamber.shape)))
        chamber = np.vstack((np.zeros(rock_type),chamber))
        
        while True:
            falling = gust_push(falling, chamber, next(gust))
            if type(fall(falling, chamber)) == type(None):
                break
            falling = fall(falling, chamber)
            
        chamber = falling.copy() + chamber
        chamber = chamber[[any(a) == 1 for a in chamber],:]
        
        height.append(chamber.shape[0]-1)
    
    height_2022 = height[2022-1]
    
    height_change = [height[i] - height[i-1] for i in range(1,len(height))]
    height_change = [height[0]] + height_change
    
    height_change_str = "".join([str(x) for x in height_change])
    
    search_len = 100
    search_start = 0
    while True:
        search_result = re.search(height_change_str[search_start:search_start+search_len], height_change_str[search_start+search_len:])
        
        if type(search_result) != type(None):
            break
        
        search_start += 1
    
    repeat_start = (search_start + search_len + search_result.start()) 
    cycle_length = repeat_start - search_start
    
    num_rock = int(1e12)
    
    num_rock_height = height[search_start-1] + ((num_rock-search_start-1)//cycle_length)*(height[repeat_start]-height[search_start]) + height[search_start+(num_rock-search_start-1)%cycle_length] - height[search_start-1]
    
    return height_2022, num_rock_height

#%%
# Day 18: Boiling Boulders

def day18(max_progress_updates = 0, visualize = False):
    raw_points = []
    with open("input18.txt") as f:
        for l in f:
            new = [int(x) for x in l.strip().split(",")]
            raw_points.append(new)
    
    points = deepcopy(raw_points)
    
    outside_points = []
    min_x = min([x[0] for x in points])
    max_x = max([x[0] for x in points])
    min_y = min([x[1] for x in points])
    max_y = max([x[1] for x in points])
    min_z = min([x[2] for x in points])
    max_z = max([x[2] for x in points])
    
    for x in [min_x - 2, max_x + 2]:
        for y in range(min_y-2, max_y+3):
            for z in range(min_z-2, max_z+3):
                outside_points.append([x,y,z])
    
    for y in [min_y - 2, max_y + 2]:
        for x in range(min_x-2, max_x+3):
            for z in range(min_z-2, max_z+3):
                outside_points.append([x,y,z])
    
            
    for z in [min_z - 2, max_z + 2]:
        for x in range(min_x-2, max_x+3):
            for y in range(min_y-2, max_y+3):
                outside_points.append([x,y,z])
      
    def path_to_outside(from_point, to_points, blocked_points, good_points, second_attempt = False):
        def flatten(array):
            new = []
            for l in array:
                for e in l:
                    new.append(e)
            return new
        
        path  = [[from_point]]
        while True:
            new_path = []
            still_moving = False
            for path in path:
                curr = path[-1]
                left = [curr[0]-1, curr[1], curr[2]]
                right = [curr[0]+1, curr[1], curr[2]]
                inward = [curr[0], curr[1]-1, curr[2]]
                outward = [curr[0], curr[1]+1, curr[2]]
                down = [curr[0], curr[1], curr[2]-1]
                up = [curr[0], curr[1], curr[2]+1]
                if not second_attempt:
                    options = [up, left, right, inward, outward, down]
                else:
                    options = [left, right, inward, outward, down, up]
                for possible in options:
                    if possible in flatten(new_path) or possible in blocked_points or possible in path:
                        continue
                    else:
                        next_point = possible
                        new_path.append(path + [next_point])
                        still_moving = True
                        break                      
                if still_moving and (next_point in to_points or next_point in good_points):
                    good_points = list(set([tuple(x) for x in good_points])) + list(set([tuple(x) for x in flatten(new_path)]))
                    return True, good_points   
            if not still_moving:
                if curr != from_point and not second_attempt:
                    second_try, second_try_good_points = path_to_outside(curr, to_points, blocked_points, good_points, second_attempt = True)
                    if second_try:
                        return True, second_try_good_points
                    else:
                        return False, good_points
                else:
                    return False, good_points                
            path = deepcopy(new_path)
    
    free_faces = 0
    external_faces = 0
    good_points = []
    for p, point in enumerate(points):
        if max_progress_updates > 0 and p > 0:
            if len(points) > max_progress_updates:
                if p%(len(points)//max_progress_updates) == 0: 
                    print(f"{(p+1)/len(points):.1%}")
            else:
                print(f"{(p+1)/len(points):.1%}")
        
        x = point[0]
        y = point[1]
        z = point[2]
        
        for i, x_add in enumerate([-1,0,1]):
            for j, y_add in enumerate([-1,0,1]):
                for k, z_add in enumerate([-1,0,1]):
                    
                    if[i,j,k].count(1) != 2:
                        continue
                    
                    from_face_point = [x+x_add, y+y_add, z+z_add]                
                    if from_face_point not in points:
                        free_faces += 1
                        if from_face_point in good_points:
                            external_faces += 1            
                        else: 
                            result, good_points = path_to_outside(from_face_point, outside_points, points, good_points)
                            if result:
                                external_faces += 1
                    
    if visualize:
        point_space = np.zeros([max_x, max_y, max_z])
        for point in points:
            point_space[tuple([point[0]-1,point[1]-1,point[2]-1])] = 1
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        ax.voxels(point_space, edgecolor = "k", facecolor="firebrick")
        
    return free_faces, external_faces

#%%
# Day 19: Not Enough Minerals

def day19():
    blueprints = []
    with open("input19.txt") as f:
        for l in f:
            new = l.strip().split()
            new = [new[6], new[12], new[18], new[21], new[27], new[30]]
            new = [int(x) for x in new]
            blueprints.append(new)
    
    class state():
        def __init__(self, blueprint, existing = [100] + [-1]*8, time=24):
            self.blueprint = blueprint
            self.ore_bot_cost = blueprint[0]
            self.clay_bot_cost = blueprint[1]
            self.obs_bot_cost = [blueprint[2], blueprint[3]]
            self.geo_bot_cost = [blueprint[4], blueprint[5]]
            
            self.max_ore = max([blueprint[0],blueprint[1],blueprint[2],blueprint[4]])
            self.max_clay = blueprint[3]
            self.max_obs = blueprint[5]
            
            self.minutes_left = min(time,existing[0])
            self.ore_bots = max(1, existing[1])
            self.clay_bots = max(0, existing[2])
            self.obs_bots = max(0, existing[3])
            self.geo_bots = max(0, existing[4])
            self.ore = max(0, existing[5])
            self.clay = max(0, existing[6])
            self.obs = max(0, existing[7])
            self.score = max(0, existing[8])
            
        def copy(self):
            current = [self.minutes_left, self.ore_bots, self.clay_bots, self.obs_bots, self.geo_bots, self.ore, self.clay, self.obs, self.score]
            return state(self.blueprint, current, time = self.minutes_left)
            
        def can_add_ore_bot(self):
            if self.ore < self.ore_bot_cost:
                return False
            else:
                return True
        def can_add_clay_bot(self):
            if self.ore < self.clay_bot_cost:
                return False
            else:
                return True
        def can_add_obs_bot(self):
            if self.ore < self.obs_bot_cost[0] or self.clay < self.obs_bot_cost[1]:
                return False
            else:
                return True
        def can_add_geo_bot(self):
            if self.ore < self.geo_bot_cost[0] or self.obs < self.geo_bot_cost[1]:
                return False
            else:
                return True
        
        def add_ore_bot(self):
            if self.ore < self.ore_bot_cost:
                raise Exception("Not enough resources")  
            self.mine_resources()
            self.ore_bots += 1
            self.ore -= self.ore_bot_cost
            self.minutes_left -= 1
            return self
        def add_clay_bot(self):
            if self.ore < self.clay_bot_cost:
                raise Exception("Not enough resources")
            self.mine_resources()
            self.clay_bots += 1
            self.ore -= self.clay_bot_cost
            self.minutes_left -= 1
            return self
        def add_obs_bot(self):
            if self.ore < self.obs_bot_cost[0] or self.clay < self.obs_bot_cost[1]:
                raise Exception("Not enough resources")
            self.mine_resources()
            self.obs_bots += 1
            self.ore -= self.obs_bot_cost[0]
            self.clay -= self.obs_bot_cost[1]
            self.minutes_left -= 1
            return self
        def add_geo_bot(self):
            if self.ore < self.geo_bot_cost[0] or self.obs < self.geo_bot_cost[1]:
                raise Exception("Not enough resources")
            self.mine_resources()
            self.geo_bots += 1
            self.score += self.minutes_left - 1
            self.ore -= self.geo_bot_cost[0]
            self.obs -= self.geo_bot_cost[1]
            self.minutes_left -= 1
            return self
        def do_nothing(self):
            self.mine_resources()
            self.minutes_left -= 1
            return self
        
        def mine_resources(self):
            self.ore += self.ore_bots
            self.clay += self.clay_bots
            self.obs += self.obs_bots
            
        def __repr__(self):
            return f"State: ({self.ore_bots}, {self.clay_bots}, {self.obs_bots}, {self.geo_bots}) ({self.ore}, {self.clay}, {self.obs}): {self.score}"
        def __str__(self):
            return self.__repr__()
            
        def sortable(self):
            part1 = [self.score, self.obs, self.clay, self.ore]
            part2 = [0, self.obs_bots, self.clay_bots, self.ore_bots]
            return tuple([part1[i] + part2[i] for i in range(len(part1))])
    
    def get_max_scores(blueprints, num_rounds):
        max_scores = []
        for b, blueprint in enumerate(blueprints):
            paths = [[state(blueprint, time = num_rounds)]]
            
            most_ore_needed = max([blueprint[0],blueprint[1],blueprint[2],blueprint[4]])
            most_clay_needed = blueprint[3]
            most_obs_needed = blueprint[5]
        
            for i in range(num_rounds-1):
                new_paths = []
                for path in paths:
                    latest = path[-1]
                    
                    to_add = []
                    if latest.can_add_geo_bot():
                        to_add = latest.copy().add_geo_bot()
                        new_paths.append(path.copy() + [to_add])
                        continue
                    
                    if latest.can_add_obs_bot() and latest.obs_bots < most_obs_needed:
                        to_add = latest.copy().add_obs_bot()
                        new_paths.append(path.copy() + [to_add])
                        
                    if latest.can_add_clay_bot() and latest.clay_bots < most_clay_needed:
                        to_add = latest.copy().add_clay_bot()
                        new_paths.append(path.copy() + [to_add])
                        
                    if latest.can_add_ore_bot() and latest.ore_bots < most_ore_needed:
                        to_add = latest.copy().add_ore_bot()
                        new_paths.append(path.copy() + [to_add])
                        
                    to_add = latest.copy().do_nothing()
                    new_paths.append(path.copy() + [to_add])
                            
                sorted_paths = sorted([x for x in new_paths], key=lambda x: x[-1].sortable(), reverse = True)
        
                paths = deepcopy(sorted_paths[:100])
                
            max_scores.append(max([x.score for x in [y[-1] for y in paths]]))
        return max_scores
        
    part1_max_scores = get_max_scores(blueprints, 24)
    quality = sum([part1_max_scores[i]*(i+1) for i in range(len(blueprints))])
    
    part2_max_scores = get_max_scores(blueprints[:3], 32)
    score_prod = np.prod(part2_max_scores)
    
    return quality, score_prod

#%%
# Day 20: Grove Positioning System

def day20():
    numbers = []
    with open("input20.txt") as f:
        for l in f:
            new = int(l.strip())
            numbers.append(new)
    
    class node():
        def __init__(self, num, before = None ,after = None):
            self.num = num
            self.before = before
            self.after = after
    
        def move(self, num_steps, num_nodes = np.inf):
            if num_steps != 0:
                self.before.after = self.after
                self.after.before = self.before
                curr_node = self
                if num_steps < 0:
                    num_steps = (num_steps % (num_nodes-1)) - num_nodes + 1
                    for i in range(-num_steps):
                        curr_node = curr_node.before
                    new_before = curr_node.before
                    curr_node.before.after = self
                    curr_node.before = self
                    self.after = curr_node
                    self.before = new_before
                elif num_steps > 0:
                    num_steps = (num_steps % (num_nodes-1))
                    for i in range(num_steps):
                        curr_node = curr_node.after
                    new_after = curr_node.after
                    curr_node.after.before = self
                    curr_node.after = self
                    self.before = curr_node
                    self.after = new_after
                    
        def __str__(self):
            return f"{self.before.num} -> {self.num} -> {self.after.num}"
        
        def __repr__(self):
            return self.__str__()
        
    
    nodes = [node(x) for x in numbers]
    for i in range(1,len(nodes)-1):
        nodes[i].before = nodes[i-1]
        nodes[i].after = nodes[i+1]
    nodes[0].before = nodes[-1]
    nodes[0].after = nodes[1]
    nodes[-1].before = nodes[-2]
    nodes[-1].after = nodes[0]
    
    num_nodes = len(nodes)
    for n in nodes:
        n.move(n.num, num_nodes)
    
    def after_zero(i, nodes: list):
        num_nodes = len(nodes)
        forward_num = i%num_nodes
        curr = nodes[[True if x.num == 0 else False for x in nodes].index(True)]
        for f in range(forward_num):
            curr = curr.after
        return curr.num
    
    grove_coord = [after_zero(1000, nodes), after_zero(2000, nodes), after_zero(3000, nodes)]
    grove_coord_sum = sum(grove_coord)
    
    
    d_key = 811589153
    
    nodes2 = [node(d_key*x) for x in numbers]
    for i in range(1,len(nodes2)-1):
        nodes2[i].before = nodes2[i-1]
        nodes2[i].after = nodes2[i+1]
    nodes2[0].before = nodes2[-1]
    nodes2[0].after = nodes2[1]
    nodes2[-1].before = nodes2[-2]
    nodes2[-1].after = nodes2[0]
    
    num_nodes2 = len(nodes2)
    for r in range(10):
        for n in nodes2:
            n.move(n.num, num_nodes2)
                  
    grove_coord2 =  [after_zero(1000, nodes2), after_zero(2000, nodes2), after_zero(3000, nodes2)]
    grove_coord_sum2 = sum(grove_coord2)
    
    
    return grove_coord_sum, grove_coord_sum2

#%%
# Day 21: Monkey Math

def day21():
    monkey_dict = {}
    with open("input21.txt") as f:
        for l in f:
            new = l.strip().split(":")
            new = [new[0], new[1].strip()]
            if new[1].isnumeric():
                monkey_dict[new[0]] = int(new[1])
            else:
                inputs = new[1].split(" ")
                inputs = [inputs[0], inputs[2]]
                if "+" in new[1]:
                    monkey_dict[new[0]] = [inputs, "+"]
                elif "-" in new[1]:
                    monkey_dict[new[0]] = [inputs, "-"]
                elif "*" in new[1]:
                    monkey_dict[new[0]] = [inputs, "*"]
                elif "/" in new[1]:
                    monkey_dict[new[0]] = [inputs, "/"]
                else:
                    raise Exception("Invalid input")
                    
    original_monkey_dict = deepcopy(monkey_dict)
    
    def monkey_math(a,b,op):
        if op == "+":
            return a + b
        elif op == "-":
            return a - b
        elif op == "*":
            return a * b
        elif op == "/":
            return a / b
        elif op == "==":
            return a == b
        else:
            raise Exception("Invalid op")
    
    
    numerics = (int, float)
    
    while type(monkey_dict["root"]) not in numerics:
        for monkey in monkey_dict.keys():
            shout = monkey_dict[monkey]
            if type(shout) in numerics:
                continue
            else:
                inp1 = monkey_dict[shout[0][0]]
                inp2 = monkey_dict[shout[0][1]]
                if type(inp1) in numerics and type(inp2) in numerics:
                    monkey_dict[monkey] = monkey_math(inp1, inp2, shout[1]) 
            
            if type(monkey_dict["root"]) in numerics:
                break
    if monkey_dict["root"] == int(monkey_dict["root"]):
        monkey_dict["root"] = int(monkey_dict["root"])
    root_monkey_shout = monkey_dict["root"]
        
    
    monkey_dict = deepcopy(original_monkey_dict)
    monkey_dict["humn"] = "nothing yet"
    wrong_root = monkey_dict["root"]
    new_root = [wrong_root[0], "=="]
    monkey_dict["root"] = new_root
    
    change_made = True
    while change_made:
        change_made = False
        for monkey in monkey_dict.keys():
            if monkey == "humn":
                continue
            
            shout = monkey_dict[monkey]
            if type(shout) in numerics:
                continue
            else:
                inp1 = monkey_dict[shout[0][0]]
                inp2 = monkey_dict[shout[0][1]]
                if type(inp1) in numerics and type(inp2) in numerics:
                    change_made = True
                    monkey_dict[monkey] = monkey_math(inp1, inp2, shout[1])
    
    class monkey_node():
        def __init__(self, name, parents = None, val = None):
            self.name = name
            self.parents = parents
            self.val = val
            
        def __repr__(self):
            return f"{self.name} = {self.val}\n"
        def __str__(self):
            return self.__repr__()
        
    def build_monkey_tree(name):
        if type(monkey_dict[name]) in numerics:
            parents = None
            val = monkey_dict[name]
        elif name == "humn":
            parents = None
            val = None
        else:
            parents =  monkey_dict[name][0]
            val = None
        if parents == None:
            return monkey_node(name, parents, val)
        else:
            monkey_node_parents = []
            for parent in parents:
                monkey_node_parents.append(build_monkey_tree(parent))
            return monkey_node(name, monkey_node_parents, val)
          
    root_monkey = build_monkey_tree("root")
        
    def solve_monkey_math(a, b, ans, op):
        if a == None and b == None:
            raise Exception("Can't Solve")
        if a != None and b != None:
            raise Exception("Why are you trying to solve this?")
        if op == "+":
            if a == None:
                return ans - b
            else:
                return ans - a
        elif op == "-":
            if a == None:
                return ans + b
            else:
                return a - ans
        elif op == "*":
            if a == None:
                return ans/b
            else:
                return ans/a
        elif op == "/":
            if a == None:
                return ans*b
            else:
                return a/ans
        else:
            raise Exception("Invalid op")
    
    #Get root equality value
    if type(root_monkey.parents[0].val) not in numerics:
        humn_side_root_parent = root_monkey.parents[0]
        solve_to = root_monkey.parents[1].val
    elif type(root_monkey.parents[1].val) not in numerics:
        humn_side_root_parent = root_monkey.parents[1]
        solve_to = root_monkey.parents[0].val
    else:
        raise Exception("Really Hard")       
        
    humn_side_root_parent.val = solve_to
    child = humn_side_root_parent
    
    #Solve backwards up to humn
    while True:
        if child.parents == None:
            break
        
        if child.parents[0].val == None:
            unknown_parent = child.parents[0]
        elif child.parents[1].val == None:
            unknown_parent = child.parents[1]
            
        unknown_parent.val = solve_monkey_math(child.parents[0].val, child.parents[1].val, child.val, monkey_dict[child.name][1])
         
        child = unknown_parent
        
    monkey_dict[child.name] = child.val
            
    if monkey_dict["humn"] == int(monkey_dict["humn"]):
        monkey_dict["humn"] = int(monkey_dict["humn"])
    human_should_shout = monkey_dict["humn"]
    
    return root_monkey_shout, human_should_shout

#%%
# Day 22: Monkey Map

def day22():
    field = []
    instructions = ""
    with open("input22.txt") as f:
        for l in f:
            if "#" in l or "." in l:
                new = l.replace("\n","")
                new = new.replace(" ","2")
                new = new.replace(".","0")
                new = new.replace("#","1")
                field.append([int(x) for x in new])
            else:
                new = l.strip()
                instructions = instructions + new
                
    instructions = [x for x in instructions]
    new_instructions = []
    num_inds = [0]
    for i in range(1,len(instructions)):
        is_num = instructions[i].isnumeric()
        if not is_num:
            num = ""
            for ind in num_inds:
                num = num + instructions[ind]
            num = int(num)
            new_instructions.append(num)
            num_inds = []
            new_instructions.append(instructions[i])
        if is_num:
            num_inds.append(i)
    if len(num_inds) != 0:
        num = ""
        for ind in num_inds:
            num = num + instructions[ind]
        num = int(num)
        new_instructions.append(num)
    instructions = new_instructions    
    
    max_len = max([len(x) for x in field])
    new_field = []
    for line in field:
        if len(line) == max_len:
            new_field.append(line)
        else:
            new_field.append(line + [2] * (max_len - len(line)))
    field = new_field
    
    field = np.asarray(field)
    field = np.vstack((2*np.ones([1,field.shape[1]]),field,2*np.ones([1,field.shape[1]])))
    field = np.hstack((2*np.ones([field.shape[0],1]),field,2*np.ones([field.shape[0],1])))
    
    del new_field
    del new_instructions
    
    def get_row_and_col_min_max(field, index):
        row_log = [x for x in range(field.shape[1]) if field[index[0], x] != 2]
        col_log = [x for x in range(field.shape[0]) if field[x, index[1]] != 2]
                   
        return row_log[0], row_log[-1], col_log[0], col_log[-1]
    
    def turn_left(facing):
        if facing == "R":
            return "U"
        elif facing == "U":
            return "L"
        elif facing == "L":
            return "D"
        elif facing == "D":
            return "R"
        else:
            raise Exception("Invalid")
            
    def turn_right(facing):
        if facing == "R":
            return "D"
        elif facing == "D":
            return "L"
        elif facing == "L":
            return "U"
        elif facing == "U":
            return "R"
        else:
            raise Exception("Invalid")
            
    loc = [1,get_row_and_col_min_max(field,[1,1])[0]]
    facing = "R"
    
    for instruct in instructions:
        if type(instruct) == int:
            for i in range(instruct):
                if facing == "R":
                    left_val = field[loc[0],get_row_and_col_min_max(field,loc)[0]]
                    if field[loc[0], loc[1] + 1] == 0:
                        loc = [loc[0], loc[1] + 1]
                    elif field[loc[0], loc[1] + 1] == 2 and left_val == 0:
                        loc = [loc[0], get_row_and_col_min_max(field,loc)[0]]
                    else:
                        break
                elif facing == "U":
                    down_val = field[get_row_and_col_min_max(field,loc)[3], loc[1]]
                    if field[loc[0] - 1, loc[1]] == 0:
                        loc = [loc[0] - 1, loc[1]]
                    elif field[loc[0] - 1, loc[1]] == 2 and down_val == 0:
                        loc = [get_row_and_col_min_max(field,loc)[3], loc[1]]
                    else:
                        break
                elif facing == "L":
                    right_val = field[loc[0],get_row_and_col_min_max(field,loc)[1]]
                    if field[loc[0], loc[1] - 1] == 0:
                        loc = [loc[0], loc[1] - 1]
                    elif field[loc[0], loc[1] - 1] == 2 and right_val == 0:
                        loc = [loc[0], get_row_and_col_min_max(field,loc)[1]]
                    else:
                        break
                elif facing == "D":
                    up_val = field[get_row_and_col_min_max(field,loc)[2], loc[1]]
                    if field[loc[0] + 1, loc[1]] == 0:
                        loc = [loc[0] + 1, loc[1]]
                    elif field[loc[0] + 1, loc[1]] == 2 and up_val == 0:
                        loc = [get_row_and_col_min_max(field,loc)[2], loc[1]]
                    else:
                        break
        elif type(instruct) == str:
            if instruct == "R":
                facing = turn_right(facing)
            elif instruct == "L":
                facing = turn_left(facing)
            else:
                raise Exception("Invalid")
        else:
            raise Exception("Invalid")
            
    facing_scores = {"R":0, "D":1, "L":2, "U":3}
    
    password1 = (loc[0]*1000)+(loc[1]*4)+facing_scores[facing]
    
    
    sideA = []; sideE = []; sideI = []; sideJ = []
    for i in range(51,101):
        sideA.append((1,i))
        sideE.append((150,i))
        sideI.append((i,51))
        sideJ.append((i,100))
    
    sideB = []; sideC = []; sideK = []; sideL = []
    for i in range(101,151):
        sideB.append((1,i))
        sideC.append((50,i))
        sideK.append((i,1))
        sideL.append((i,100))
        
    sideD = []; sideF = []; sideG = []; sideH = []
    for i in range(1,51):
        sideD.append((101,i))
        sideF.append((200,i))
        sideG.append((i,51))
        sideH.append((i,150))
        
    
    sideM = []; sideN = []
    for i in range(151,201):
        sideM.append((i,1))
        sideN.append((i,50))
    
    side_translator = {}
    for a,m in zip(sideA, sideM):
        side_translator[(a, "U")] = [list(m), "R"]
        side_translator[(m, "L")] = [list(a), "D"]
    for b,f in zip(sideB, sideF):
        side_translator[(b, "U")] = [list(f), "U"]
        side_translator[(f, "D")] = [list(b), "D"]
    for c,j in zip(sideC, sideJ):
        side_translator[(c, "D")] = [list(j), "L"]
        side_translator[(j, "R")] = [list(c), "U"]
    for d,i in zip(sideD, sideI):
        side_translator[(d, "U")] = [list(i), "R"]
        side_translator[(i, "L")] = [list(d), "D"]
    for e,n in zip(sideE, sideN):
        side_translator[(e, "D")] = [list(n), "L"]
        side_translator[(n, "R")] = [list(e), "U"]
    for g,k in zip(sideG, sideK[::-1]):
        side_translator[(g, "L")] = [list(k), "R"]
        side_translator[(k, "L")] = [list(g), "R"]
    for h,l in zip(sideH, sideL[::-1]):
        side_translator[(h, "R")] = [list(l), "L"]
        side_translator[(l, "R")] = [list(h), "L"]
        
    loc = [1,get_row_and_col_min_max(field,[1,1])[0]]
    facing = "R"
    
    for instruct in instructions:
        if type(instruct) == int:
            for i in range(instruct):
                if facing == "R":
                    if field[loc[0], loc[1] + 1] == 0:
                        loc = [loc[0], loc[1] + 1]
                    elif field[loc[0], loc[1] + 1] == 2:
                        if field[tuple(side_translator[(tuple(loc), "R")][0])] == 1:
                            break
                        else:
                            facing = side_translator[(tuple(loc), "R")][1]
                            loc = side_translator[(tuple(loc), "R")][0]
                    else:
                        break
                elif facing == "U":
                    if field[loc[0] - 1, loc[1]] == 0:
                        loc = [loc[0] - 1, loc[1]]
                    elif field[loc[0] - 1, loc[1]] == 2:
                        if field[tuple(side_translator[(tuple(loc), "U")][0])] == 1:
                            break
                        else:
                            facing = side_translator[(tuple(loc), "U")][1]
                            loc = side_translator[(tuple(loc), "U")][0]
                    else:
                        break
                elif facing == "L":
                    if field[loc[0], loc[1] - 1] == 0:
                        loc = [loc[0], loc[1] - 1]
                    elif field[loc[0], loc[1] - 1] == 2:
                        if field[tuple(side_translator[(tuple(loc), "L")][0])] == 1:
                            break
                        else:
                            facing = side_translator[(tuple(loc), "L")][1]
                            loc = side_translator[(tuple(loc), "L")][0]
                    else:
                        break
                elif facing == "D":
                    if field[loc[0] + 1, loc[1]] == 0:
                        loc = [loc[0] + 1, loc[1]]
                    elif field[loc[0] + 1, loc[1]] == 2:
                        if field[tuple(side_translator[(tuple(loc), "D")][0])] == 1:
                            break
                        else:
                            facing = side_translator[(tuple(loc), "D")][1]
                            loc = side_translator[(tuple(loc), "D")][0]
                    else:
                        break
        elif type(instruct) == str:
            if instruct == "R":
                facing = turn_right(facing)
            elif instruct == "L":
                facing = turn_left(facing)
            else:
                raise Exception("Invalid")
        else:
            raise Exception("Invalid")
    
    password2 = (loc[0]*1000) + (loc[1]*4) + facing_scores[facing]
    
    return password1, password2

#%%
# Day 23: Unstable Diffusion

def day23():
    print("This one takes forever.. Might want to reconsider running it.\n\nThe answers to parts 1 and 2 were 3990 and 1057.")
    
    init_elves = []
    with open("input23.txt") as f:
        for l in f:
            new = l.strip()
            init_elves.append([x for x in new])
    
    elf_pos = {}
    num_elves = 0
    elves = []
    for i, row in enumerate(init_elves):
        for j, entry in enumerate(row):
            if entry == "#":
                num_elves += 1
                elf_pos[num_elves] = [i, j]
                elves.append(num_elves)
                
    priority_orders = [["N", "S", "W", "E"],
                       ["S", "W", "E", "N"],
                       ["W", "E", "N", "S"],
                       ["E", "N", "S", "W"]]


    def pick_next_spot(priority_order, elf_pos):
        move_count = 0
        priority_order = ["all"] + priority_order
        elf_pos_items = list(elf_pos.items())
        elf_nums = np.array([x[0] for x in elf_pos_items])
        elf_locs = [x[1] for x in elf_pos_items]
        NW = np.array([[x[0]-1, x[1]-1] not in elf_locs for x in elf_locs])
        N = np.array([[x[0]-1, x[1]] not in elf_locs for x in elf_locs])
        NE = np.array([[x[0]-1, x[1]+1] not in elf_locs for x in elf_locs])
        W = np.array([[x[0], x[1]-1] not in elf_locs for x in elf_locs])
        E = np.array([[x[0], x[1]+1] not in elf_locs for x in elf_locs])
        SW = np.array([[x[0]+1, x[1]-1] not in elf_locs for x in elf_locs])
        S = np.array([[x[0]+1, x[1]] not in elf_locs for x in elf_locs])
        SE = np.array([[x[0]+1, x[1]+1] not in elf_locs for x in elf_locs])
        
        temp = deepcopy(elf_locs)
        elf_locs = np.empty(len(temp), dtype=object)
        elf_locs[:] = temp
        
        north_clear = NW * N * NE
        south_clear = SW * S * SE
        west_clear = NW * W * SW
        east_clear = NE * E * SE
        all_clear = NW * N * NE * W * E * SW * S * SE
        clear_sides = {"N":north_clear, "S":south_clear, "W":west_clear, "E":east_clear, "all":all_clear}
        
        all_props = {}
        for priority in priority_order:
            clear_log = clear_sides[priority]
            clear_elf_nums = elf_nums[clear_log]
            clear_elf_locs = elf_locs[clear_log]
            if priority == "N":
                new_elf_pos = [[x[0]-1, x[1]] for x in clear_elf_locs]
                all_props["N"] = dict(zip(clear_elf_nums,new_elf_pos))
            elif priority == "S":
                new_elf_pos = [[x[0]+1, x[1]] for x in clear_elf_locs]
                all_props["S"] = dict(zip(clear_elf_nums,new_elf_pos))
            elif priority == "W":
                new_elf_pos = [[x[0], x[1]-1] for x in clear_elf_locs]
                all_props["W"] = dict(zip(clear_elf_nums,new_elf_pos))
            elif priority == "E":
                new_elf_pos = [[x[0], x[1]+1] for x in clear_elf_locs]
                all_props["E"] = dict(zip(clear_elf_nums,new_elf_pos))
            elif priority == "all":
                new_elf_pos = [x for x in clear_elf_locs]
                all_props["all"] = dict(zip(clear_elf_nums,new_elf_pos))
            else:
                raise Exception("Invalid")            
         
        elf_prop = {}
        for priority in priority_order[::-1]:
            elf_prop = {**elf_prop, **all_props[priority]}
            
        move_count = len(elf_prop) - len(all_props["all"])
            
        return elf_prop, move_count     
    
    
    def show_elves(elf_pos):
        all_pos = list(elf_pos.values())
        all_pos_y = [x[0] for x in all_pos]
        all_pos_x = [x[1] for x in all_pos]
        
        min_x = min(all_pos_x)
        max_x = max(all_pos_x)
        min_y = min(all_pos_y)
        max_y = max(all_pos_y)
        
        view = np.zeros([max_y-min_y+1, max_x-min_x+1])
        
        for x, y in zip(all_pos_x, all_pos_y):
            view[y-min_y,x-min_x] = 1
        
        plt.figure()
        plt.imshow(view)
    
    
    rnd = -1
    while True:
        rnd += 1
        round_number = rnd + 1
        priority_order = priority_orders[rnd%len(priority_orders)]
        elf_prop, move_count = pick_next_spot(priority_order, elf_pos)
            
        prop_locs = list(elf_prop.values())
        
        elf_nums = np.array(list(elf_prop.keys()))
        if move_count == 0:
            break
        
        no_conflict = np.array([prop_locs.count(x) == 1 for x in prop_locs])
    
        temp_prop_locs = deepcopy(prop_locs)   
        prop_locs = np.empty(len(temp_prop_locs), dtype=object)
        prop_locs[:] = temp_prop_locs
        
        elf_pos = {**elf_pos, **dict(zip(elf_nums[no_conflict],prop_locs[no_conflict]))}      
        
        if rnd == 10-1:
            all_pos = list(elf_pos.values())
            all_pos_x = [x[0] for x in all_pos]
            all_pos_y = [x[1] for x in all_pos]
            
            length = max(all_pos_x) - min(all_pos_x) + 1
            width = max(all_pos_y) - min(all_pos_y) + 1
            
            total_spaces = length*width
            free_spaces = total_spaces - len(elf_pos)
    
    return free_spaces, round_number
        
#%%
# Day 24: Blizzard Basin

def day24(visualize = False):
    raw = []
    with open("input24.txt") as f:
        for l in f:
            raw.append([x for x in l.strip()])
            
    up = deepcopy(raw)
    right = deepcopy(raw)
    down = deepcopy(raw)
    left = deepcopy(raw)
    for i, row in enumerate(raw):
        for j, el in enumerate(row):
            if el == "#":
                up[i][j] = 2
                right[i][j] = 2
                down[i][j] = 2
                left[i][j] = 2
            elif el == ".":
                up[i][j] = 0
                right[i][j] = 0
                down[i][j] = 0
                left[i][j] = 0
            elif el == "^":
                up[i][j] = 1
                right[i][j] = 0
                down[i][j] = 0
                left[i][j] = 0
            elif el == ">":
                up[i][j] = 0
                right[i][j] = 1
                down[i][j] = 0
                left[i][j] = 0
            elif el == "v":
                up[i][j] = 0
                right[i][j] = 0
                down[i][j] = 1
                left[i][j] = 0
            elif el == "<":
                up[i][j] = 0
                right[i][j] = 0
                down[i][j] = 0
                left[i][j] = 1
            else:
                raise Exception(f"Invalid: {el}")
                
    up = np.asarray(up)
    right = np.asarray(right)
    down = np.asarray(down)
    left = np.asarray(left)
    
    start = (0,1)
    end = (up.shape[0]-1, up.shape[1]-2)
    
    def move_storm(storm_mat, storm_type):
        storm_body = storm_mat[1:-1,1:-1]
        storm_top = storm_mat[0,1:-1].reshape(1,-1)
        storm_bottom = storm_mat[-1,1:-1].reshape(1,-1)
        storm_left = storm_mat[:,0].reshape(-1,1)
        storm_right = storm_mat[:,-1].reshape(-1,1)
        
        if storm_type == "^":
            new_storm = np.vstack([storm_body[1:,:],storm_body[0,:]])
        elif storm_type == ">":
            new_storm = np.hstack([storm_body[:,-1:],storm_body[:,:-1]])
        elif storm_type == "v":
            new_storm = np.vstack([storm_body[-1:,:],storm_body[:-1,:]])
        elif storm_type == "<":
            new_storm = np.hstack([storm_body[:,1:], storm_body[:,:1]])
            
        new_storm = np.vstack([storm_top, new_storm, storm_bottom])
        new_storm = np.hstack([storm_left, new_storm, storm_right])
        
        return new_storm
    
    def move_storms(up, down, left, right):
        new_up = move_storm(up, "^")
        new_down = move_storm(down, "v")
        new_left = move_storm(left, "<")
        new_right = move_storm(right, ">")
        
        return new_up, new_down, new_left, new_right
    
    locs = [start]
    minutes = 0
    crossed = False
    back = False
    while True:
        minutes += 1
        up, down, left, right = move_storms(up, down, left, right)
        storms = up + down + left + right
        new_locs = []
        for loc in locs:
            if loc != start and storms[loc[0]-1,loc[1]] == 0:
                new_locs.append((loc[0]-1,loc[1]))
            if loc != end and storms[loc[0]+1,loc[1]] == 0:
                new_locs.append((loc[0]+1,loc[1]))
            if storms[loc[0],loc[1]-1] == 0:
                new_locs.append((loc[0],loc[1]-1))
            if storms[loc[0],loc[1]+1] == 0:
                new_locs.append((loc[0],loc[1]+1))
            if storms[loc[0],loc[1]] == 0:
                new_locs.append((loc[0],loc[1]))    
        locs = list(set(new_locs))
        
        if visualize:
            plt.cla()
            plt.imshow(storms, vmin = 0, vmax = 4)
            plt.scatter([x[1] for x in locs], [x[0] for x in locs], c = "indianred", alpha = 0.5)
            plt.pause(0.05)
        
        if end in locs and not crossed:
            crossed = True
            locs = [(up.shape[0]-1, up.shape[1]-2)]
            to_cross = minutes
        
        if start in locs and crossed and not back:
            back = True
            locs = [(0,1)]
            
        if end in locs and back:
            double_cross = minutes
            break
            
    return to_cross, double_cross

#%%
# Day 25: Full of Hot Air

def day25():
    snafus = []
    with open("input25.txt") as f:
        for l in f:
            snafus.append(l.strip())
            
    def snafu_to_num(snafu):
        convert = {"2": 2, "1":1, "0":0, "-":-1, "=":-2} 
        num = 0
        for i, s in enumerate(snafu[::-1]):
            num += (5**i)*convert[s]
        return num
    
    total = 0
    for snafu in snafus:
        total += snafu_to_num(snafu)
        
    def num_to_snafu(num):
        total = num
        max_degree = 0
        while total > 2*(5**max_degree):
            max_degree += 1
        
        convert = {-2:"=", -1:"-", 0:"0", 1:"1", 2:"2"}
        
        total_snafu = ""
        running_total = total
        for d in range(max_degree, -1, -1):
            options = [-2,-1,0,1,2]
            mults = [x*(5**d) for x in options]
            dist_to_mults = [abs(running_total-x) for x in mults]
            ind = dist_to_mults.index(min(dist_to_mults))
            
            next_mults = [x*(5**(d-1)) for x in options]
            
            if running_total >= min(next_mults) and running_total <= max(next_mults):
                total_snafu += "0"        
            else:
                running_total -= mults[ind]
                total_snafu += convert[options[ind]]
                    
        return total_snafu
    
    bob_snafu = num_to_snafu(total)
    return bob_snafu