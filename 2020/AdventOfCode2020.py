#%% 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation  
import re
from itertools import product
from math import prod

#%%
# Timer Wrapper Function

def time_this_func(func):
    from time import time
    def timed_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        print(f"{time()-t1:0.3f} s runtime")
        return result
    return timed_func

#%%
# Day 1: Report Repair

@time_this_func
def day1():
    entries = []
    with open("input1.txt") as f:
        for l in f:
            entries.append(int(l))
    
    mult_found = False
    for i, num1 in enumerate(entries):
        for j, num2 in enumerate(entries):
            if i == j:
                continue
            
            if num1 + num2 == 2020:
                mult = num1 * num2
                mult_found = True
                break
        
        if mult_found:
            break
        
    mult1 = mult
        
    
    mult_found = False
    for i, num1 in enumerate(entries):
        for j, num2 in enumerate(entries):
            for k, num3 in enumerate(entries):
                if len({i, j, k}) < 3:
                    continue
                
                if num1 + num2 + num3 == 2020:
                    mult = num1 * num2 * num3
                    mult_found = True
                    break
            
            if mult_found:
                break
        
        if mult_found:
            break
        
    mult2 = mult
    
    
    return mult1, mult2

#%%
# Day 2: Password Philosophy

@time_this_func
def day2():
    pass_info = []
    with open("input2.txt") as f:
        for l in f:
            new = l.split(": ")
            req_info = new[0].split()
            req_letter = req_info[1]
            req_range = [int(x) for x in req_info[0].split("-")]
            password = new[1]
            
            pass_info.append(((tuple(req_range),req_letter),password))
            
    valid_passwords1 = 0
    for info in pass_info:
        if info[0][0][0] <= info[1].count(info[0][1]) <= info[0][0][1]:
            valid_passwords1 += 1
    
    
    valid_passwords2 = 0
    for info in pass_info:
        if (info[1][info[0][0][0]-1] == info[0][1]) != (info[1][info[0][0][1]-1] == info[0][1]):
            valid_passwords2 += 1
            
    
    return valid_passwords1, valid_passwords2

#%%
# Day 3: Toboggan Trajectory

@time_this_func
def day3():
    forest = []
    with open("input3.txt") as f:
        for l in f:
            forest.append([x for x in l.strip()])
    forest = np.array(forest)
    
    trees = 0
    loc = (0,0)
    while loc[0] < forest.shape[0]:
        if forest[loc] == "#":
            trees += 1
        loc = (loc[0]+1, (loc[1]+3)%forest.shape[1])
        
    trees13 = trees
        
    
    slopes = [(1,1), (1,3), (1,5), (1,7), (2,1)]
    tree_counts = []
    for slope in slopes:
        if slope == (1,3):
            tree_counts.append(trees13)
        else:
            trees = 0
            loc = (0,0)
            while loc[0] < forest.shape[0]:
                if forest[loc] == "#":
                    trees += 1
                loc = (loc[0]+slope[0], (loc[1]+slope[1])%forest.shape[1])
            tree_counts.append(trees)
        
    tree_mult = np.prod(tree_counts, dtype = np.int64)
    
    
    return trees13, tree_mult

#%%
# Day 4: Passport Processing

@time_this_func
def day4():
    with open("input4.txt") as f:
        passports = f.read()      
    passports = passports.split("\n\n")
    
    valid_passports = 0
    initially_valid_passports = []
    for passport in passports:
        if "byr:" in passport and \
            "iyr:" in passport and \
            "eyr:" in passport and \
            "hgt:" in passport and \
            "hcl:" in passport and \
            "ecl:" in passport and \
            "pid:" in passport:
                valid_passports += 1
                initially_valid_passports.append(passport)
    
    valid_passports1 = valid_passports
    
    
    valid_passports = 0
    for passport in initially_valid_passports:
        passport_fields = sorted(passport.split())
        passport_dict = {}
        bad_format = False
        for field in passport_fields:
            key_value = field.split(":")
            if key_value[0] == "cid":
                continue
            elif key_value[0] == "byr" and not key_value[1].isnumeric():
                bad_format = True
                break
            elif key_value[0] == "iyr" and not key_value[1].isnumeric():
                bad_format = True
                break
            elif key_value[0] == "eyr" and not key_value[1].isnumeric():
                bad_format = True
                break
            elif key_value[0] == "hgt" and key_value[1][-2:] not in {"cm","in"}:
                bad_format = True
                break
            elif key_value[0] == "pid" and not key_value[1].isnumeric():
                bad_format = True
                break
             
            if key_value[1].isnumeric() and key_value[0] != "pid":
                passport_dict[key_value[0]] = int(key_value[1])
            else:
                passport_dict[key_value[0]] = key_value[1]
        
        if bad_format:
            continue
        
        if passport_dict["byr"] < 1920 or passport_dict["byr"] > 2002:
            continue
        
        if passport_dict["iyr"] < 2010 or passport_dict["iyr"] > 2020:
            continue
        
        if passport_dict["eyr"] < 2020 or passport_dict["eyr"] > 2030:
            continue
        
        height = int(passport_dict["hgt"][:-2])
        units = passport_dict["hgt"][-2:]
        if units == "cm":
            if height < 150 or height > 193:
                continue
        elif units == "in":
            if height < 59 or height > 76:
                continue
        
        hair_color = passport_dict["hcl"]
        if hair_color[0] != "#":
            continue
            
        accepted = {x for x in "0123456789abcdef"}
        not_accepted = False
        for n, c in enumerate(hair_color[1:]):
            if c not in accepted:
                not_accepted = True

        if not_accepted or n != 5:
            continue
                
        if passport_dict["ecl"] not in {"amb", "blu", "brn", "gry", "grn", "hzl", "oth"}:
            continue
        
        if len(passport_dict["pid"]) != 9:
            continue
        
        valid_passports += 1
    
    valid_passports2 = valid_passports
    
    
    return valid_passports1, valid_passports2

#%%
# Day 5: Binary Boarding

@time_this_func
def day5():
    passes = []
    with open("input5.txt") as f:
        for l in f:
            passes.append(l.strip())
    
    row_dividers = [pow(2,i) for i in range(7-1,-1,-1)]
    col_dividers = [pow(2,i) for i in range(3-1,-1,-1)]
    seat_IDs = []
    for p in passes:
        row_bins = p[:7]
        col_bins = p[-3:]
        
        row = 0
        for row_bin, row_divider in zip(row_bins, row_dividers):
            if row_bin == "B":
                row += row_divider
                
        col = 0
        for col_bin, col_divider in zip(col_bins, col_dividers):
            if col_bin == "R":
                col += col_divider
                       
        seat_IDs.append(8*row + col)
        
    max_seat_ID = max(seat_IDs)
    
    
    seat_IDs = sorted(seat_IDs)
    for i in range(len(seat_IDs)):
        if seat_IDs[i] + 1 != seat_IDs[i+1]:
            break
        
    my_seat_ID = seat_IDs[i]+1
    
    
    return max_seat_ID, my_seat_ID

#%%
# Day 6: Custom Customs

@time_this_func
def day6():
    with open("input6.txt") as f:
        groups = f.read().split("\n\n")
        
    unique_questions = [len(set(x.replace("\n",""))) for x in groups]
    sum_total = sum(unique_questions)
    
    
    group_sets = [[set(y) for y in x.split()] for x in groups]
    
    shared_questions = [len(set.intersection(*x)) for x in group_sets]
    sum_shared = sum(shared_questions)
    
    
    return sum_total, sum_shared

#%%
# Day 7: Handy Haversacks

@time_this_func
def day7():
    rules = {}
    with open("input7.txt") as f:
        for l in f:
            new = l.strip().split(" bags contain ")
            contained_bags_raw = new[1].split(", ")
            contained_bags = []
            if "no other bags" not in contained_bags_raw[0]:
                for contained_bag in contained_bags_raw:
                    num = int(contained_bag[:contained_bag.index(" ")+1])
                    bag_color = contained_bag[contained_bag.index(" "):contained_bag.index("bag")].strip()
                    contained_bags.append((num, bag_color))
            rules[new[0]] = contained_bags
            
    def contains_shiny_gold(color):
        contains_colors = {x[1] for x in rules[color]}
        if "shiny gold" in contains_colors:
            return True
        
        for contained_color in contains_colors:
            if contains_shiny_gold(contained_color):
                return True
        return False
    
    can_contain_shiny_gold = 0
    for color in rules:
        if contains_shiny_gold(color):
            can_contain_shiny_gold += 1
            
    
    def count_bags_inside(color, num):
        if len(rules[color]) == 0:
            return 0
        
        bags_inside = 0
        for number, contained_color in rules[color]:
            bags_inside += num * number
            bags_inside += num * count_bags_inside(contained_color, number)
            
        return bags_inside
    
    inside_shiny_gold = count_bags_inside("shiny gold", 1)
    
    
    return can_contain_shiny_gold, inside_shiny_gold

#%%
# Day 8: Handheld Halting

@time_this_func
def day8():
    operations = []
    with open("input8.txt") as f:
        for l in f:
            new = l.strip().split(" ")
            operations.append((new[0],int(new[1])))
    
    accumulator = 0
    executed = set()
    i = 0
    i_range = set(range(len(operations)))
    while i in i_range:
        op = operations[i]
        
        if i in executed:
            break
        else:
            executed.add(i)
        
        if op[0] == "acc":
            accumulator += op[1]
            i += 1
            
        elif op[0] == "jmp":
            i += op[1]
            
        elif op[0] == "nop":
            i += 1
            
    accumulator1 = accumulator
    
            
    for op_ind in i_range: 
        changed_operations = operations.copy()
        
        if changed_operations[op_ind][0] == "acc":
            continue
        elif changed_operations[op_ind][0] == "jmp":
            changed_operations[op_ind] = ("nop", changed_operations[op_ind][1])
        elif changed_operations[op_ind][0] == "nop":
            changed_operations[op_ind] = ("jmp", changed_operations[op_ind][1])
        
        accumulator = 0
        executed = set()
        i = 0
        infinite = False
        while i in i_range:
            op = changed_operations[i]
            
            if i in executed:
                infinite = True
                break
            else:
                executed.add(i)
            
            if op[0] == "acc":
                accumulator += op[1]
                i += 1
                
            elif op[0] == "jmp":
                i += op[1]
                
            elif op[0] == "nop":
                i += 1
        
        if not infinite:
            break
        
    accumulator2 = accumulator
    
    
    return accumulator1, accumulator2

#%%
# Day 9: Encoding Error

@time_this_func
def day9():
    from itertools import combinations
    
    with open("input9.txt") as f:
        output = [int(x) for x in f.readlines()]
    
    for i in range(25,len(output)):
        num = output[i]
        
        preamble = output[i-25:i]
        preamble_adds = {sum(x) for x in combinations(preamble, 2)}
        
        if num not in preamble_adds:
            break
        
    goal_num = num
    
    
    for i in range(len(output)):
        nums = [output[i]]
        j = 0
        while sum(nums) < goal_num:
            j += 1
            nums.append(output[i+j])
        
        if sum(nums) == goal_num:
            break
    
    min_plus_max = min(nums) + max(nums)
    
    
    return goal_num, min_plus_max

#%%
# Day 10: Adapter Array

@time_this_func
def day10():
    with open("input10.txt") as f:
        adaptors = sorted([int(x) for x in f.readlines()])
    adaptors = [0] + adaptors + [max(adaptors)+3]
    
    step_ups = {1:0, 2:0, 3:0}
    for i in range(1,len(adaptors)):
        step_up = adaptors[i] - adaptors[i-1]
        step_ups[step_up] += 1
    
    ones_times_threes = step_ups[1]*step_ups[3]
    
    
    def get_after(adaptor):
        after = []
        for a in adaptors:
            if adaptor < a <= adaptor + 3:
                after.append(a)
        return after
    
    def num_paths(adaptor):
        if adaptor in paths_to_finish:
            return paths_to_finish[adaptor]
        
        paths = 0
        for a in get_after(adaptor):
            paths += num_paths(a)   
            
        return paths
    
    paths_to_finish = {adaptors[-1]:1}
    for adaptor in adaptors[-2::-1]:
        paths_to_finish[adaptor] = num_paths(adaptor)
        
    total_paths = paths_to_finish[0]
    
    
    return ones_times_threes, total_paths

#%%
# Day 11: Seating System

@time_this_func
def day11(visualize = False):
    state = []
    with open("input11.txt") as f:
        for l in f:
            new = l.strip()
            new = new.replace(".","0")
            new = new.replace("L","1")
            new = new.replace("#","2")
            state.append([int(x) for x in new])
    state = np.array(state)
    state = np.pad(state, 1, "constant", constant_values = 0)
    
    def get_adjacents1(i, j):
        adjacents = []
        for r in [i-1, i, i+1]:
            for c in [j-1, j, j+1]:
                if r == i and c == j:
                    continue
                adjacents.append(last_state[r,c])
        return adjacents
    
    last_state = state.copy()
    if visualize:   history1 = [last_state.copy()]
    while True:
        new_state = last_state.copy()
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if last_state[i,j] == 1:
                    if 2 not in get_adjacents1(i, j):
                        new_state[i,j] = 2
                
                elif last_state[i,j] == 2:
                    if get_adjacents1(i, j).count(2) >= 4:
                        new_state[i,j] = 1
        
        if visualize:   history1.append(new_state.copy())
        
        if np.array_equal(last_state, new_state):
            break
        
        last_state = new_state
    
    num_occupied1 = np.sum(new_state == 2)
    
    
    i_options = set(range(state.shape[0]))
    j_options = set(range(state.shape[1]))
    def get_adjacents2(i, j):
        adjacents = []
        for ns in [-1,0,1]:
            for we in [-1,0,1]:
                if ns == 0 and we == 0:
                    continue
                
                search_i = i
                search_j = j
                while True:
                    search_i += ns
                    search_j += we
                    
                    if search_i not in i_options or search_j not in j_options:
                        adjacents.append(0)
                        break
                    
                    if last_state[search_i, search_j] in {1,2}:
                        adjacents.append(last_state[search_i, search_j])
                        break
        
        return adjacents
    
    last_state = state.copy()
    if visualize:   history2 = [last_state.copy()]
    while True:
        new_state = last_state.copy()
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if last_state[i,j] == 1:
                    if 2 not in get_adjacents2(i, j):
                        new_state[i,j] = 2
                
                elif last_state[i,j] == 2:
                    if get_adjacents2(i, j).count(2) >= 5:
                        new_state[i,j] = 1
        
        if visualize:   history2.append(new_state.copy())
        
        if np.array_equal(last_state, new_state):
            break
    
        last_state = new_state
    
    num_occupied2 = np.sum(new_state == 2)
    
    if visualize:
        fig, ax = plt.subplots(1,2)
        ims = []
        axes = ax.ravel()
        for i in range(0,max(len(history1),len(history2)),2):
            if i in range(len(history1)):
               im1 = axes[0].imshow(history1[i], animated=True, vmin = 0, vmax = 3)
            if i in range(len(history2)):
               im2 = axes[1].imshow(history2[i], animated=True, vmin = 0, vmax = 3)
               
            
            
            ims.append([im1, im2])
        
        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=10)
    
    
    if not visualize:
        return num_occupied1, num_occupied2
    else:
        return num_occupied1, num_occupied2, ani

#%%
# Day 12: Rain Risk

@time_this_func
def day12():
    operations = []
    with open("input12.txt") as f:
        for l in f:
            operations.append((l[0],int(l[1:])))
            
    def turn(direction, degrees, current_facing):
        now_facing = current_facing
        for _ in range(degrees//90):
            if direction == "R":
                if now_facing == "N":
                    now_facing = "E"
                elif now_facing == "E":
                    now_facing = "S"
                elif now_facing == "S":
                    now_facing = "W"
                elif now_facing == "W":
                    now_facing = "N"
            elif direction == "L":
                if now_facing == "N":
                    now_facing = "W"
                elif now_facing == "W":
                    now_facing = "S"
                elif now_facing == "S":
                    now_facing = "E"
                elif now_facing == "E":
                    now_facing = "N"
        return now_facing
    
    facing = "E"
    loc = [0,0]
    
    for op in operations:
        if op[0] == "N":
            loc[0] -= op[1]
        
        elif op[0] == "S":
            loc[0] += op[1]
            
        elif op[0] == "W":
            loc[1] -= op[1]
            
        elif op[0] == "E":
            loc[1] += op[1]
            
        elif op[0] == "L":
            facing = turn("L", op[1], facing)
        
        elif op[0] == "R":
            facing = turn("R", op[1], facing)
            
        elif op[0] == "F":
            if facing == "N":
                loc[0] -= op[1]
            elif facing == "S":
                loc[0] += op[1]
            elif facing == "W":
                loc[1] -= op[1]
            elif facing == "E":
                loc[1] += op[1]
                
    manhattan_distance_away1 = abs(loc[0]) + abs(loc[1])
    
    
    def rotate_waypoint(direction, degrees, current_waypoint):
        now_waypoint = current_waypoint.copy()
        for _ in range(degrees//90):
            if direction == "R":
                now_waypoint = [now_waypoint[1], -now_waypoint[0]]
            elif direction == "L":
                now_waypoint = [-now_waypoint[1], now_waypoint[0]]
        return now_waypoint
    
    ship = [0,0]
    waypoint = [-1,10] #relative to ship
    
    for op in operations:
        if op[0] == "N":
            waypoint[0] -= op[1]
        
        elif op[0] == "S":
            waypoint[0] += op[1]
            
        elif op[0] == "W":
            waypoint[1] -= op[1]
            
        elif op[0] == "E":
            waypoint[1] += op[1]
            
        elif op[0] == "L":
            waypoint = rotate_waypoint("L", op[1], waypoint)
        
        elif op[0] == "R":
            waypoint = rotate_waypoint("R", op[1], waypoint)
            
        elif op[0] == "F":
            ship[0] += op[1]*waypoint[0]
            ship[1] += op[1]*waypoint[1]
    
            
    manhattan_distance_away2 = abs(ship[0]) + abs(ship[1])
    
    
    return manhattan_distance_away1, manhattan_distance_away2

#%%
# Day 13: Shuttle Search

@time_this_func
def day13():
    with open("input13.txt") as f:
        raw = f.readlines()
    earliest_departure = int(raw[0])
    bus_IDs = list({int(x) for x in raw[1].split(",") if x != "x"})
    
    time = earliest_departure
    while True:
        time += 1
        leftover = {time%x for x in bus_IDs}
        if 0 in leftover:
            break
        
    wait = time - earliest_departure
    chosen_bus_ID = bus_IDs[[time%x for x in bus_IDs].index(0)]
    
    wait_x_ID = wait * chosen_bus_ID
    
    
    buses_orders = [(int(x), i) for i,x in enumerate(raw[1].split(",")) if x != "x"]
    
    #Chinese Remainder Theorem
    modulii = [x[0] for x in buses_orders] #Relatively prime. No common factors other than 1
    remainders  = np.array([(bus_ID-i)%bus_ID for bus_ID, i in buses_orders], dtype = np.int64)
    mod_prod = prod(modulii)
    N_is = np.array([mod_prod/n for n in modulii], dtype = np.int64)
    x_is = []
    for mod, N_i in zip(modulii,N_is): #Modular inverses. Could also find with [pow(int(N_i),-1,modulo) for N_i, modulo in zip(N_is, modulii)]
        multiplier = N_i%mod
        x = 0
        while True:
            x += 1
            if (multiplier*x)%mod == 1:
                break
        x_is.append(x)
    x_is = np.array(x_is, dtype = np.int64)
    
    first_time = np.sum(remainders*N_is*x_is)%mod_prod
    
    
    return wait_x_ID, first_time

#%%
# Day 14: Docking Data

@time_this_func
def day14():
    mask_set = ("placeholder", [])
    mask_sets = []
    with open("input14.txt") as f:
        for l in f:
            if "mask" in l:
                if len(mask_set[1]) > 0:
                    mask_sets.append(mask_set)
                mask_set = (l.strip()[7:], [])
            else:
                mask_set[1].append([int(x) for x in re.findall(r"[0-9]+",l)])
        mask_sets.append(mask_set)
        
    def bitify_36(num):
        start_bit = bin(num)[2:]
        if len(start_bit) > 36:
            raise Exception("Number to bitify is greater than 36-bit")
        
        bit_num = "".join([(36-len(start_bit))*"0",start_bit])
        return bit_num
    
    def debitify_36(bit_num):
        num = 0
        for i, b in enumerate(bit_num[::-1]):
            num += int(b)*(pow(2,i))
        return num
    
    def apply_mask1(mask, num):
        num_list = [x for x in bitify_36(num)]
        
        for i, b in enumerate(mask):
            if b != "X":
                num_list[i] = b
                
        bit_num = "".join(num_list)
        
        return debitify_36(bit_num)
    
    mem1 = {}
    for mask_set in mask_sets:
        mask = mask_set[0]
        for store_to, num in mask_set[1]:
            mem1[store_to] = apply_mask1(mask, num)
            
    sum_mem1 = sum(mem1.values())
    
    
    def apply_mask2(mask, num):
        num_list = [x for x in bitify_36(num)]
        X_inds = []
        for i, b in enumerate(mask):
            if b == "X":
                num_list[i] = "X"
                X_inds.append(i)
            elif b == "1":
                num_list[i] = "1"
        
        debitified_nums = []
        for X_vals in product(["0","1"], repeat = len(X_inds)):
            num_list_copy = num_list.copy()
            for X_val_ind, X_ind in enumerate(X_inds):
                num_list_copy[X_ind] = X_vals[X_val_ind]
            
            debitified_nums.append(debitify_36("".join(num_list_copy)))
        
        return debitified_nums
    
    mem2 = {}
    for mask_set in mask_sets:
        mask = mask_set[0]
        for address, store_value in mask_set[1]:
            mem_addresses = apply_mask2(mask, address)
            
            for mem_address in mem_addresses:
                mem2[mem_address] = store_value
    
    sum_mem2 = sum(mem2.values())
    
    
    return sum_mem1, sum_mem2

#%%
# Day 15: Rambunctious Recitation

@time_this_func
def day15():
    start_numbers = [0,13,1,16,6,17]
    
    when_said = dict(zip(start_numbers, [[x] for x in range(len(start_numbers))]))
    last_said = start_numbers[-1]
    for i in range(6,30_000_000):
        if last_said in when_said and len(when_said[last_said]) > 1:
            gap = i-1 - when_said[last_said][0]
            if gap in when_said:
                when_said[gap] = [when_said[gap][-1], i]
            else:
                when_said[gap] = [i]
                
            last_said = gap
            
        else:
            when_said[0] = [when_said[0][-1], i]
            last_said = 0
            
        if i == 2019:
            said_2020 = [key for key, value in when_said.items() if 2020-1 in value][0]
            
    said_30M = [key for key, value in when_said.items() if 30_000_000-1 in value][0]
    
    return said_2020, said_30M

#%%
# Day 16: Ticket Translation

@time_this_func
def day16():
    sections = (x for x in ["fields", "mine", "others"])
    section = next(sections)
    fields = {}
    others = []
    with open("input16.txt") as f:
        for l in f:
            if l == "\n":
                section = next(sections)
                continue
            
            if section == "fields":
                new = l.strip().split(": ")
                field = new[0]
                field_ranges_raw = new[1].split(" or ")
                field_ranges = []
                for field_range_raw in field_ranges_raw:
                    field_ranges.append([int(x) for x in field_range_raw.split("-")])
                
                fields[field] = field_ranges
                
            elif section == "mine":
                if "your ticket" in l:
                    continue
                mine = [int(x) for x in l.split(",")]
                
            elif section == "others":
                if "nearby tickets" in l:
                    continue
                others.append([int(x) for x in l.split(",")])
                
    def num_valid_some_field(num):
        for field in fields:
            for val_range in fields[field]:
                if num in range(val_range[0], val_range[1]+1):
                    return True
        return False
    
    error_rate = 0
    invalid_others_inds = set()
    for i, other in enumerate(others):
        for num in other:
            if not num_valid_some_field(num):
                error_rate += num
                invalid_others_inds.add(i)
                break
                
    
    valid_others = [other for i, other in enumerate(others) if i not in invalid_others_inds]
    
    ticket_nums_by_pos = [[nums[i] for nums in valid_others] for i in range(len(valid_others[0]))]
        
    field_could_be = {}
    for field in fields: 
        field_could_be[field] = set()
        
        for pos in range(len(fields)):
            
            could_be = True
            for num in ticket_nums_by_pos[pos]:
                if num not in range(fields[field][0][0],fields[field][0][1]+1) and \
                    num not in range(fields[field][1][0],fields[field][1][1]+1):
                        could_be = False
                        break
            
            if could_be:
                field_could_be[field].add(pos)
    
    identified = {}
    while len(identified) < 20:
        for field in field_could_be:
            if len(field_could_be[field]) == 1:
                identified_pos = field_could_be[field].pop()
                identified[field] = identified_pos
                for f in field_could_be:
                    field_could_be[f].discard(identified_pos)
    
    mine_identified = {field:mine[identified[field]] for field in fields}
        
    my_departure_info_prod = 1
    for field in mine_identified:
        if "departure" in field:
            my_departure_info_prod *= mine_identified[field]
            
    
    return error_rate, my_departure_info_prod

#%%
# Day 17: Conway Cubes

@time_this_func
def day17():
    original_active_cubes = set()
    with open("input17.txt") as f:
        for i, l in enumerate(f):
            for j, status in enumerate(l.strip()):
                if status == "#":
                    original_active_cubes.add((i,j,0))
    
    def check_neighbors_3d(loc):
        neighbors = []
        for i in [loc[0]-1, loc[0], loc[0]+1]:
            for j in [loc[1]-1, loc[1], loc[1]+1]:
                for k in [loc[2]-1, loc[2], loc[2]+1]:
                    if (i,j,k) == loc:
                        continue
                    
                    if (i,j,k) in active_cubes:
                        neighbors.append(1)
                    else:
                        neighbors.append(0)
        return neighbors
    
    active_cubes = original_active_cubes.copy()
    
    for _ in range(6):
        new_active_cubes = active_cubes.copy()
        all_coords = [[x[dim] for x in active_cubes] for dim in range(3)]
        min_i = min(all_coords[0])
        max_i = max(all_coords[0])
        min_j = min(all_coords[1])
        max_j = max(all_coords[1])
        min_k = min(all_coords[2])
        max_k = max(all_coords[2])
        
        for i in range(min_i-1, max_i+2):
            for j in range(min_j-1, max_j+2):
                for k in range(min_k-1, max_k+2):
                    neighbors = check_neighbors_3d((i,j,k))
                    
                    if (i,j,k) in active_cubes and sum(neighbors) not in {2,3}:
                        new_active_cubes.remove((i,j,k))
                        
                    elif (i,j,k) not in active_cubes and sum(neighbors) == 3:
                        new_active_cubes.add((i,j,k))
        
        active_cubes = new_active_cubes
        
    num_active_cubes_3d = len(active_cubes)
    
    
    def check_neighbors_4d(loc):
        neighbors = []
        for i in [loc[0]-1, loc[0], loc[0]+1]:
            for j in [loc[1]-1, loc[1], loc[1]+1]:
                for k in [loc[2]-1, loc[2], loc[2]+1]:
                    for l in [loc[3]-1, loc[3], loc[3]+1]:
                        if (i,j,k,l) == loc:
                            continue
                        
                        if (i,j,k,l) in active_cubes:
                            neighbors.append(1)
                        else:
                            neighbors.append(0)
        return neighbors
    
    active_cubes = {(*cube,0) for cube in original_active_cubes}
        
    for _ in range(6):
        new_active_cubes = active_cubes.copy()
        all_coords = [[x[dim] for x in active_cubes] for dim in range(4)]
        min_i = min(all_coords[0])
        max_i = max(all_coords[0])
        min_j = min(all_coords[1])
        max_j = max(all_coords[1])
        min_k = min(all_coords[2])
        max_k = max(all_coords[2])
        min_l = min(all_coords[3])
        max_l = max(all_coords[3])
        
        for i in range(min_i-1, max_i+2):
            for j in range(min_j-1, max_j+2):
                for k in range(min_k-1, max_k+2):
                    for l in range(min_l-1, max_l+2):
                        neighbors = check_neighbors_4d((i,j,k,l))
                        
                        if (i,j,k,l) in active_cubes and sum(neighbors) not in {2,3}:
                            new_active_cubes.remove((i,j,k,l))
                            
                        elif (i,j,k,l) not in active_cubes and sum(neighbors) == 3:
                            new_active_cubes.add((i,j,k,l))
        
        active_cubes = new_active_cubes
        
    num_active_cubes_4d = len(active_cubes)
    
    
    return num_active_cubes_3d, num_active_cubes_4d

#%%
# Day 18: Operation Order

@time_this_func
def day18():
    expressions = []
    with open("input18.txt") as f:
        for l in f:
            expressions.append(l.strip())
            
    def parenthesis_depth(expression):
        depth = 0
        paren_depth = []
        for c in expression:
            if c == "(":
                depth += 1
                paren_depth.append(depth)
            elif c == ")":
                paren_depth.append(depth)
                depth -= 1
            else:
                paren_depth.append(depth)
        return paren_depth
    
    def eval_new_math1(expression):
        if "(" in expression[1:] or ")" in expression[:-1]:
            raise Exception("eval_new_math cannot handle internal parentheses")
        
        expression = expression.replace("(","")
        expression = expression.replace(")","")
        
        parts = expression.split()
        leftmost = int(parts[0])
        next_op = None
        for part in parts[1:]:
            if not part.isnumeric():
                next_op = part
            else:
                if next_op == "+":
                    leftmost += int(part)
                elif next_op == "*":
                    leftmost *= int(part)
        
        return str(leftmost)
    
    def eval_expression_new_math(expression, part = "part1"):
        while " " in expression:
            paren_depth = parenthesis_depth(expression)
            max_paren_depth = max(paren_depth)
            deep_sections = []
            deep_section = ""
            for i, c in enumerate(paren_depth):
                if c != max_paren_depth and deep_section != "":
                    deep_sections.append(deep_section)
                    deep_section = ""
                
                elif c == max_paren_depth:
                    deep_section += expression[i]
            
            if deep_section != "":
                deep_sections.append(deep_section)
                
            new_expression = ""
            replacement_made = False
            for i, c in enumerate(paren_depth):
                if c != max_paren_depth:
                    new_expression += expression[i]
                    replacement_made = False
                    
                elif c == max_paren_depth and not replacement_made:
                    if part == "part1":
                        new_expression += eval_new_math1(deep_sections.pop(0))
                    else:
                        new_expression += eval_new_math2(deep_sections.pop(0))
                    replacement_made = True
            
            expression = new_expression
    
        return int(expression)
    
    sum_evaluated1 = sum([eval_expression_new_math(x) for x in expressions])
    
    
    def eval_new_math2(expression):
        if "(" in expression[1:] or ")" in expression[:-1]:
            raise Exception("eval_new_math cannot handle internal parentheses")
        
        expression = expression.replace("(","")
        expression = expression.replace(")","")
        
        parts = expression.split("*")
    
        result = prod([eval(part) for part in parts])
        
        return str(result)
    
    sum_evaluated2 = sum([eval_expression_new_math(x, "part2") for x in expressions])
    
    
    return sum_evaluated1, sum_evaluated2

#%%
# Day 19: Monster Messages

@time_this_func
def day19():
    rules = {}
    messages = []
    section = "rules"
    with open("input19.txt") as f:
        for l in f:
            if l == "\n":
                section = "messages"
                continue
            
            if section == "rules":
                new = l.strip().split(": ")
                rule_contents = new[1]
                if '"' in rule_contents:
                    rules[int(new[0])] = rule_contents.replace('"',"")
                else:
                    rule_contents = rule_contents.split(" | ")
                    rules[int(new[0])] = [[int(x) for x in rule_content.split()] for rule_content in rule_contents]
                    
            elif section == "messages":
                messages.append(l.strip())
                
    def get_solutions(rule_num):
        if rule_num in solved_rules:
            return solved_rules[rule_num]
        
        if type(rules[rule_num]) == str:
            solved_rules[rule_num] = [rules[rule_num]]
            return [rules[rule_num]]
                    
        solutions = []
        for solution_part in rules[rule_num]:
            sol = [get_solutions(x) for x in solution_part]
            
            if len(sol) == 1:
                solutions.append(sol[0])
                
            elif len(sol) == 2:
                sol_to_add = []
                for i in sol[0]:
                    for j in sol[1]:
                        sol_to_add.append(i+j)
                solutions.append(sol_to_add)
        
        solutions = list(set(sum(solutions, [])))
        solved_rules[rule_num] = solutions
        return solutions
    
    solved_rules = {}
    solutions_0 = set(get_solutions(0))
    
    num_messages_satisfy_0_part1 = sum([x in solutions_0 for x in messages])
    
    
    solutions_42 = solved_rules[42]
    solutions_31 = solved_rules[31]
    
    solution_42_len = len(solutions_42[0]) #All elements have same length
    solution_31_len = len(solutions_31[0]) #All elements have same length
    
    solutions_42 = set(solutions_42)
    solutions_31 = set(solutions_31)
    
    evaluated_messages = [(x,0,0) for x in messages]
    while True:
        next_evaluated_messages = []
        for message, num_42s, num_31s in evaluated_messages:
            if message[-solution_31_len:] in solutions_31:
                next_evaluated_messages.append((message[:-solution_31_len], num_42s, num_31s+1))
                
            else:
                next_evaluated_messages.append((message, num_42s, num_31s))
                
        if next_evaluated_messages == evaluated_messages:
            break
        
        evaluated_messages = next_evaluated_messages
        
    evaluated_messages = [x for x in evaluated_messages if len(x[0]) > 0]
    while True:
        next_evaluated_messages = []
        for message, num_42s, num_31s in evaluated_messages:
            if message[:solution_42_len] in solutions_42:
                next_evaluated_messages.append((message[solution_42_len:], num_42s+1, num_31s))
                
            else:
                next_evaluated_messages.append((message, num_42s, num_31s))
                
        if next_evaluated_messages == evaluated_messages:
            break
        
        evaluated_messages = next_evaluated_messages
        
    num_messages_satisfy_0_part2 = sum([len(x[0])==0 and x[1]>x[2] and x[2]>0 for x in evaluated_messages])
    
        
    return num_messages_satisfy_0_part1, num_messages_satisfy_0_part2

#%%
# Day 20: Jurassic Jigsaw

@time_this_func
def day20(visualize = False):
    with open("input20.txt") as f:
        raw = f.read()
    split_raw = raw.split("\n\n")[:-1]
    
    tiles = {}
    for raw_tile in split_raw:
        tile = raw_tile.split("\n")
        tile_id = int(tile[0].split()[1][:-1])
        tiles[tile_id] = np.array([[x for x in t] for t in tile[1:]])
        
    def edge_options(tile_mat):
        edge1 = "".join(tile_mat[:,0])
        edge2 = edge1[::-1]
        edge3 = "".join(tile_mat[:,-1])
        edge4 = edge3[::-1]
        edge5 = "".join(tile_mat[0,:])
        edge6 = edge5[::-1]
        edge7 = "".join(tile_mat[-1,:])
        edge8 = edge7[::-1]
        return {edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8}
    
    edges = {k:edge_options(v) for k,v in tiles.items()}
    
    neighbors = {}
    corners = []
    for i in tiles:
        neighbors[i] = []
        for j in tiles:
            
            if not edges[i].isdisjoint(edges[j]) and i != j:
                neighbors[i].append(j)
                
                if len(neighbors[i]) == 4:
                    continue
                
        if len(neighbors[i]) == 2:
            corners.append(i)
            
    corner_prod = prod(corners)
    

    start_corner = corners[0]
    
    active_tile = tiles[start_corner]
    while True:
        right_matched = False
        bottom_matched = False
        
        right_edge = "".join(active_tile[:,-1])
        bottom_edge = "".join(active_tile[-1,:])
        
        for j in neighbors[start_corner]:
            if right_edge in edges[j]:
                right_matched = True
            elif bottom_edge in edges[j]:
                bottom_matched = True
                
        if right_matched and bottom_matched:
            break
        active_tile = np.rot90(active_tile)
    
    oriented_ids = [[start_corner]]
    oriented = [[active_tile]]
    while len(sum(oriented_ids, [])) < len(tiles):
        if len(oriented[-1]) != 12:
            edge_to_match = "".join(oriented[-1][-1][:,-1])
            possible_active_ids = neighbors[oriented_ids[-1][-1]]
            adding_to = "right"
        else:
            edge_to_match = "".join(oriented[-1][0][-1,:])
            possible_active_ids = neighbors[oriented_ids[-1][0]]
            oriented.append([])
            oriented_ids.append([])
            adding_to = "bottom"
            
        for possible_active_id in possible_active_ids:
            if edge_to_match in edges[possible_active_id]:
                active_id = possible_active_id
                break
            
        active_tile = tiles[active_id]

        rots = [0, 1, 1, 1, 4, 1, 1, 1]
        
        for rot in rots:
            if rot != 4:
                active_tile = np.rot90(active_tile, rot)
            else:
                active_tile = np.rot90(active_tile)
                active_tile = np.fliplr(active_tile)
                
            if adding_to == "right":
                connector = "".join(active_tile[:,0])
            else:
                connector = "".join(active_tile[0,:])
                
            if connector == edge_to_match:
                oriented_ids[-1].append(active_id)
                oriented[-1].append(active_tile)
                break       
    
    pic = np.vstack([np.hstack(x) for x in  [[x[1:-1,1:-1] for x in y] for y in oriented]])
    pic = np.array(pic == "#", dtype = np.int32)
    
    seasnake = np.zeros([3,20])
    seasnake[0,18] = 1
    seasnake[1,0] = 1
    seasnake[1,5:7] = 1
    seasnake[1,11:13] = 1
    seasnake[1,-3:] = 1
    seasnake[2,1::3] = 1
    seasnake[2,-1] = 0
    
    edited_pic = pic.copy()
    num_seasnakes = 0
    for rot in rots:
        if rot != 4:
            pic = np.rot90(pic, rot)
            edited_pic = np.rot90(edited_pic, rot)
        else:
            pic = np.rot90(pic, rot)
            pic = np.fliplr(pic)
            edited_pic = np.rot90(edited_pic, rot)
            edited_pic = np.fliplr(edited_pic)
            
        for i in range(pic.shape[1]-seasnake.shape[1]+1):
            for j in range(pic.shape[0]-seasnake.shape[0]+1):
                if np.array_equal(pic[j:j+seasnake.shape[0], i:i+seasnake.shape[1]] * seasnake, seasnake):
                    num_seasnakes += 1
                    edited_pic[j:j+seasnake.shape[0], i:i+seasnake.shape[1]] = \
                        edited_pic[j:j+seasnake.shape[0], i:i+seasnake.shape[1]] + 10*seasnake
        
        if num_seasnakes > 0:
            break

    if visualize:
        plt.imshow(edited_pic, vmin = 0, vmax = 5)
        plt.xticks([])
        plt.yticks([])
    
    water_roughness = int(np.sum(edited_pic == 1))
    
    
    return corner_prod, water_roughness

#%%
# Day 21: Allergen Assessment

@time_this_func
def day21():
    foods = {}
    ingredients = set()
    allergens = set()
    with open("input21.txt") as f:
        for l in f:
            new = l.strip().split("(contains ")
            new_ingredients = new[0].split()
            new_allergens = set(new[1][:-1].split(", "))
            ingredients = ingredients | set(new_ingredients)
            allergens = allergens | new_allergens
            foods[tuple(new_ingredients)] = new_allergens
    
    allergen_maybe_in = {allergen:ingredients for allergen in allergens}
    food_sets = []
    for food in foods:
        food_sets.append(set(food))
        for allergen in foods[food]:
            allergen_maybe_in[allergen] = allergen_maybe_in[allergen] & food_sets[-1]
            
    ingredient_contains = {}
    while len(ingredient_contains) != len(allergens):
        singled = {v.pop():k for k,v in allergen_maybe_in.items() if len(v) == 1}
        ingredient_contains.update(singled)
        for v in singled:
            for k in allergen_maybe_in:
                allergen_maybe_in[k].discard(v)
    
    cleared_ingredients = ingredients - set(ingredient_contains)
    
    cleared_ingredient_count = 0
    for cleared_ingredient in cleared_ingredients:
        for food in food_sets:
            cleared_ingredient_count += cleared_ingredient in food
            
    cannonical_dangerous_list = ",".join(sorted(ingredient_contains, key = lambda x: ingredient_contains[x]))
    
    return cleared_ingredient_count, cannonical_dangerous_list

#%%
# Day 22: Crab Combat

@time_this_func
def day22():
    with open("input22.txt") as f:
        raw = f.read().strip()
        
    split_raw = raw.split("\n\n")
    decks = [[int(x) for x in split_raw[y].split("\n")[1:]] for y in range(len(split_raw))]
    
    while min([len(x) for x in decks]) != 0:
        drawn = [x.pop(0) for x in decks]
        winner = np.argmax(drawn)
        decks[winner] += sorted(drawn, reverse = True)
        
    winning_deck = decks[winner]
    score1 = sum([(i+1)*x for i, x in enumerate(winning_deck[::-1])])
    
    
    decks = [[int(x) for x in split_raw[y].split("\n")[1:]] for y in range(len(split_raw))]
    
    def play(temp_decks):
        rounds = [set() for x in range(len(temp_decks))]
        while min([len(x) for x in temp_decks]) != 0:
            if True in {tuple(temp_decks[x]) in rounds[x] for x in range(len(temp_decks))}:
                return 0, temp_decks[0]
            rounds = [rounds[x] | {tuple(temp_decks[x])} for x in range(len(temp_decks))]
            
            drawn = [x.pop(0) for x in temp_decks]
            if False in {len(temp_decks[x]) >= drawn[x] for x in range(len(temp_decks))}:
                winner = np.argmax(drawn)
                temp_decks[winner] += sorted(drawn, reverse = True)
            else:
                winner, _ = play([temp_decks[x][:drawn[x]] for x in range(len(temp_decks))])
                temp_decks[winner] += [drawn.pop(winner)] + drawn
        
        return winner, temp_decks[winner]
    
    _, winning_deck = play(decks)
    score2 = sum([(i+1)*x for i, x in enumerate(winning_deck[::-1])])
    
    
    return score1, score2

#%%
# Day 23: Crab Cups

@time_this_func
def day23():
    raw_cups = [int(x) for x in "643719258"]
    
    cups = {raw_cups[x]:raw_cups[(x+1)%len(raw_cups)] for x in range(len(raw_cups))}
    
    def play(cups, rounds, starter):
        max_cups = max(cups)
        
        current = starter
        for _ in range(rounds):
            clock_from_curr = cups[current]
            cups[current] = cups[cups[cups[clock_from_curr]]]
            
            removed = {clock_from_curr, cups[clock_from_curr], cups[cups[clock_from_curr]]}
            
            destination = current-1
            if destination == 0:
                destination = max_cups
            while destination in removed:
                destination -= 1
                if destination == 0:
                    destination = max_cups
                    
            cups[cups[cups[clock_from_curr]]] = cups[destination]
            cups[destination] = clock_from_curr
            current = cups[current]
            
        return cups
        
    final_cups = play(cups, 100, raw_cups[0])
    
    after_1 = ""
    x = 1
    while len(after_1) < len(raw_cups)-1:
        after_1 += str(final_cups[x])
        x = final_cups[x]
    after_1 = int(after_1)
    
    
    new_raw_cups = raw_cups + list(range(max(raw_cups)+1,1_000_001))
    cups = {new_raw_cups[x]:new_raw_cups[(x+1)%len(new_raw_cups)] for x in range(len(new_raw_cups))}
    
    final_cups = play(cups, 10_000_000, new_raw_cups[0])
    
    clock_2_mult = final_cups[1] * final_cups[final_cups[1]]
    
    
    return after_1, clock_2_mult
    
#%%
# Day 24: Lobby Layout

@time_this_func
def day24():
    instructions = []
    with open("input24.txt") as f:
        for l in f:
            new = l.strip()
            new_instructions = []
            new_instruction = ""
            for c in new:
                if c in {"e", "w"}:
                    new_instructions.append(new_instruction + c)
                    new_instruction = ""
                else:
                    new_instruction += c
            instructions.append(new_instructions)
           
    tiles = {} 
    for instruction in instructions:
        loc = (0,0)
        for inst in instruction:
            if inst == "nw":
                loc = (loc[0]-1, loc[1]-0.5)
            elif inst == "ne":
                loc = (loc[0]-1, loc[1]+0.5)
            elif inst == "e":
                loc = (loc[0], loc[1]+1)
            elif inst == "se":
                loc = (loc[0]+1, loc[1]+0.5)
            elif inst == "sw":
                loc = (loc[0]+1, loc[1]-0.5)
            elif inst == "w":
                loc = (loc[0], loc[1]-1)
        
        if loc not in tiles or tiles[loc] == "white":
            tiles[loc] = "black"
        else:
            tiles[loc] = "white"
            
    num_blacks = list(tiles.values()).count("black")
    
    
    def black_neighbors(tile_loc):
        num = 0
        for diff0, diff1 in [(-1,-0.5), (-1,0.5), (0,1), (1,0.5), (1,-0.5), (0,-1)]:
            num += (tile_loc[0]+diff0, tile_loc[1]+diff1) in tiles and tiles[(tile_loc[0]+diff0, tile_loc[1]+diff1)] == "black"
        return num
    
    def get_checklist(tiles):
        to_check = set()
        for tile in tiles:
            for diff0, diff1 in [(0,0), (-1,-0.5), (-1,0.5), (0,1), (1,0.5), (1,-0.5), (0,-1)]:
                if black_neighbors((tile[0]+diff0, tile[1]+diff1)) != 0:
                    to_check.add((tile[0]+diff0, tile[1]+diff1))
        return to_check
    
    for _ in range(100):
        new_tiles = {}
        for tile in get_checklist(tiles):
            num_black_neighbors = black_neighbors(tile)
            if tile not in tiles or tiles[tile] == "white":
                if num_black_neighbors == 2:
                    new_tiles[tile] = "black"
                else:
                    new_tiles[tile] = "white"
            else:
                if num_black_neighbors == 0 or num_black_neighbors > 2:
                    new_tiles[tile] = "white"
                else:
                    new_tiles[tile] = "black"
        tiles = new_tiles
    
    num_blacks_2 = list(tiles.values()).count("black")
    
    
    return num_blacks, num_blacks_2

#%%
# Day 25: Combo Breaker

@time_this_func
def day25():
    with open("input25.txt") as f:
        raw = f.read().strip().split()
        
    card_public_key = int(raw[0])
    door_public_key = int(raw[1])
    
    def get_loops(public_key, subject_number = 7):
        val = 1
        counter = 0
        while val != public_key:
            val = (val * subject_number)%20201227
            counter += 1
        return counter
    
    door_loops = get_loops(door_public_key)
    
    def apply_loops(loops, subject_number):
        val = 1
        for _ in range(loops):
            val = (val * subject_number)%20201227
        return val
    
    encryption_key = apply_loops(door_loops, card_public_key)
    
    return encryption_key