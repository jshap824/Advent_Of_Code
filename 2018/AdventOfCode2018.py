#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation   

#%%
# Timer Decorator
def time_this_func(func):
    from time import time
    def timed_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        print(f"{time()-t1:0.3f} s runtime")
        return result
    return timed_func

#%%
# Day 1: Chronal Calibration

@time_this_func
def day1():
    with open("input1.txt") as f:
        changes = tuple(int(x) for x in f.readlines())
            
    final_freq = sum(changes)
    
    freq = 0
    reached_freqs = {freq}
    first_double = None
    while first_double == None:
        for c in changes:
            freq += c
            if freq in reached_freqs:
                first_double = freq
                break
            else:
                reached_freqs.add(freq)           
                
    return final_freq, first_double

#%%
# Day 2: Inventory Management System

@time_this_func
def day2():
    with open("input2.txt") as f:
        boxes = tuple(f.readlines())
        
    doubles = 0
    triples = 0
    for box in boxes:
        counts = {box.count(x) for x in "abcdefghijklmnopqrstuvwxyz"}
        if 2 in counts:
            doubles += 1
        if 3 in counts:
            triples += 1
            
    checksum = doubles*triples
    
    
    boxes = [np.array([y for y in x.strip()]) for x in boxes]
    for box in boxes:
        for comp_box in boxes:
            if np.array_equal(box, comp_box):
                continue
            
            differs = 0
            for i in range(len(box)):
                if box[i] != comp_box[i]:
                    differs += 1
                    
                    if differs > 1:
                        break
            
            if differs == 1:
                break
                
        if differs == 1:
            break
        
    shared = "".join(box[box == comp_box])
    
    
    return checksum, shared

#%%
# Day 3: No Matter How You Slice It

@time_this_func
def day3():
    claims = []
    with open("input3.txt") as f:
        for l in f:
            new = l.split()[2:]
            offsets = tuple(int(x) for x in new[0][:-1].split(",")[::-1])
            dims = tuple(int(x) for x in new[1].split("x")[::-1])
            claims.append((offsets, dims))
    claims = tuple(claims)
    
    fabric = np.zeros([1000,1000])
    
    for claim in claims:
        fabric[claim[0][0]:claim[0][0]+claim[1][0], claim[0][1]:claim[0][1]+claim[1][1]] = fabric[claim[0][0]:claim[0][0]+claim[1][0], claim[0][1]:claim[0][1]+claim[1][1]] + 1
        
    overlapped = np.sum(fabric > 1)
    
    
    for i, claim in enumerate(claims):
        if np.max(fabric[claim[0][0]:claim[0][0]+claim[1][0], claim[0][1]:claim[0][1]+claim[1][1]]) == 1:
            break
    
    no_overlap_id = i+1
    
    
    return overlapped, no_overlap_id

#%%
# Day 4: Repose Record

@time_this_func
def day4():
    from datetime import datetime
    
    log = []
    with open("input4.txt") as f:
        for l in f:
            new = l.strip().split("] ")
            log.append((datetime.strptime(new[0][1:], "%Y-%m-%d %H:%M"), new[1]))
    
    log.sort(key = lambda x: x[0])
    
    log = tuple((x[0].minute, x[1]) for x in log)
    
    guards_asleep = {}
    for event in log:
        if "Guard" in event[1]:
            guard_number = int(event[1].split()[1][1:])
            if guard_number not in guards_asleep:
                guards_asleep[guard_number] = []
        
        elif "falls asleep" in event[1]:
            start_sleep = event[0]
        elif "wakes up" in event[1]:
            guards_asleep[guard_number] += list(range(start_sleep,event[0]))
            
    sleepiest_guard = sorted(guards_asleep.items(), key = lambda x: len(x[1]))[-1][0]
    sleepiest_minute =  sorted([(x,guards_asleep[sleepiest_guard].count(x)) for x in set(guards_asleep[sleepiest_guard])], key = lambda x: x[1])[-1][0]
    
    sleepiest_guard_minute = sleepiest_guard*sleepiest_minute
    
    
    most_consistently_asleep_guard = sorted([(k,max([v.count(x) for x in set(v)])) for k,v in guards_asleep.items() if len(v) != 0], key = lambda x: x[1])[-1][0]
    most_consistently_asleep_minute = sorted([(x,guards_asleep[most_consistently_asleep_guard].count(x)) for x in set(guards_asleep[most_consistently_asleep_guard])], key = lambda x: x[1])[-1][0]
    
    most_consistently_asleep_guard_minute = most_consistently_asleep_guard*most_consistently_asleep_minute
    
    
    return sleepiest_guard_minute, most_consistently_asleep_guard_minute
    
#%%
# Day 5: Alchemical Reduction

@time_this_func #73+ s. Come back to improve this one?
def day5():
    with open("input5.txt") as f:
        polymer = f.read().strip()
    
    def react(polymer):
        i = 0
        while True:
            curr = polymer[i]
            nxt = polymer[i+1]
            
            if curr.lower() == nxt.lower() and curr.islower() != nxt.islower():
                polymer.pop(i)
                polymer.pop(i)
                if i != 0:
                    i -= 1
            else:
                i += 1
            
            if i+1 == len(polymer):
                break
        
        return len([x for x in polymer])
    
    fully_reacted = react([x for x in polymer])
    
    
    best = np.inf
    for l in "abcdefghijklmnopqrstuvwxyz":
        clean_poly = polymer.replace(l,"").replace(l.upper(),"")
    
        final_len = react([x for x in clean_poly])
        if final_len < best:
            best = final_len
    
    
    return fully_reacted, best

#%%
# Day 6: Chronal Coordinates

@time_this_func
def day6():
    coords = {}
    with open("input6.txt") as f:
        for l in f:
            coords[tuple(int(x) for x in l.split(","))] = []
    
    row_min = min([x[0] for x in coords])
    row_max = max([x[0] for x in coords])
    col_min = min([x[1] for x in coords])
    col_max = max([x[1] for x in coords])
    
    for i in range(row_min,row_max+1):
        for j in range(col_min, col_max+1):
            point = (i,j)
            dists = {}
            for coord in coords:
                dist = abs(point[0]-coord[0]) + abs(point[1]-coord[1])
                
                if dist not in dists:
                    dists[dist] = [coord]
                else:
                    dists[dist].append(coord)
            
            closest = dists[sorted(dists)[0]]
            
            if len(closest) == 1:
                coords[closest[0]].append(point)
                
    biggest_finite = 0
    for coord in coords:
        all_closest_rows = {x[0] for x in coords[coord]}
        all_closest_cols = {x[1] for x in coords[coord]}
        if row_min not in all_closest_rows and row_max not in all_closest_rows:
            if col_min not in all_closest_cols and col_max not in all_closest_cols:
                if len(coords[coord]) > biggest_finite:
                    biggest_finite = len(coords[coord])
                
    
    sum_max = 10000-1
    safe_region_size = 0
    for i in range(row_min,row_max+1):
        for j in range(col_min, col_max+1):
            point = (i,j)
            dist_sum = 0
            for coord in coords:
                dist_sum += abs(point[0]-coord[0]) + abs(point[1]-coord[1])
                if dist_sum > sum_max:
                    break
            
            if dist_sum <= sum_max:
                safe_region_size += 1
                
                
    return biggest_finite, safe_region_size

#%%
# Day 7: The Sum of Its Parts

@time_this_func
def day7():
    dependency_pairs = []
    with open("input7.txt") as f:
        for l in f:
            new = l.split()
            dependency_pairs.append((new[1], new[-3]))
                
    dependencies = {}
    for pair in dependency_pairs:
        if pair[1] not in dependencies:
            dependencies[pair[1]] = {pair[0]}
        else:
            dependencies[pair[1]].add(pair[0])
            
    can_do = {x[0] for x in dependency_pairs} - {x[1] for x in dependency_pairs}
    to_do = ({x[0] for x in dependency_pairs} | {x[1] for x in dependency_pairs}) - can_do
    done = set()
    order = ""
    
    while len(can_do | to_do) > 0:
        order += sorted(can_do)[0]
        can_do.remove(order[-1])
        done.add(order[-1])
        
        for task in to_do:
            if dependencies[task] - done == set():
                can_do.add(task)
        to_do = to_do - can_do
        
    
    duration = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",range(60+1,60+27)))
    progress = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",[0]*26))
    workers = dict(zip(range(1,6), [None]*5))
    
    can_do = {x[0] for x in dependency_pairs} - {x[1] for x in dependency_pairs}
    to_do = ({x[0] for x in dependency_pairs} | {x[1] for x in dependency_pairs}) - can_do
    working_on = set()
    done = set()
    
    assembly_time = 0
    while len(can_do | to_do | working_on) > 0:
        assembly_time += 1
        
        for w in workers:
            if workers[w] != None:
                continue
            
            if len(can_do) > 0:
                workers[w] = can_do.pop()
                working_on.add(workers[w])
        
        new_done = False
        for w in workers:
            if workers[w] == None:
                continue
            
            progress[workers[w]] += 1
            if progress[workers[w]] == duration[workers[w]]:
                new_done = True
                done.add(workers[w])
                working_on.remove(workers[w])
                workers[w] = None
                
        if new_done:
            for task in to_do:
                if dependencies[task] - done == set():
                    can_do.add(task)
            to_do = to_do - can_do
    
    
    return order, assembly_time

#%%
# Day 8: Memory Maneuver

@time_this_func
def day8():
    with open("input8.txt") as f:
        info = [int(x) for x in f.read().split()]
        
    part1 = info.copy()
    metadata_sum = 0
    while True:
        i = 0
        while True:
            if part1[i] == 0:
                part1.pop(i)
                num_metas = part1.pop(i)
                for _ in range(num_metas):
                    metadata_sum += part1.pop(i)
                break
                
            else:
                i += 2
            
        if len(part1) != 0:
            part1[i-2] -= 1
        else:
            break
        
    
    part2 = info.copy()
    while True:
        i = 0
        while True:
            if part2[i] == 0 and type(part2[i]) == int:
                part2.pop(i)
                num_metas = part2.pop(i)
                val = 0
                for _ in range(num_metas):
                    val -= part2.pop(i)
                part2.insert(i, float(val))
                part2.insert(i+1, float(val))
                break
            
            elif part2[i] > 0 and False not in {part2[x] < 0 for x in range(i+2,i+2+(2*part2[i]),2)}:
                num_children = part2.pop(i)
                num_metas = part2.pop(i)
                vals = []
                for _ in range(num_children):
                    vals.append(part2.pop(i))
                    part2.pop(i)
                    
                inds = []
                for _ in range(num_metas):
                    possible_ind = part2.pop(i)
                    if possible_ind in range(1,num_children+1):
                        inds.append(possible_ind)
                        
                val = 0
                for val_ind in inds:
                    val += vals[val_ind-1]
                part2.insert(i, float(val))
                part2.insert(i+1, float(val))
                break
            
            else:
                i += 2
                
        if len(part2) == 2:
            break
        
    root_val = -int(part2[0])
    
    
    return metadata_sum, root_val

#%%
# Day 9: Marble Mania

@time_this_func
def day9():
    with open("input9.txt") as f:
        new = f.read().split()
        num_players = int(new[0])
        max_marble_num = int(new[-2])
        
    def player_gen(number_players):
        players = range(1,number_players+1)
        i = 0
        while True:
            yield players[i%number_players]
            i += 1
            
    class Marble():
        def __init__(self, num):
            self.num = num
            self.counter = self
            self.clock = self
            
        def insert_after(self, after_this):
            before_this = after_this.clock
            after_this.clock = self
            before_this.counter = self
            self.counter = after_this
            self.clock = before_this
            return self
    
    def play(max_marble_num, num_players = num_players):
        player = player_gen(num_players)
        marble_nums = tuple(x for x in range(1,max_marble_num+1))
        
        player_scores = dict(zip(range(1,num_players+1),[0]*num_players))
        curr_marble = Marble(0)
        
        for next_marble_num in marble_nums:
            curr_player = next(player)
        
            if next_marble_num%23 != 0:
                new_marble = Marble(next_marble_num)
                curr_marble = new_marble.insert_after(curr_marble.clock)
                
            else:
                player_scores[curr_player] += next_marble_num
                for _ in range(7):
                    curr_marble = curr_marble.counter
                    
                player_scores[curr_player] += curr_marble.num
                curr_marble.counter.clock = curr_marble.clock
                curr_marble.clock.counter = curr_marble.counter
                curr_marble = curr_marble.clock
    
        return max(player_scores.values())
    
    return play(max_marble_num), play(100*max_marble_num)

#%%
# Day 10: The Stars Align

@time_this_func
def day10():
    import re
    
    class Light():
        def __init__(self, pos, vel):
            self.pos = pos
            self.vel = vel
            
        def move(self):
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]
            
        def copy(self):
            return Light(self.pos.copy(), self.vel)
            
    lights = []
    with open("input10.txt") as f:
        for l in f:
            vectors = re.findall("<.*?>", l)
            pos = [int(x) for x in vectors[0][1:-1].split(",")[::-1]]
            vel = tuple(int(x) for x in vectors[1][1:-1].split(",")[::-1])
            lights.append(Light(pos, vel))
            
    def space_taken(lights):
        poses = [x.pos for x in lights]
        min_row = min([x[0] for x in poses])
        max_row = max([x[0] for x in poses])
        min_col = min([x[1] for x in poses])
        max_col = max([x[1] for x in poses]) 
        return (max_row - min_row)*(max_col - min_col)
            
    def visualize(lights):
        poses = [x.pos for x in lights]
        min_row = min([x[0] for x in poses])
        max_row = max([x[0] for x in poses])
        min_col = min([x[1] for x in poses])
        max_col = max([x[1] for x in poses])
        
        sky = np.zeros([max_row - min_row + 1, max_col - min_col + 1], dtype = np.bool_)
        for pos in poses:
            sky[pos[0]-min_row, pos[1]-min_col] = 1
        plt.imshow(sky)
    
    wait = 0
    while True:
        last_lights = [x.copy() for x in lights]
        for light in lights:
            light.move()
        if space_taken(lights) > space_taken(last_lights):
            break
        wait += 1
        
    visualize(last_lights); return wait

#%%
# Day 11: Chronal Charge

@time_this_func
def day11():
    serial = 5093
    
    cells = np.zeros([300,300])
    for x_ind in range(300):
        for y_ind in range(300):
            x = x_ind + 1
            y = y_ind + 1
            rack_id = x + 10
            power_level = (rack_id * y) + serial
            power_level *= rack_id
            power_level_string = str(power_level)
            if len(power_level_string) < 3:
                power_level = 0
            else:
                power_level = int(power_level_string[-3])
            power_level -= 5
            
            cells[y_ind,x_ind] = power_level
            
    max_power = 0
    for x_ind in range(cells.shape[1]-2):
        for y_ind in range(cells.shape[0]-2):
            curr_power = np.sum(cells[y_ind:y_ind+3, x_ind:x_ind+3])
            if curr_power > max_power:
                max_power = curr_power
                max_power_3_loc = (x_ind+1, y_ind+1)
    
    
    max_power = 0
    for s in range(1,301):
        for x_ind in range(cells.shape[1]-(s-1)):
            for y_ind in range(cells.shape[0]-(s-1)):
                curr_power = np.sum(cells[y_ind:y_ind+s, x_ind:x_ind+s])
                if curr_power > max_power:
                    max_power = curr_power
                    max_power_loc_and_size = (x_ind+1, y_ind+1, s)
                    
                    
    return max_power_3_loc, max_power_loc_and_size

#%%
# Day 12: Subterranean Sustainability

@time_this_func
def day12():    
    rules = {}
    with open("input12.txt") as f:
        for l in f:
            if "initial state" in l:
                state = [1 if x == "#" else 0 for x in l.strip().split()[2]]
            elif "=>" in l:
                rule_parts = l.strip().split(" => ")
                if rule_parts[1] == ".":
                    rules[tuple(1 if x == "#" else 0 for x in rule_parts[0])] = 0
                else:
                    rules[tuple(1 if x == "#" else 0 for x in rule_parts[0])] = 1
                    
    pot_nums = list(range(len(state)))
    
    generations_states = []
    generations_pot_nums = []
    for g in range(50000000000):
        add_pots_left_num = -(state.index(1) - 4)
        if add_pots_left_num < 0:
            pot_nums = pot_nums[-add_pots_left_num:]
            state = state[-add_pots_left_num:]
        else:
            for _ in range(add_pots_left_num):
                pot_nums = [pot_nums[0]-1] + pot_nums
                state = [0] + state
        
        add_pots_right_num = -state[::-1].index(1) + 4
        if add_pots_right_num < 0:
            pot_nums = pot_nums[:add_pots_right_num]
            state = state[:add_pots_right_num]
        else:
            for _ in range(add_pots_right_num):
                pot_nums = pot_nums + [pot_nums[-1]+1]
                state = state + [0]
        
        if state in generations_states:
            break
        else:
            generations_states.append(state)
            generations_pot_nums.append(pot_nums)
        
        next_state = state.copy()
        for rule in rules:    
            for i in range(2,len(state)-2):
                if state[i-2:i+3] == list(rule):
                    next_state[i] = rules[rule]
        state = next_state
        
        if g == 20-1:
            plant_pots_20_num_sum = sum([pot_nums[i] for i in range(len(pot_nums)) if state[i] == 1])
            
    first_appearance = generations_states.index(state)
    first_apperance_pot_nums = generations_pot_nums[first_appearance]
    
    pot_nums_shift_per_gen = (pot_nums[0] - first_apperance_pot_nums[0])//(g-first_appearance)
    num_plants = sum(state)
    curr_pot_sum = sum([pot_nums[i] for i in range(len(pot_nums)) if state[i] == 1])
    generations_to_go = 50000000000 - g
    plant_pots_50B_num_sum = curr_pot_sum + (num_plants*pot_nums_shift_per_gen)*generations_to_go
    
    return plant_pots_20_num_sum, plant_pots_50B_num_sum

#%%
# Day 13: Mine Cart Madness

@time_this_func
def day13():
    class Cart():
        def __init__(self, loc, facing):
            self.loc = loc
            self.facing = facing
            self.next_turn = "left"
        
        
        def turn(self):
            if self.next_turn == "left":
                if self.facing == "N":
                    self.facing = "W"
                elif self.facing == "W":
                    self.facing = "S"
                elif self.facing == "S":
                    self.facing = "E"
                elif self.facing == "E":
                    self.facing = "N"
                self.next_turn = "straight"
            elif self.next_turn == "straight":
                self.next_turn = "right"
            elif self.next_turn == "right":
                if self.facing == "N":
                    self.facing = "E"
                elif self.facing == "E":
                    self.facing = "S"
                elif self.facing == "S":
                    self.facing = "W"
                elif self.facing == "W":
                    self.facing = "N"
                self.next_turn = "left"
       
    tracks = []
    carts = []
    with open("input13.txt") as f:
        row_num = -1
        for l in f:
            row_num += 1
            row = [x for x in l[:-1]]
            for col_num in range(len(row)):
                if row[col_num] == ">":
                    carts.append(Cart([row_num, col_num], "E"))
                    row[col_num] = "-"
                elif row[col_num] == "<":
                    carts.append(Cart([row_num, col_num], "W"))
                    row[col_num] = "-"
                elif row[col_num] == "^":
                    carts.append(Cart([row_num, col_num], "N"))
                    row[col_num] = "|"
                elif row[col_num] == "v":
                    carts.append(Cart([row_num, col_num], "S"))
                    row[col_num] = "|"
            tracks.append(row)
    tracks = np.array(tracks)
    
    first_crashed = False
    while True:
        crashed_cart_nums = []
        carts = sorted(sorted(carts, key = lambda x: x.loc[1]), key = lambda x: x.loc[0])
        for cart_num, cart in enumerate(carts):
            if cart_num in crashed_cart_nums:
                continue
            
            if cart.facing == "W":
                next_loc = (cart.loc[0], cart.loc[1]-1)
                if tracks[next_loc] == "\\":
                    cart.facing = "N"
                elif tracks[next_loc] == "/":
                    cart.facing = "S"
                elif tracks[next_loc] == "+":
                    cart.turn()
                cart.loc[1] -= 1
            
            elif cart.facing == "E":
                next_loc = (cart.loc[0], cart.loc[1]+1)
                if tracks[next_loc] == "\\":
                    cart.facing = "S"
                elif tracks[next_loc] == "/":
                    cart.facing = "N"
                elif tracks[next_loc] == "+":
                    cart.turn()
                cart.loc[1] += 1     
                
            elif cart.facing == "N":
                next_loc = (cart.loc[0]-1, cart.loc[1])
                if tracks[next_loc] == "\\":
                    cart.facing = "W"
                elif tracks[next_loc] == "/":
                    cart.facing = "E"
                elif tracks[next_loc] == "+":
                    cart.turn()
                cart.loc[0] -= 1 
                
            elif cart.facing == "S":
                next_loc = (cart.loc[0]+1, cart.loc[1])
                if tracks[next_loc] == "\\":
                    cart.facing = "E"
                elif tracks[next_loc] == "/":
                    cart.facing = "W"
                elif tracks[next_loc] == "+":
                    cart.turn()
                cart.loc[0] += 1 
                
            all_cart_locs = [x.loc for x in carts]
            if all_cart_locs.count(cart.loc) > 1:
                if not first_crashed:
                    first_crash_loc = cart.loc[::-1]
                    first_crashed = True
                
                crashed_cart_nums += [i for i in range(len(carts)) if carts[i].loc == cart.loc]
                for crashed_i in crashed_cart_nums:
                    carts[crashed_i].loc = [-1,-1]
        
        active_carts = []
        for i in range(len(carts)):
            if i not in crashed_cart_nums:
                active_carts.append(carts[i])
        carts = active_carts
        
        if len(carts) == 1:
            break
        
    survivor_loc = carts[0].loc[::-1]
    
    return first_crash_loc, survivor_loc

#%%
# Day 14: Chocolate Charts

@time_this_func
def day14():
    score_after = 440231
    score_pattern = str(score_after)
    
    recipes = [3,7]
    elf1_loc = 0
    elf2_loc = 1
    
    score_10_not_found = True
    pattern_not_found = True
    while score_10_not_found or pattern_not_found:
        recipes += [int(x) for x in str(recipes[elf1_loc] + recipes[elf2_loc])]
        elf1_loc = (elf1_loc + recipes[elf1_loc] + 1)%len(recipes)
        elf2_loc = (elf2_loc + recipes[elf2_loc] + 1)%len(recipes)
        
        if len(recipes) >= score_after + 10 and score_10_not_found:
            score_10 = "".join([str(x) for x in recipes[score_after:score_after+10]])
            score_10_not_found = False
            
        if len(recipes) >= len(score_pattern) and score_pattern in "".join([str(x) for x in recipes[-len(score_pattern)-1:]]):
            scores_before_pattern = "".join([str(x) for x in recipes]).index(score_pattern)
            pattern_not_found = False
    
    return score_10, scores_before_pattern

#%%
# Day 15: Beverage Bandits

@time_this_func
def day15(verbose = True, visualize = False, run_full = False):
    if verbose:
        print("Base Case")
        
    class Elf():
        def __init__(self, loc):
            self.hp = 200
            self.attack = 3
            self.loc = loc
            
    class Goblin():
        def __init__(self, loc):
            self.hp = 200
            self.attack = 3
            self.loc = loc
            
    walls = set()
    units = []
    battleground = []
    with open("input15.txt") as f:
        row_num = -1
        for l in f:
            row = []
            row_num += 1
            for col_num, s in enumerate(l[:-1]):
                if s == "#":
                    walls.add((row_num, col_num))
                    row.append(1)
                else:
                    if s == "E":
                        units.append(Elf((row_num, col_num)))
                    elif s == "G":
                        units.append(Goblin((row_num, col_num)))
                    row.append(0)
            battleground.append(row)
    battleground = np.array(battleground)
    
    def get_view():
        elves = [x for x in units if type(x) == Elf and x.hp > 0]
        goblins = [x for x in units if type(x) == Goblin and x.hp > 0]
        
        elf_ground = np.zeros(battleground.shape)
        for e in elves:
            elf_ground[e.loc] = 2
            
        goblin_ground = np.zeros(battleground.shape)
        for g in goblins:
            goblin_ground[g.loc] = 3
            
        return battleground + elf_ground + goblin_ground
            
    
    def is_free(loc):
        if loc in walls | {x.loc for x in units if x.hp > 0}:
            return False
        else:
            return True
            
    def dist(location1, location2):
        return abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])
                    
    def shortest_path_len(from_loc, to_loc, prune_num = 10, current_shortest_len = np.inf, max_prune = 10):
        if dist(from_loc, to_loc) > current_shortest_len:
            return np.inf
            
        paths = [[from_loc]]
        while True:
            new_paths = []
            for path in paths:
                latest = path[-1]
                visited = set(sum(paths, []) + sum(new_paths, []))
                if is_free((latest[0]-1, latest[1])) and (latest[0]-1, latest[1]) not in visited:
                    new_paths.append(path + [(latest[0]-1, latest[1])])
                if is_free((latest[0]+1, latest[1])) and (latest[0]+1, latest[1]) not in visited:
                    new_paths.append(path + [(latest[0]+1, latest[1])])            
                if is_free((latest[0], latest[1]-1)) and (latest[0], latest[1]-1) not in visited:
                    new_paths.append(path + [(latest[0], latest[1]-1)])                
                if is_free((latest[0], latest[1]+1)) and (latest[0], latest[1]+1) not in visited:
                    new_paths.append(path + [(latest[0], latest[1]+1)])     
            
            sorted_paths = sorted(new_paths, key = lambda x: dist(x[-1], to_loc))
            paths = sorted_paths[:prune_num]
            if len(paths) == 0:
                if prune_num >= max_prune:
                    return np.inf
                else:
                    return shortest_path_len(from_loc, to_loc, prune_num+10, current_shortest_len)
            elif len(paths[0]) > current_shortest_len:
                return np.inf
            elif to_loc in {x[-1] for x in paths}:
                return len(paths[0])
            
    def shortest_path_start_loc(from_loc, to_loc, shortest, max_prune = 10):
        start_locs = []
        if is_free((from_loc[0]-1, from_loc[1])):
            start_locs.append((from_loc[0]-1, from_loc[1]))
        if is_free((from_loc[0]+1, from_loc[1])):
            start_locs.append((from_loc[0]+1, from_loc[1]))
        if is_free((from_loc[0], from_loc[1]-1)):
            start_locs.append((from_loc[0], from_loc[1]-1))
        if is_free((from_loc[0], from_loc[1]+1)):
            start_locs.append((from_loc[0], from_loc[1]+1))
            
        for start_option in start_locs:
            if start_option == to_loc:
                return start_option
            
        dists = []
        for start_option in start_locs:
            next_len = shortest_path_len(start_option, to_loc, current_shortest_len = shortest, max_prune = max_prune)
            if next_len < shortest:
                shortest = next_len
            dists.append(next_len)
            
        best_starts = [start_locs[i] for i in range(len(dists)) if dists[i] == min(dists)]
        if len(best_starts) == 0:
            return best_starts[0]
        else:
            return sorted(sorted(best_starts, key = lambda x: x[1]), key = lambda x: x[0])[0]
    
    history = []
    history.append(get_view())
    
    battle_over = False
    rounds = 0
    while True:
        units = sorted(sorted(units, key = lambda x: x.loc[1]), key = lambda x: x.loc[0])
        for i in range(len(units)):
            unit = units[i]
            
            if unit.hp <= 0:
                continue
            
            if type(unit) == Elf:
                enemies = [units[i] for i in range(len(units)) if type(units[i]) == Goblin and units[i].hp > 0]
            elif type(unit) == Goblin:
                enemies = [units[i] for i in range(len(units)) if type(units[i]) == Elf and units[i].hp > 0]
                
            if len(enemies) == 0:
                battle_over = True
                break
            
            enemy_adj_locs = []
            move = True
            for e in enemies:
                enemy_adj_locs += [(e.loc[0]-1, e.loc[1]), (e.loc[0]+1, e.loc[1]), (e.loc[0], e.loc[1]-1), (e.loc[0], e.loc[1]+1)]
                
                if unit.loc in enemy_adj_locs:
                    move = False
                    break
            
            if move:
                dests = list((set(enemy_adj_locs) - {x.loc for x in units if x.hp > 0}) - walls)
                path_lens = []
                shortest = np.inf
                for d in dests:
                    next_len = shortest_path_len(unit.loc, d, current_shortest_len = shortest)
                    if next_len < shortest:
                        shortest = next_len
                    path_lens.append(next_len)
                
                closest_dests = [dests[i] for i in range(len(dests)) if path_lens[i] == min(path_lens) and min(path_lens) != np.inf]
                
                if len(closest_dests) > 0:
                    if len(closest_dests) == 1:
                        dest = closest_dests[0]
                    else:
                        dest = sorted(sorted(closest_dests, key = lambda x: x[1]), key = lambda x: x[0])[0]
                    
                    move_to = shortest_path_start_loc(unit.loc, dest, shortest = min(path_lens)-1)
                    unit.loc = move_to
                    
                    history.append(get_view())
                    
            if unit.loc in enemy_adj_locs:
                attackables = [enemies[i] for i in range(len(enemies)) if dist(unit.loc, enemies[i].loc) == 1]
                if len(attackables) == 1:
                    attacked = attackables[0]
                else:
                    attacked = sorted(sorted(sorted(attackables, key = lambda x: x.loc[1]), key = lambda x: x.loc[0]), key = lambda x: x.hp)[0]
                
                attacked.hp -= unit.attack
             
        if battle_over:
                break
        
        rounds += 1
        total_elf_hp = sum([x.hp for x in units if type(x) == Elf and x.hp > 0])
        total_goblin_hp = sum([x.hp for x in units if type(x) == Goblin and x.hp > 0])
        if verbose:
            print(f"Round {rounds} done. Elf HP: {total_elf_hp}, Goblin HP: {total_goblin_hp}")
        
    history.append(get_view())
    
    if visualize:
        plt.figure()
        for history_entry in history:
            plt.cla()
            plt.imshow(history_entry)
            plt.pause(.001)
    
    remaining_hp = sum([x.hp for x in units if x.hp > 0])
    outcome_3 = rounds*remaining_hp
    
    
    if run_full:
        elf_attack = 3
    else:
        elf_attack = 11
    while True:
        elf_attack += 1
        if verbose:
            print(f"\nElf attack level: {elf_attack}")
        class Elf():
            def __init__(self, loc):
                self.hp = 200
                self.attack = elf_attack
                self.loc = loc
                
        class Goblin():
            def __init__(self, loc):
                self.hp = 200
                self.attack = 3
                self.loc = loc
                
        units = []
        with open("input15.txt") as f:
            row_num = -1
            for l in f:
                row_num += 1
                for col_num, s in enumerate(l[:-1]):
                    if s == "E":
                        units.append(Elf((row_num, col_num)))
                    elif s == "G":
                        units.append(Goblin((row_num, col_num)))
        
        history = []
        history.append(get_view())
        
        battle_over = False
        rounds = 0
        while True:
            units = sorted(sorted(units, key = lambda x: x.loc[1]), key = lambda x: x.loc[0])
            for i in range(len(units)):
                unit = units[i]
                
                if unit.hp <= 0:
                    continue
                
                if type(unit) == Elf:
                    enemies = [units[i] for i in range(len(units)) if type(units[i]) == Goblin and units[i].hp > 0]
                elif type(unit) == Goblin:
                    enemies = [units[i] for i in range(len(units)) if type(units[i]) == Elf and units[i].hp > 0]
                    
                if len(enemies) == 0:
                    battle_over = True
                    break
                
                enemy_adj_locs = []
                move = True
                for e in enemies:
                    enemy_adj_locs += [(e.loc[0]-1, e.loc[1]), (e.loc[0]+1, e.loc[1]), (e.loc[0], e.loc[1]-1), (e.loc[0], e.loc[1]+1)]
                    
                    if unit.loc in enemy_adj_locs:
                        move = False
                        break
                
                if move:
                    dests = list((set(enemy_adj_locs) - {x.loc for x in units if x.hp > 0}) - walls)
                    path_lens = []
                    shortest = np.inf
                    for d in dests:
                        next_len = shortest_path_len(unit.loc, d, current_shortest_len = shortest)
                        if next_len < shortest:
                            shortest = next_len
                        path_lens.append(next_len)
                    
                    closest_dests = [dests[i] for i in range(len(dests)) if path_lens[i] == min(path_lens) and min(path_lens) != np.inf]
                    
                    if len(closest_dests) > 0:
                        if len(closest_dests) == 1:
                            dest = closest_dests[0]
                        else:
                            dest = sorted(sorted(closest_dests, key = lambda x: x[1]), key = lambda x: x[0])[0]
                        
                        move_to = shortest_path_start_loc(unit.loc, dest, shortest = min(path_lens)-1)
                        unit.loc = move_to
                        
                        history.append(get_view())
                        
                if unit.loc in enemy_adj_locs:
                    attackables = [enemies[i] for i in range(len(enemies)) if dist(unit.loc, enemies[i].loc) == 1]
                    if len(attackables) == 1:
                        attacked = attackables[0]
                    else:
                        attacked = sorted(sorted(sorted(attackables, key = lambda x: x.loc[1]), key = lambda x: x.loc[0]), key = lambda x: x.hp)[0]
                    
                    attacked.hp -= unit.attack
                    
                    if type(attacked) == Elf and attacked.hp <= 0:
                        battle_over = True
                        break
                
            if battle_over:
                    break
            
            rounds += 1
            total_elf_hp = sum([x.hp for x in units if type(x) == Elf and x.hp > 0])
            total_goblin_hp = sum([x.hp for x in units if type(x) == Goblin and x.hp > 0])
            if verbose:
                print(f"Round {rounds} done. Elf HP: {total_elf_hp}, Goblin HP: {total_goblin_hp}")
            
        history.append(get_view())
        
        if visualize:
            plt.figure()
            for history_entry in history:
                plt.cla()
                plt.imshow(history_entry)
                plt.pause(.001)
        
        remaining_hp = sum([x.hp for x in units if x.hp > 0])
        outcome_elf = rounds*remaining_hp
        
        if min(x.hp for x in units if type(x) == Elf) > 0:
            break
    
    
    return outcome_3, outcome_elf

#%%
# Day 16: Chronal Classification

@time_this_func
def day16():
    samples = []
    test_program = []
    with open("input16.txt") as f:
        counter = 0
        to_add = []
        first_section = True
        skip_count = 0
        for l in f:
            if first_section:
                if counter < 3 and len(l.strip()) > 0:
                    skip_count = 0
                    if counter in [0,2]:
                        new = l.split()[1:]
                        new = tuple(int(x) for x in "".join(new)[1:-1].split(","))
                    else:
                        new = tuple(int(x) for x in l.split())
                    to_add.append(new)
                    counter += 1
                elif len(to_add) != 0:
                    samples.append(to_add)
                    counter = 0
                    to_add = []
            
            if len(l.strip()) == 0:
                skip_count += 1
                if skip_count > 1:
                    first_section = False
                    
            if not first_section:
                if len(l.strip()) != 0:
                    test_program.append(tuple(int(x) for x in l.strip().split()))
    
    addr = set()
    addi = set()
    mulr = set()
    muli = set()
    banr = set()
    bani = set()
    borr = set()
    bori = set()
    setr = set()
    seti = set()
    gtir = set()
    gtri = set()
    gtrr = set()
    eqir = set()
    eqri = set()
    eqrr = set()
    
    num_behave = 0
    for sample in samples:
        before  = sample[0]
        op_code = sample[1][0]
        a = sample[1][1]
        b = sample[1][2]
        c = sample[1][3]
        after = sample[2]
        
        matches = 0
        if before[a] + before[b] == after[c]:
            matches += 1
            addr.add(op_code)
        if before[a] + b == after[c]:
            matches += 1
            addi.add(op_code)
        if before[a] * before[b] == after[c]:
            matches += 1
            mulr.add(op_code)
        if before[a] * b == after[c]:
            matches += 1
            muli.add(op_code)
        if before[a] & before[b] == after[c]:
            matches += 1
            banr.add(op_code)
        if before[a] & b == after[c]:
            matches += 1
            bani.add(op_code)
        if before[a] | before[b] == after[c]:
            matches += 1
            borr.add(op_code)
        if before[a] | b == after[c]:
            matches += 1
            bori.add(op_code)
        if before[a] == after[c]:
            matches += 1
            setr.add(op_code)
        if a == after[c]:
            matches += 1
            seti.add(op_code)
        if (a > before[b]) == after[c]:
            matches += 1
            gtir.add(op_code)
        if (before[a] > b) == after[c]:
            matches += 1
            gtri.add(op_code)
        if (before[a] > before[b]) == after[c]:
            matches += 1
            gtrr.add(op_code)
        if (a == before[b]) == after[c]:
            matches += 1
            eqir.add(op_code)
        if (before[a] == b) == after[c]:
            matches += 1
            eqri.add(op_code)
        if (before[a] == before[b]) == after[c]:
            matches += 1
            eqrr.add(op_code)
    
        if matches >= 3:
            num_behave += 1    
    
    
    ops = [addr,addi,mulr,muli,banr,bani,borr,bori,setr,seti,gtir,gtri,gtrr,eqir,eqri,eqrr]
    while set in {type(x) for x in ops}:
        for i in range(len(ops)):
            op = ops[i]
            if type(op) == set and len(op) == 1:
                ops[i] = op.pop()
                for j in range(len(ops)):
                    if type(ops[j]) == set:
                        ops[j].discard(ops[i])
               
    registers = {0:0, 1:0, 2:0, 3:0}
    for operation in test_program:
        op_code = operation[0]
        a = operation[1]
        b = operation[2]
        c = operation[3]
    
        if op_code == ops[0]:
            registers[c] = registers[a] + registers[b]
        elif op_code == ops[1]:
            registers[c] = registers[a] + b
        elif op_code == ops[2]:
            registers[c] = registers[a] * registers[b]
        elif op_code == ops[3]:
            registers[c] = registers[a] * b
        elif op_code == ops[4]:
            registers[c] = registers[a] & registers[b]
        elif op_code == ops[5]:
            registers[c] = registers[a] & b
        elif op_code == ops[6]:
            registers[c] = registers[a] | registers[b]
        elif op_code == ops[7]:
            registers[c] = registers[a] | b
        elif op_code == ops[8]:
            registers[c] = registers[a]
        elif op_code == ops[9]:
            registers[c] = a
        elif op_code == ops[10]:
            registers[c] = int(a > registers[b])
        elif op_code == ops[11]:
            registers[c] = int(registers[a] > b)
        elif op_code == ops[12]:
            registers[c] = int(registers[a] > registers[b])
        elif op_code == ops[13]:
            registers[c] = int(a == registers[b])
        elif op_code == ops[14]:
            registers[c] = int(registers[a] == b)
        elif op_code == ops[15]:
            registers[c] = int(registers[a] == registers[b])
            
    
    return num_behave, registers[0]

#%%
# Day 17: Reservoir Research

@time_this_func
def day17(visualize = False):
    clay_inds = []
    with open("input17.txt") as f:
        for l in f:
            new = l.strip().split(",")
            if "y" in new[0]:
                new = new[::-1]
            
            for i in range(len(new)):
                if ".." in new[i]:
                    new[i] = sorted([int(x) for x in new[i].strip()[2:].split("..")])
                else:
                    new[i] = int(new[i][2:])
                    
            clay_inds.append(new)
            
    all_x = []
    all_y = []
    for inds in clay_inds:
        if type(inds[0]) == int:
            all_x += [inds[0]]
        else:
            all_x += inds[0]
        if type(inds[1]) == int:
            all_y += [inds[1]]
        else:
            all_y += inds[1]
            
    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)
    
    cave = np.zeros([max_y+1, max_x+1])
    
    for inds in clay_inds:
        if type(inds[0]) == int:
            x = inds[0]
            for y in range(inds[1][0], inds[1][1]+1):
                cave[y,x] = 2
        else:
            y = inds[1]
            for x in range(inds[0][0], inds[0][1]+1):
                cave[y,x] = 2
                
    source = (min_y-1,500)
    
    def fall_and_is_full(fall_from):
        loc = fall_from
        while cave[(loc[0]+1, loc[1])] in {0,0.5}:
            loc = (loc[0]+1, loc[1])
            if loc[0]+1 == max_y + 1:
                return True
            
        if cave[(loc[0]+1, loc[1])] == 2:
            return False
        
        while cave[(loc[0], loc[1]-1)] != 2 and cave[(loc[0]+1, loc[1]-1)] != 0:
            loc = (loc[0], loc[1]-1)
        if cave[(loc[0]+1, loc[1]-1)] == 0:
            return True
        
        while cave[(loc[0], loc[1]+1)] != 2 and cave[(loc[0]+1, loc[1]+1)] != 0:
            loc = (loc[0], loc[1]+1)
        if cave[(loc[0]+1, loc[1]+1)] == 0:
            return True
        else:
            return False
         
    def fall_only(fall_from):
        loc = fall_from
        while cave[(loc[0]+1, loc[1])] in {0,0.5}:
            loc = (loc[0]+1, loc[1])
            cave[loc] = 0.5
            if loc[0]+1 == max_y + 1:
                return       
        
    def fall_and_fill(fall_from):
        loc = fall_from
        while cave[(loc[0]+1, loc[1])] in {0,0.5}:
            loc = (loc[0]+1, loc[1])
            cave[loc] = 0.5
            if loc[0]+1 == max_y + 1:
                return       
        cave[loc] = 1
        
        fall_left = False
        fall_right = False
        while cave[(loc[0], loc[1]-1)] != 2 and cave[(loc[0]+1, loc[1]-1)] != 0:
            loc = (loc[0], loc[1]-1)
            cave[loc] = 0.5
        if cave[(loc[0], loc[1]-1)] != 2:
            if not fall_and_is_full((loc[0]-1,loc[1]-1)):
                fall_and_fill((loc[0]-1,loc[1]-1))
            else:
                fall_only((loc[0]-1,loc[1]-1))
            fall_left = True
            
        while cave[(loc[0], loc[1]+1)] != 2 and cave[(loc[0]+1, loc[1]+1)] != 0:
            loc = (loc[0], loc[1]+1)
            cave[loc] = 0.5
        if cave[(loc[0], loc[1]+1)] != 2:
            if not fall_and_is_full((loc[0]-1,loc[1]+1)):
                fall_and_fill((loc[0]-1,loc[1]+1))
            else:
                fall_only((loc[0]-1,loc[1]+1))
            fall_right = True
        
        elif not fall_left and not fall_right:
            while cave[(loc[0], loc[1]-1)] != 2 and cave[(loc[0]+1, loc[1]-1)] != 0:
                loc = (loc[0], loc[1]-1)
                cave[loc] = 1
            while cave[(loc[0], loc[1]+1)] != 2 and cave[(loc[0]+1, loc[1]+1)] != 0:
                loc = (loc[0], loc[1]+1)
                cave[loc] = 1
            if cave[fall_from] == 0:
                fall_and_fill(fall_from)
            else:
                fall_and_fill((fall_from[0]-1, fall_from[1]))
            
    fall_and_fill(source)
    
    for i in range(cave.shape[0]):
        for j in range(min_x-1, max_x+1):
            if cave[i,j] == 0.5 and cave[i-1,j] == 1:
                cave[i,j] = 1
    
    if visualize:
        plt.figure()
        plt.imshow(cave[:,min_x-1:max_x+1])
             
    total_water = np.sum((cave == 1) + (cave == 0.5))
    permanent_water = np.sum(cave == 1)
    
    return total_water, permanent_water

#%%
# Day 18: Settlers of The North Pole

@time_this_func
def day18(visualize = False):
    landscape = []
    with open("input18.txt") as f:
        for l in f:
            row = []
            for s in l.strip():
                if s == ".":
                    row.append(0)
                elif s == "|":
                    row.append(1)
                elif s == "#":
                    row.append(2)
            landscape.append(row)
    landscape = np.pad(np.array(landscape), 1, constant_values = -1)
    
    def counts(loc):
        bare = 0
        tree = 0
        lumber = 0
        for i in range(loc[0]-1, loc[0]+2):
            for j in range(loc[1]-1, loc[1]+2):
                if (i,j) == loc:
                    continue
                
                if last_landscape[i,j] == 0:
                    bare += 1
                elif last_landscape[i,j] == 1:
                    tree += 1
                elif last_landscape[i,j] == 2:
                    lumber += 1
                    
        return bare, tree, lumber
    
    m = 0
    history = [landscape.copy()]
    while True:
        m += 1
        last_landscape = landscape.copy()
        for i in range(1, landscape.shape[0]-1):
            for j in range(1, landscape.shape[1]-1):
                bare_count, tree_count, lumber_count = counts((i,j))
                
                if last_landscape[i,j] == 0:
                    if tree_count >= 3:
                        landscape[i,j] = 1
                        
                if last_landscape[i,j] == 1:
                    if lumber_count >= 3:
                        landscape[i,j] = 2
                        
                if last_landscape[i,j] == 2:
                    if not (tree_count >= 1 and lumber_count >= 1):
                        landscape[i,j] = 0
                        
        if True in {np.array_equiv(landscape,x) for x in history}:
            break
        
        history.append(landscape.copy())
        
        if m == 10-1:                 
            total_value_10 = np.sum(landscape == 1) * np.sum(landscape == 2)
            
    first_appearance = [np.array_equiv(landscape,x) for x in history].index(True)
    cycle_len = m-first_appearance
    
    minutes_past_cycle_start = (1000000000-first_appearance)%cycle_len
    end_landscape = history[first_appearance+minutes_past_cycle_start]
    total_value_end = np.sum(end_landscape == 1) * np.sum(end_landscape == 2)
    
    if visualize:
        fig, ax = plt.subplots()
        ims = []
        for i in range(len(history)):
            im = ax.imshow(history[i], animated=True)
            ims.append([im])
        
        ani = animation.ArtistAnimation(fig, ims, interval=45, blit=True, repeat_delay=1000)
        plt.show()
    
    if not visualize:
        return total_value_10, total_value_end
    else:
        return total_value_10, total_value_end, ani

#%%
# Day 19: Go With The Flow

@time_this_func
def day19():
    operations = []
    with open("input19.txt") as f:
        for l in f:
            new = l.split()
            for i in range(1,len(new)):
                new[i] = int(new[i])
            
            if "#ip" in new:
                pointer = new[1]
            else:
                operations.append(new)
    
    registers = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    i = registers[pointer]
    while i in range(len(operations)):
        operation = operations[i]
        
        op_code = operation[0]
        a = operation[1]
        b = operation[2]
        c = operation[3]
    
        if op_code == "addr":
            registers[c] = registers[a] + registers[b]
        elif op_code == "addi":
            registers[c] = registers[a] + b
        elif op_code == "mulr":
            registers[c] = registers[a] * registers[b]
        elif op_code == "muli":
            registers[c] = registers[a] * b
        elif op_code == "banr":
            registers[c] = registers[a] & registers[b]
        elif op_code == "bani":
            registers[c] = registers[a] & b
        elif op_code == "borr":
            registers[c] = registers[a] | registers[b]
        elif op_code == "bori":
            registers[c] = registers[a] | b
        elif op_code == "setr":
            registers[c] = registers[a]
        elif op_code == "seti":
            registers[c] = a
        elif op_code == "gtir":
            registers[c] = int(a > registers[b])
        elif op_code == "gtri":
            registers[c] = int(registers[a] > b)
        elif op_code == "gtrr":
            registers[c] = int(registers[a] > registers[b])
        elif op_code == "eqir":
            registers[c] = int(a == registers[b])
        elif op_code == "eqri":
            registers[c] = int(registers[a] == b)
        elif op_code == "eqrr":
            registers[c] = int(registers[a] == registers[b])
            
        registers[pointer] += 1
        i = registers[pointer]
            
    after_start_0 = registers[0]
    
    
    #Still don't understand how to do these problems.. Requires rewriting the input.
    registers = {0:1, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    i = registers[pointer]
    for _ in range(20): #20 enough to get register[pointer] to the "decider" value
        operation = operations[i]
        
        op_code = operation[0]
        a = operation[1]
        b = operation[2]
        c = operation[3]
    
        if op_code == "addr":
            registers[c] = registers[a] + registers[b]
        elif op_code == "addi":
            registers[c] = registers[a] + b
        elif op_code == "mulr":
            registers[c] = registers[a] * registers[b]
        elif op_code == "muli":
            registers[c] = registers[a] * b
        elif op_code == "banr":
            registers[c] = registers[a] & registers[b]
        elif op_code == "bani":
            registers[c] = registers[a] & b
        elif op_code == "borr":
            registers[c] = registers[a] | registers[b]
        elif op_code == "bori":
            registers[c] = registers[a] | b
        elif op_code == "setr":
            registers[c] = registers[a]
        elif op_code == "seti":
            registers[c] = a
        elif op_code == "gtir":
            registers[c] = int(a > registers[b])
        elif op_code == "gtri":
            registers[c] = int(registers[a] > b)
        elif op_code == "gtrr":
            registers[c] = int(registers[a] > registers[b])
        elif op_code == "eqir":
            registers[c] = int(a == registers[b])
        elif op_code == "eqri":
            registers[c] = int(registers[a] == b)
        elif op_code == "eqrr":
            registers[c] = int(registers[a] == registers[b])
            
        registers[pointer] += 1
        i = registers[pointer]
        
    num = registers[pointer+1]
    
    def get_divisors(n):
        divisors = []
        for i in range(1,n+1):
            if n%i == 0:
                divisors.append(i)
        return divisors
    
    after_start_1 = sum(get_divisors(num))
    
    
    return after_start_0, after_start_1

#%%
# Day 20: A Regular Map

@time_this_func
def day20(visualize = False): 
    with open("input20.txt") as f:
        regex = f.read().strip()
    
    rooms = {(0,0)}
    doors = set()
    
    def follow(from_loc, move_direction):
        if move_direction == "E":
            loc = (from_loc[0], from_loc[1] + 1)
            doors.add(tuple(loc))
            loc = (loc[0], loc[1] + 1)
            rooms.add(tuple(loc))
        elif move_direction == "W":
            loc = (from_loc[0], from_loc[1] - 1)
            doors.add(tuple(loc))
            loc = (loc[0], loc[1] - 1)
            rooms.add(tuple(loc))
        elif move_direction == "N":
            loc = (from_loc[0] - 1, from_loc[1])
            doors.add(tuple(loc))
            loc = (loc[0] - 1, loc[1])
            rooms.add(tuple(loc))
        elif move_direction == "S":
            loc = (from_loc[0] + 1, from_loc[1])
            doors.add(tuple(loc))
            loc = (loc[0] + 1, loc[1])
            rooms.add(tuple(loc))
        
        return loc
    
    def get_parens_chunk(string):
        i = 0
        if string[i] == "(":
            enclosed = 1
        else:
            raise Exception("Improper use")
        
        while enclosed != 0:
            i += 1
            if string[i] == "(":
                enclosed += 1
            elif string[i] == ")":
                enclosed -= 1
                
        return string[1:i], i

    def explore(from_loc, string):
        i = 0
        loc = from_loc
        while i in range(len(string)):
            if string[i] in ["N","S","W","E"]:
                loc = follow(loc, string[i])
                
            elif string[i] == "|":
                loc = from_loc
                
            elif string[i] == "(":
                in_parens, i_shift = get_parens_chunk(string[i:])
                explore(loc, in_parens)
                i += i_shift
            i += 1
            
    explore((0,0), regex[1:-1])
    
    min_row = min([x[0] for x in rooms | doors])
    max_row = max([x[0] for x in rooms | doors])
    min_col = min([x[1] for x in rooms | doors])
    max_col = max([x[1] for x in rooms | doors])
    
    floorplan = np.zeros([max_row-min_row+1, max_col-min_col+1])
    for room, door in zip(rooms,doors):
        floorplan[room[0]-min_row, room[1]-min_col] = 1
        floorplan[door[0]-min_row, door[1]-min_col] = 2
    
    history = [floorplan.copy()]
    
    def far_rooms_info(home):
        paths = [[home]]
        most_doors = 0
        far_rooms = set()
        while True:
            new_paths = []
            for path in paths:
                latest = path[-1]
                if (latest[0]-1, latest[1]) in rooms | doors and \
                    (latest[0]-1, latest[1]) not in set(sum(paths, []) + sum(new_paths, [])):
                    new_paths.append(path + [(latest[0]-1, latest[1])])
                if (latest[0]+1, latest[1]) in rooms | doors and \
                    (latest[0]+1, latest[1]) not in set(sum(paths, []) + sum(new_paths, [])):
                    new_paths.append(path + [(latest[0]+1, latest[1])])
                if (latest[0], latest[1]-1) in rooms | doors and \
                    (latest[0], latest[1]-1) not in set(sum(paths, []) + sum(new_paths, [])):
                    new_paths.append(path + [(latest[0], latest[1]-1)])
                if (latest[0], latest[1]+1) in rooms | doors and \
                    (latest[0], latest[1]+1) not in set(sum(paths, []) + sum(new_paths, [])):
                    new_paths.append(path + [(latest[0], latest[1]+1)])
            
            if len(new_paths) == 0:
                break
            
            for l in [x[-1] for x in new_paths]:
                floorplan[l[0]-min_row, l[1]-min_col] = 10
            history.append(floorplan.copy())
                    
            paths = new_paths
            most_doors_loop = [sum([1 for x in y if x in doors]) for y in paths][0]
            if most_doors_loop >= 1000:
                far_rooms = far_rooms | {x for x in [y[-1] for y in paths] if x in rooms}
            if most_doors_loop > most_doors:
                most_doors = most_doors_loop
                
        return most_doors, len(far_rooms)
                
    most_doors_away, num_far_rooms = far_rooms_info((0,0))
    
    if visualize:
        fig, ax = plt.subplots()
        ims = []
        for i in range(len(history)):
            im = ax.imshow(history[i], animated=True, cmap = "hot")
            ims.append([im])
        
        ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=10)
        plt.show()
    
    if not visualize:
        return most_doors_away, num_far_rooms
    else:
        return most_doors_away, num_far_rooms, ani

#%%
# Day 21: Chronal Conversion

@time_this_func
def day21():
    print("16 minute runtime")
    operations = []
    with open("input21.txt") as f:
        for l in f:
            new = l.split()
            for i in range(1,len(new)):
                new[i] = int(new[i])
            
            if "#ip" in new:
                pointer = new[1]
            else:
                operations.append(new)
    
    #Read through input to see that program exits if reg(0) matches reg(4) when checked
    stop_states = set()
    first_stop_found = False
    registers = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    i = registers[pointer]
    while i in range(len(operations)):
        operation = operations[i]
        
        op_code = operation[0]
        a = operation[1]
        b = operation[2]
        c = operation[3]
    
        if op_code == "addr":
            registers[c] = registers[a] + registers[b]
        elif op_code == "addi":
            registers[c] = registers[a] + b
        elif op_code == "mulr":
            registers[c] = registers[a] * registers[b]
        elif op_code == "muli":
            registers[c] = registers[a] * b
        elif op_code == "banr":
            registers[c] = registers[a] & registers[b]
        elif op_code == "bani":
            registers[c] = registers[a] & b
        elif op_code == "borr":
            registers[c] = registers[a] | registers[b]
        elif op_code == "bori":
            registers[c] = registers[a] | b
        elif op_code == "setr":
            registers[c] = registers[a]
        elif op_code == "seti":
            registers[c] = a
        elif op_code == "gtir":
            registers[c] = int(a > registers[b])
        elif op_code == "gtri":
            registers[c] = int(registers[a] > b)
        elif op_code == "gtrr":
            registers[c] = int(registers[a] > registers[b])
        elif op_code == "eqir":
            registers[c] = int(a == registers[b])
        elif op_code == "eqri":
            registers[c] = int(registers[a] == b)
        elif op_code == "eqrr":
            if {a,b} == {0,4}:
                if not first_stop_found:
                    first_stop = registers[4]
                    first_stop_found = True
                    
                if registers[4] not in stop_states:
                    stop_states.add(registers[4])
                    latest = registers[4]
                else:
                    break
            registers[c] = int(registers[a] == registers[b])
            
        registers[pointer] += 1
        i = registers[pointer]
    
    last_stop = latest
    
    return first_stop, last_stop

#%%
# Day 22: Mode Maze

@time_this_func
def day22():
    with open("input22.txt") as f:
        for l in f:
            if "depth" in l:
                depth = int(l.split()[1])
            elif "target" in l:
                target = l.split()[1]
                target = tuple(int(x) for x in target.split(",")[::-1])
    
    cave_erosion_level = np.zeros([target[0]+1, target[1]+1])
    for y in range(cave_erosion_level.shape[0]):
        for x in range(cave_erosion_level.shape[1]):
            if x == 0 and y == 0:
                geo_ind = 0
            elif x == target[1] and y == target[0]:
                geo_ind = 0
            elif y == 0:
                geo_ind = x*16807
            elif x == 0:
                geo_ind = y*48271
            else:
                geo_ind = cave_erosion_level[(y,x-1)] * cave_erosion_level[(y-1,x)]
                
            erosion_level = (geo_ind + depth)%20183
            cave_erosion_level[y,x] = erosion_level
    
    cave_type = cave_erosion_level%3
    
    cave_risk = int(np.sum(cave_type))
    
    
    def add_row(cave_erosion_level, cave_type):
        new_row_ind = cave_erosion_level.shape[0]
        num_cols = cave_erosion_level.shape[1]
        new_row_erosion_level = np.zeros([1,num_cols])
        for j in range(num_cols):
            if j == 0:
                new_row_erosion_level[0,j] = (new_row_ind*48271 + depth)%20183
            else:
                new_row_erosion_level[0,j] = (new_row_erosion_level[0,j-1]*cave_erosion_level[-1,j] + depth)%20183
        
        cave_erosion_level = np.vstack([cave_erosion_level, new_row_erosion_level])
        cave_type = np.vstack([cave_type, new_row_erosion_level%3])
        
        return cave_erosion_level, cave_type
    
    def add_col(cave_erosion_level, cave_type):
        new_col_ind = cave_erosion_level.shape[1]
        num_rows = cave_erosion_level.shape[0]
        new_col_erosion_level = np.zeros([num_rows,1])
        for i in range(num_rows):
            if i == 0:
                new_col_erosion_level[i,0] = (new_col_ind*16807 + depth)%20183
            else:
                new_col_erosion_level[i,0] = (new_col_erosion_level[i-1,0]*cave_erosion_level[i,-1] + depth)%20183
        
        cave_erosion_level = np.hstack([cave_erosion_level, new_col_erosion_level])
        cave_type = np.hstack([cave_type, new_col_erosion_level%3])
        
        return cave_erosion_level, cave_type
            
    tool_reqs = {0: ["gear", "torch"], 1: ["gear", "nothing"], 2: ["torch", "nothing"]}
    
    def switch_tool(curr_type, curr_gear):
        return tool_reqs[curr_type][::-1][tool_reqs[curr_type].index(curr_gear)]
    
    def path_time(path_locs, path_tools):
        time = 0
        for i in range(1,len(path_tools)):
            time += 1
            if path_tools[i] != path_tools[i-1]:
                time += 7
                
        if path_locs[-1] == target and path_tools[-1] != "torch":
            time += 7
            
        return time
    
    def min_finish_time(path_locs, path_tools):
        curr_time = path_time(path_locs, path_tools)
        
        latest = path_locs[-1]
        min_time_target = abs(latest[0]-target[0]) + abs(latest[1]-target[1])
        
        return curr_time + min_time_target
    
    naive_path = [(0,0)]
    naive_tools = ["torch"]
    for i in range(1, target[0]+1):
        naive_path.append((i,0))
        if naive_tools[-1] in tool_reqs[cave_type[(i,0)]]:
            naive_tools.append(naive_tools[-1])
        else:
            naive_tools.append(switch_tool(cave_type[i-1,0], naive_tools[-1]))
    for j in range(1, target[1]+1):
        naive_path.append((target[0],j))
        if naive_tools[-1] in tool_reqs[cave_type[(target[0],j)]]:
            naive_tools.append(naive_tools[-1])
        else:
            naive_tools.append(switch_tool(cave_type[target[0],j-1], naive_tools[-1]))
            
    naive_time = path_time(naive_path, naive_tools)
    min_time = naive_time
    
    paths = [[(0,0)]]
    tools = [["torch"]]
    visited = {((0,0),"torch"):0}
    while True:
        new_paths = []
        new_tools = []
        for path, tool in zip(paths,tools):
            latest_loc = path[-1]
            latest_tool = tool[-1]
            
            next_pos_locs = [(latest_loc[0]-1, latest_loc[1]),\
                             (latest_loc[0]+1, latest_loc[1]),\
                             (latest_loc[0], latest_loc[1]-1),\
                             (latest_loc[0], latest_loc[1]+1)]
            
            for next_pos_loc in next_pos_locs:
                    
                if next_pos_loc not in path and \
                    next_pos_loc[0] in range(cave_type.shape[0]) and \
                    next_pos_loc[1] in range(cave_type.shape[1]):
                        
                    new_pos_path = path + [next_pos_loc]
                        
                    if latest_tool in tool_reqs[cave_type[next_pos_loc]]:  
                        next_pos_tool = latest_tool
                    else:
                        next_pos_tool = switch_tool(cave_type[latest_loc], latest_tool)
                    new_pos_tool = tool + [next_pos_tool]
                    
                    time_to_pos_loc = path_time(new_pos_path, new_pos_tool)
                    if (next_pos_loc, next_pos_tool) not in visited or time_to_pos_loc < visited[(next_pos_loc, next_pos_tool)]:
                        visited[(next_pos_loc, next_pos_tool)] = time_to_pos_loc
                        new_paths.append(new_pos_path)
                        new_tools.append(new_pos_tool)
                    
                    
        finished_paths = [i for i in range(len(new_paths)) if new_paths[i][-1] == target]
        for fin_i in finished_paths:
            finish_time = path_time(new_paths[fin_i], new_tools[fin_i])
            if finish_time < min_time:
                min_time = finish_time
        
        paths = []
        tools = []
        for path, tool in zip(new_paths, new_tools):
            if min_finish_time(path, tool) <= min_time:
                paths.append(path)
                tools.append(tool)
        
        if len(paths) == 0:
            break
        
        paths, tools = zip(*sorted(zip(paths, tools), key = lambda x: min_finish_time(x[0],x[1])))
        paths = paths[:200]
        tools = tools[:200]
        
        if cave_type.shape[0]-1 in {x[-1][0] for x in paths}:
            cave_erosion_level, cave_type = add_row(cave_erosion_level, cave_type)
        if cave_type.shape[1]-1 in {x[-1][1] for x in paths}:
            cave_erosion_level, cave_type = add_col(cave_erosion_level, cave_type)
            
            
    return cave_risk, min_time

#%%
# Day 23: Experimental Emergency Teleportation

@time_this_func
def day23():
    print("finished part1, couldn't solve part2\n")
    class nanobot():
        def __init__(self, pos, rad):
            self.pos = pos
            self.rad = rad
            
        def dist_to(self, point):
            return abs(self.pos[0] - point[0]) + abs(self.pos[1] - point[1]) + abs(self.pos[2] - point[2]) 
        
        def contains(self, point):
            return self.dist_to(point) <= self.rad
        
        def overlaps(self, other):
            return self.dist_to(other.pos) <= self.rad + other.rad
                
        def all_points(self):
            for x in range(self.pos[0]-self.rad, self.pos[0]+self.rad+1):
                for y in range(self.pos[1]-self.rad, self.pos[1]+self.rad+1):
                    for z in range(self.pos[2]-self.rad, self.pos[2]+self.rad+1):
                        if not self.dist_to((x,y,z)) == self.rad:
                            continue
                        yield (x,y,z)
        
    bots = []
    with open("input23.txt") as f:
        for l in f:
            new = l.strip().split(">")
            pos = tuple(int(x) for x in new[0][5:].split(","))
            rad = int(new[1][4:])
            bots.append(nanobot(pos, rad))
            
    max_rad_i = np.argmax([x.rad for x in bots])
    num_bots_in_max_rad = sum([1 for x in bots if bots[max_rad_i].contains(x.pos)])
    
    
    #Could not do part2 after days and days of work. Come back to it?
    
    #Stolen solution:
    bots = [[bot.pos[0], bot.pos[1], bot.pos[2], bot.rad] for bot in bots]
        
    from queue import PriorityQueue
    
    q = PriorityQueue()
    for x,y,z,r in bots:
      d = abs(x) + abs(y) + abs(z)
      q.put((max(0, d - r),1))
      q.put((d + r + 1,-1))
    count = 0
    maxCount = 0
    result = 0
    while not q.empty():
      dist,e = q.get()
      count += e
      if count > maxCount:
        result = dist
        maxCount = count
    
    closest_most_contacts = result
    
    
    return num_bots_in_max_rad, closest_most_contacts
    # class box():
    #     def __init__(self, min_x, min_y, min_z, max_x, max_y, max_z):
    #         self.min_x = min_x
    #         self.min_y = min_y
    #         self.min_z = min_z
    #         self.max_x = max_x
    #         self.max_y = max_y
    #         self.max_z = max_z
            
    #     def to_origin(self):
    #         corners = []
    #         for x in [self.min_x, self.max_x]:
    #             for y in [self.min_y, self.max_y]:
    #                 for z in [self.min_z, self.max_z]:
    #                     corners.append(sum([x,y,z]))
    #         return min(corners)
            
    #     def volume(self):
    #         return (self.max_x-self.min_x)*(self.max_y-self.min_y)*(self.max_z-self.min_z)
        
    #     def all_points(self):
    #         for x in range(self.min_x, self.max_x+1):
    #             for y in range(self.min_y, self.max_y+1):
    #                 for z in range(self.min_z, self.max_z+1):
    #                     yield (x,y,z)
    
    #     def break_into_eight(self):
    #         min_x = self.min_x
    #         min_y = self.min_y
    #         min_z = self.min_z
    #         max_x = self.max_x
    #         max_y = self.max_y
    #         max_z = self.max_z
            
    #         box1 = box(min_x, min_y, min_z, (min_x+max_x)//2, (min_y+max_y)//2, (min_z+max_z)//2)
    #         box2 = box(min_x, (min_y+max_y)//2+1, min_z, (min_x+max_x)//2, max_y, (min_z+max_z)//2)
    #         box3 = box((min_x+max_x)//2+1, min_y, min_z, max_x, (min_y+max_y)//2, (min_z+max_z)//2)
    #         box4 = box((min_x+max_x)//2+1, (min_y+max_y)//2+1, min_z, max_x, max_y, (min_z+max_z)//2)
    #         box5 = box(min_x, min_y, (min_z+max_z)//2+1, (min_x+max_x)//2, (min_y+max_y)//2, max_z)
    #         box6 = box(min_x, (min_y+max_y)//2+1, (min_z+max_z)//2+1, (min_x+max_x)//2, max_y, max_z)
    #         box7 = box((min_x+max_x)//2+1, min_y, (min_z+max_z)//2+1, max_x, (min_y+max_y)//2, max_z)
    #         box8 = box((min_x+max_x)//2+1, (min_y+max_y)//2+1, (min_z+max_z)//2+1, max_x, max_y, max_z)
            
    #         return (box1, box2, box3, box4, box5, box6, box7, box8)
        
    #     def break_into_two(self):
    #         min_x = self.min_x
    #         min_y = self.min_y
    #         min_z = self.min_z
    #         max_x = self.max_x
    #         max_y = self.max_y
    #         max_z = self.max_z
            
    #         box1 = box(min_x, min_y, min_z, max_x, max_y, (min_z+max_z)//2)
    #         box2 = box(min_x, min_y, (min_z+max_z)//2+1, max_x, max_y, max_z)
            
    #         return (box1, box2)
        
    #     def break_horizontal(self):
    #         min_x = self.min_x
    #         min_y = self.min_y
    #         min_z = self.min_z
    #         max_x = self.max_x
    #         max_y = self.max_y
    #         max_z = self.max_z
            
    #         box1 = box(min_x, min_y, min_z, (min_x+max_x)//2, (min_y+max_y)//2, max_z)
    #         box2 = box(min_x, (min_y+max_y)//2+1, min_z, (min_x+max_x)//2, max_y, max_z)
    #         box3 = box((min_x+max_x)//2+1, min_y, min_z, max_x, (min_y+max_y)//2, max_z)
    #         box4 = box((min_x+max_x)//2+1, (min_y+max_y)//2+1, min_z, max_x, max_y, max_z)
            
    #         return (box1, box2, box3, box4)
            
    
    # all_pos = {x.pos for x in bots}
    # min_x = min([x[0] for x in all_pos])
    # min_y = min([x[1] for x in all_pos])
    # min_z = min([x[2] for x in all_pos])
    # max_x = max([x[0] for x in all_pos])
    # max_y = max([x[1] for x in all_pos])
    # max_z = max([x[2] for x in all_pos])
    
    # def box_in_range(cube, bot):
    #     bot_x = bot.pos[0]
    #     bot_y = bot.pos[1]
    #     bot_z = bot.pos[2]
        
    #     d = 0
    #     if bot_x <= cube.min_x: d += cube.min_x - bot_x
    #     if bot_x >= cube.max_x: d += bot_x - cube.max_x
    #     if bot_y <= cube.min_y: d += cube.min_y - bot_y
    #     if bot_y >= cube.max_y: d += bot_y - cube.max_y
    #     if bot_z <= cube.min_z: d += cube.min_z - bot_z
    #     if bot_z >= cube.max_z: d += bot_z - cube.max_z
        
    #     return d <= bot.rad
    
    # def get_most_covered_point(all_points):
    #     point_counts = []
    #     for p in all_points:
    #         in_range_num = 0
    #         for bot in bots:
    #             in_range_num += bot.contains(p)
    #         point_counts.append((p,in_range_num))
    #     result = sorted([x for x in point_counts if x[1] == max([y[1] for y in point_counts])], key = lambda x: sum(x[0]))[0]
    #     return (sum(result[0]), result[1])
    
    
    # boxes = box(min_x, min_y, min_z, max_x, max_y, max_z).break_into_eight()
    # boxes_and_connections = []
    # levels = 0
    # while True:
    #     connections = [0]*len(boxes)
    #     for i, cube in enumerate(boxes):
    #         for bot in bots:
    #             connections[i] += box_in_range(cube, bot)
                
    #     boxes_and_connections.append(tuple(zip(*sorted(zip(boxes, connections), key = lambda x: x[1]))))
        
    #     best_boxes_inds = [i for i in range(len(connections)) if connections[i] == max(connections)]
    #     if len(best_boxes_inds) == 1:
    #         best_box = boxes[best_boxes_inds[0]]
    #     else:
    #         all_best = sorted([boxes[i] for i in best_boxes_inds], key = lambda x: x.to_origin())
    #         best_box = all_best[0]
    
    #     if best_box.volume() < 1000:
    #         break
            
    #     boxes = best_box.break_into_eight()
    #     levels += 1
    
    # naive_start = get_most_covered_point(best_box.all_points())
    
    # class oc_node():
    #     def __init__(self, val, p):
    #         self.val = val
    #         self.p = p
            
    #     def add_children(self):
    #         new_boxes = self.val.break_into_eight()
    #         self.c1 = oc_node(new_boxes[0], self)
    #         self.c2 = oc_node(new_boxes[1], self)
    #         self.c3 = oc_node(new_boxes[2], self)
    #         self.c4 = oc_node(new_boxes[3], self)
    #         self.c5 = oc_node(new_boxes[4], self)
    #         self.c6 = oc_node(new_boxes[5], self)
    #         self.c7 = oc_node(new_boxes[6], self)
    #         self.c8 = oc_node(new_boxes[7], self)
            
    #     def children_iter(self):
    #         cs = [self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8]
    #         for c in cs:
    #             if c != None:
    #                 yield c
    
    # orig_box = box(min_x, min_y, min_z, max_x, max_y, max_z)
    
    # root_node = oc_node(orig_box, None)
    # root_node.add_children()
    
    # #Giving up
    
    # def in_skip_list(inds_tup):
    #     for depth in range(25):
    #         if inds_tup[:depth+1] in skip_set:
    #             return True
    #     return False
    
    # all_orders = product(range(8), repeat = 27)
    
    # skip_set = set()
    # current_best = naive_start
    # for o, order in enumerate(all_orders):
    #     if o%100000 == 0: print(o)
    #     boxes = box(min_x, min_y, min_z, max_x, max_y, max_z).break_into_eight()
    #     if not in_skip_list(order):
    #         skip = False
    #         for l, ind in enumerate(order):
    #             connections = 0
    #             for bot in bots:
    #                 connections += box_in_range(boxes[ind], bot)
    #             if connections < current_best[1]:
    #                 skip_set.add(order[:l+1])
    #                 skip = True
    #                 break
    #             boxes = boxes[ind].break_into_eight()
                
    #         if not skip:  
    #             for end_box in boxes:
    #                 result = get_most_covered_point(end_box.all_points())
    #                 if result[1] > current_best[1] or (result[1] == current_best[1] and result[0] < current_best[0]):
    #                     current_best = result
    #                     print(current_best)
    
    #25 for loops do it next time? Cut off whole branches somehow.
    # current_best = naive_start
    # boxes_0 = box(min_x, min_y, min_z, max_x, max_y, max_z).break_into_eight()
    # for l1_i in range(8):
    #     connections = 0
    #     for bot in bots:
    #         connections += box_in_range(boxes_0[l1_i])
    #     if connections < current_best[1]:
    #         continue
    #     boxes_1 = boxes_0[l1_i].break_into_eight()
        
    #     for l2_i in range(8):
    #         connections = 0
    #         for bot in bots:
    #             connections += box_in_range(boxes_1[l2_i])
    #         if connections < current_best[1]:
    #             continue
    #         boxes_2 = boxes_1[l2_i].break_into_eight()
    
            
            
        
    
    
    
    
    
    # current_best = list(naive_start)
    # all_orders = product(range(8), repeat = levels + 1) #25 levels
    # order_num = 0
    # for order in all_orders:
    #     order_num += 1
    #     if order_num % 1000 == 0: print(order_num)
        
    #     boxes = box(min_x, min_y, min_z, max_x, max_y, max_z).break_into_eight()
    #     level = 0
    #     skip = False
    #     while True:
    #         connections = [0]*len(boxes)
    #         for i, cube in enumerate(boxes):
    #             for bot in bots:
    #                 connections[i] += box_in_range(cube, bot)
                    
    #         if connections[order[level]] < current_best[1]:
    #             skip = True
    #             break
            
    #         next_box = boxes[order[level]]
    
    #         if next_box.volume() < 1000:
    #             break
                
    #         boxes = next_box.break_into_eight()
    #         levels += 1
            
    #     if not skip:
    #         result = get_most_covered_point(next_box.all_points())
    #         if result[1] == current_best[1]:
    #             current_best[0] = min(result[0], current_best[0])
    #         elif result[1] > current_best[1]:
    #             current_best = list(result)
    
    # def get_best2(base_box):
    #     boxes = base_box.break_into_eight()
    #     while True:
    #         connections = [0]*len(boxes)
    #         for i, cube in enumerate(boxes):
    #             for bot in bots:
    #                 connections[i] += box_in_range(cube, bot)
                    
    #         boxes_and_connections.append(tuple(zip(*sorted(zip(boxes, connections), key = lambda x: x[1]))))
            
    #         best_boxes_inds = [i for i in range(len(connections)) if connections[i] == max(connections)]
    #         if len(best_boxes_inds) == 1:
    #             best_box = boxes[best_boxes_inds[0]]
    #         else:
    #             all_best = sorted([boxes[i] for i in best_boxes_inds], key = lambda x: x.to_origin())
    #             best_box = all_best[0]
    
    #         if best_box.volume() < 1000:
    #             break
                
    #         boxes = best_box.break_into_eight()
    
    #     point_counts = []
    #     for p in best_box.all_points():
    #         in_range_num = 0
    #         for bot in bots:
    #             in_range_num += bot.contains(p)
    #         point_counts.append((p,in_range_num))
    #     result = sorted([x for x in point_counts if x[1] == max([y[1] for y in point_counts])], key = lambda x: sum(x[0]))[0]
    #     final_best = (sum(result[0]), result[1])
    #     print(final_best)
    #     return final_best
    
    # go through all_boxes, results in (1276667298, 848)
    # best = [np.inf, 0]
    # for b in all_boxes:
    #     result = get_best2(b)
    #     if result[1] >= best[1]:
    #         if result[1] == best[1]:
    #             best[0] = min(best[0],result[0])
    #         else:
    #             best = [result[0], result[1]]
    
    # depth greater than 1 takes too long. Works fine for depth 1  
    # def get_best(boxes_and_connections, depth, bound = 0, volume_threshold = 1000,):
    #     start_boxes = boxes_and_connections[0]
    #     connections = boxes_and_connections[1]
    #     best_depth = []
    #     for i in range(depth):
    #         if connections[i] < bound:
    #             best_depth.append((np.inf, 0))
    #             continue
    #         if start_boxes[i].volume() <= volume_threshold: 
    #             point_counts = []
    #             for p in start_boxes[i].all_points():
    #                 in_range_num = 0
    #                 for bot in bots:
    #                     in_range_num += bot.contains(p)
    #                 point_counts.append((p,in_range_num))
    #             result = sorted([x for x in point_counts if x[1] == max([y[1] for y in point_counts])], key = lambda x: sum(x[0]))[0]
    #             best_depth.append((sum(result[0]), result[1]))
                
    #             if best_depth[-1][1] > bound:
    #                 bound = best_depth[-1][1]
                
    #         else:
    #             next_boxes = start_boxes[i].break_into_eight()
                
    #             connections = [0]*len(next_boxes)
    #             for j, cube in enumerate(next_boxes):
    #                 for bot in bots:
    #                     connections[j] += box_in_range(cube, bot)
                
    #             next_boxes_and_connections = tuple(zip(*(sorted(zip(next_boxes, connections), reverse = True, key = lambda x: x[1]))))
                
    #             best_depth.append(get_best(next_boxes_and_connections, depth, bound))
                
    #     return sorted(sorted(best_depth, key = lambda x: x[0]), reverse = True, key = lambda x: x[1])[0]
            
            
    # current_most_contacts = get_best(boxes_and_connections, 1, )[1]
    # print(get_best(boxes_and_connections, 3, current_most_contacts))
    
#%%
# Day 24: Day 24: Immune System Simulator 20XX

@time_this_func
def day24():
    from copy import deepcopy
    
    class battle_group():
        def __init__(self, group_type, units, unit_hp, weaknesses, immunities, damage_type, damage, initiative):
            self.type = group_type
            self.units = units
            self.unit_hp = unit_hp
            self.weaknesses = weaknesses
            self.immunities = immunities
            self.damage_type = damage_type
            self.damage = damage
            self.initiative = initiative
            
        def eff_pow(self):
            return self.units * self.damage
        
        def eval_attack(self, other):
            if self.damage_type in other.weaknesses:
                return self.eff_pow()*2
            elif self.damage_type in other.immunities:
                return 0
            else:
                return self.eff_pow()
            
        def attack(self, other):
            other.units -= self.eval_attack(other)//other.unit_hp
            if other.units < 0:
                other.units = 0
                
        def __str__(self):
            return f"{self.type} group: {self.units} {self.unit_hp}-hp units, weak to {self.weaknesses}, immune to {self.immunities}, does {self.damage} {self.damage_type} damage"
        def __repr__(self):
            return self.__str__()
                
    groups_og = []
    with open("input24.txt") as f:
        for l in f:
            if l.strip() == "":
                continue
            
            if l == "Immune System:\n":
                group_type = "immune"
                continue
            elif l == "Infection:\n":
                group_type = "infection"
                continue
            
            gen_info = l.split("hit points")[0].split()
            units = int(gen_info[0])
            unit_hp = int(gen_info[-1])
            
            if "weak" in l:
                weaknesses_info = l.split("weak to ")[1]
                if ";" in weaknesses_info:
                    weaknesses = set(weaknesses_info.split(";")[0].split(", "))
                else:
                    weaknesses = set(weaknesses_info.split(")")[0].split(", "))
            else:
                weaknesses = set()
            
            if "immune" in l:
                immunities_info = l.split("immune to ")[1]
                if ";" in immunities_info:
                    immunities = set(immunities_info.split(";")[0].split(", "))
                else:
                    immunities = set(immunities_info.split(")")[0].split(", "))
            else:
                immunities = set()
            
            damage_and_init_info = l.strip().split("an attack that does ")[1].split()
            damage = int(damage_and_init_info[0])
            damage_type = damage_and_init_info[1]
            initiative = int(damage_and_init_info[-1])
            
            groups_og.append(battle_group(group_type, units, unit_hp, weaknesses, immunities,  damage_type, damage, initiative))
    
    def battle(groups):
        while sum([x.units for x in groups if x.type == "immune"]) > 0 and sum([x.units for x in groups if x.type == "infection"]) > 0:
            units_before_round = sum([x.units for x in groups])
            groups = sorted(sorted(groups, key = lambda x: -x.initiative), key = lambda x: -x.eff_pow())
            targeting = [None]*len(groups)
            for g, group in enumerate(groups):
                if group.units == 0:
                    continue
                
                best_target = [None, 0]
                for o, other in enumerate(groups):
                    if group.type == other.type or o in targeting or other.units == 0:
                        continue
                    
                    damage_to_be_done = group.eval_attack(other)
                    if damage_to_be_done > best_target[1]:
                        best_target = [o,damage_to_be_done]
                    elif damage_to_be_done == best_target[1] and damage_to_be_done != 0:
                        if other.eff_pow() > groups[best_target[0]].eff_pow() or (other.eff_pow() == groups[best_target[0]].eff_pow() and other.initiative > groups[best_target[0]].initiative):
                            best_target = [o,damage_to_be_done]   
                            
                targeting[g] = best_target[0]
            
            targeting_groups = groups
            groups, targeting = zip(*sorted(zip(groups, targeting), key = lambda x: -x[0].initiative))
            for g, group in enumerate(groups):
                if group.units == 0 or targeting[g] == None:
                    continue
                
                group.attack(targeting_groups[targeting[g]])
                
            if sum([x.units for x in groups]) == units_before_round:
                return (units_before_round, None)
                
        remaining_units = int(sum(x.units for x in groups))
        winner = {x.type for x in groups if x.units > 0}.pop()
    
        return (remaining_units, winner)

    remaining_units_og = battle(deepcopy(groups_og))[0]
    
    
    boost = 0
    while True:
        boost += 1
        groups = deepcopy(groups_og)
        for g in groups:
            if g.type == "immune":
                g.damage += boost
        result = battle(groups)
        if result[1] == "immune":
            break
        
    winning_remaining_units = result[0]
    
    
    return remaining_units_og, winning_remaining_units

#%%
# Day 25: Four-Dimensional Adventure

@time_this_func
def day25():
    points = []
    with open("input25.txt") as f:
        for l in f:
            points.append(tuple(int(x) for x in l.strip().split(",")))
            
    def dist(p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) + abs(p1[3]-p2[3])
    
    links_to = []
    for from_point in points:
        links_to.append(set())
        for to_ind, to_point in enumerate(points):
            if dist(from_point, to_point) <= 3:
                links_to[-1].add(to_ind)
                
    constellations = []
    while len(links_to) > 0:
        new_constellation = links_to.pop(0)
        while True:
            no_new = True
            new_links_to = []
            for point_set in links_to:
                if not new_constellation.isdisjoint(point_set):
                    new_constellation = new_constellation | point_set
                    no_new = False
                else:
                    new_links_to.append(point_set)
            links_to = new_links_to
            if no_new:
                break
        constellations.append(new_constellation)
        
    num_constellations = len(constellations)
    
    return num_constellations