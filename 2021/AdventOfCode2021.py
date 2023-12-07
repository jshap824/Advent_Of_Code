#%% 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation  
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
# Day 1: Sonar Sweep

@time_this_func
def day1():
    with open("input1.txt") as f:
        measurements = np.array([int(x) for x in f.read().strip().split()])

    increases1 = sum(measurements[1:] > measurements[:-1])
    
    three_sums = measurements[:-2] + measurements[1:-1] + measurements[2:]
    increases2 = sum(three_sums[1:] > three_sums[:-1])
            
    return increases1, increases2

#%%
# Day 2: Dive!

@time_this_func
def day2():
    instructions = []
    with open("input2.txt") as f:
        for l in f:
            new = l.split()
            instructions.append((new[0], int(new[1])))
            
    hor1 = 0
    depth1 = 0
    aim = 0
    hor2 = 0
    depth2 = 0
    for direction, num in instructions:
        if direction == "forward":
            hor1 += num
            hor2 += num
            depth2 += aim*num
        elif direction == "down":
            depth1 += num
            aim += num
        elif direction == "up":
            depth1 -= num
            aim -= num
            
    hor_depth_mult1 = hor1*depth1
            
    hor_depth_mult2 = hor2*depth2
    
    return hor_depth_mult1, hor_depth_mult2

#%%
# Day 3: Binary Diagnostic

@time_this_func
def day3():
    with open("input3.txt") as f:
        bins = f.read().strip().split()
        
    bins_by_ind = [[x[i] for x in bins] for i in range(len(bins[0]))]
    
    gamma_bin = "".join([max(["0","1"], key=x.count) for x in bins_by_ind])
    epsilon_bin = "".join([min(["0","1"], key=x.count) for x in bins_by_ind])
            
    gamma = int(gamma_bin, 2)
    epsilon = int(epsilon_bin, 2)
    
    power_consumption = gamma * epsilon
    
    
    O2_bin = bins.copy()
    CO2_bin = bins.copy()
    i = -1
    while len(O2_bin) != 1 or len(CO2_bin) != 1:
        i += 1
        if len(O2_bin) != 1:
            of_interest = [x[i] for x in O2_bin]
            counts = [of_interest.count(x) for x in ["0","1"]]
            if counts[1] >= counts[0]:
                most_common = "1"
            else:
                most_common = "0"
            O2_bin = [x for x in O2_bin if x[i] == most_common]
        
        if len(CO2_bin) != 1:
            of_interest = [x[i] for x in CO2_bin]
            counts = [of_interest.count(x) for x in ["0","1"]]
            if counts[1] < counts[0]:
                least_common = "1"
            else:
                least_common = "0"
            CO2_bin = [x for x in CO2_bin if x[i] == least_common]
    
    O2 = int(O2_bin[0], 2)
    CO2 = int(CO2_bin[0], 2)
    
    life_support = O2 * CO2
    
    
    return power_consumption, life_support

#%%
# Day 4: Giant Squid

@time_this_func
def day4():
    with open("input4.txt") as f:
        raw = f.read().strip().split("\n\n")
    
    drawings = (x for x in [int(x) for x in raw[0].split(",")])
    
    boards = [ np.array([[int(x) for x in y.split()] for y in x.split("\n")]) for x in raw[1:]]
    num_boards = len(boards)
    
    def board_wins(board):
        needed = board.shape[0]
        marked_board = np.isin(board,drawn)
        line_sums = np.hstack((np.sum(marked_board, axis = 0), np.sum(marked_board, axis = 1)))
        if needed in line_sums:
            return True
        else:
            return False
    
    drawn = []
    winners = []
    while len(boards) != 0:
        new_draw = next(drawings)
        drawn.append(new_draw)
        to_remove = []
        for i, board in enumerate(boards):
            if board_wins(board):
                if len(boards) == num_boards or len(boards) == 1:
                    winners.append([board, drawn.copy(), new_draw])
                to_remove.append(i)
        
        for i in to_remove[::-1]:
            del boards[i]
        
    marked_winning_board = np.isin(winners[0][0], winners[0][1])
    winner_score = np.sum(winners[0][0] * (marked_winning_board == False))*winners[0][2]
    
    marked_last_winning_board = np.isin(winners[-1][0], winners[-1][1])
    last_winner_score = np.sum(winners[-1][0] * (marked_last_winning_board == False))*winners[-1][2]
    
    return winner_score, last_winner_score

#%%
# Day 5: Hydrothermal Venture

@time_this_func
def day5(visualize = False):
    vent_coords = []
    with open("input5.txt") as f:
        for l in f:
            new = l.split(" -> ")
            vent_coords.append([[int(x) for x in y.split(",")] for y in new])
            
    hor_vert_vent_coords = [x for x in vent_coords if x[0][0] == x[1][0] or x[0][1] == x[1][1]]
    
    max_x = max(sum([[x[0][0], x[1][0]] for x in vent_coords], []))
    max_y = max(sum([[x[0][1], x[1][1]] for x in vent_coords], []))
    
    floor_map = np.zeros([max_x+1, max_y+1])
    
    def map_points(coords):
        for p1, p2 in coords:
            num_points = max(abs(p1[0]-p2[0])+1, abs(p1[1]-p2[1])+1)
            for x,y in zip(np.linspace(p1[0], p2[0], num_points, dtype = np.int32),
                           np.linspace(p1[1], p2[1], num_points, dtype = np.int32)):
                floor_map[x,y] += 1
    
    map_points(hor_vert_vent_coords)
    intersections1 = np.sum(floor_map > 1)
    
    
    diag_vent_coords = [x for x in vent_coords if x[0][0] != x[1][0] and x[0][1] != x[1][1]]
    
    map_points(diag_vent_coords)
    intersections2 = np.sum(floor_map > 1)
    
    if visualize:
        plt.imshow(floor_map)
        plt.xticks([])
        plt.yticks([])
    
    
    return intersections1, intersections2

#%%
# Day 6: Lanternfish

@time_this_func
def day6():
    with open("input6.txt") as f:
        fish_orig = [int(x) for x in f.read().strip().split(",")]
    
    fish = [fish_orig.count(x) for x in range(9)]
    for d in range(256):
        spawns = fish.pop(0)
        fish.append(spawns)
        fish[6] += spawns
        
        if d == 79:
            num_fish_80 = sum(fish)
    
    num_fish_256 = sum(fish)
    
    return num_fish_80, num_fish_256

#%%
# Day 7: The Treachery of Whales

@time_this_func
def day7():
    with open("input7.txt") as f:
        crabs = [int(x) for x in f.read().strip().split(",")]
        
    min_hor = min(crabs)
    max_hor = max(crabs)
    
    calc_fuel2 = lambda diff: (1 + diff)*(diff/2)
    
    min_fuel1 = np.inf
    min_fuel2 = np.inf
    for h in range(min_hor, max_hor + 1):
        fuel1 = sum([abs(x-h) for x in crabs])
        if fuel1 < min_fuel1:
            min_fuel1 = fuel1
            
        fuel2 = sum([calc_fuel2(abs(x-h)) for x in crabs])
        if fuel2 < min_fuel2:
            min_fuel2 = fuel2
    
    min_fuel2 = int(min_fuel2)
    
    return min_fuel1, min_fuel2

#%%
# Day 8: Seven Segment Search

@time_this_func
def day8():
    data = []
    with open("input8.txt") as f:
        for l in f:
            first, second = l.split(" | ")
            data.append([first.split(), second.split()])
            
    output = sum([x[1] for x in data], [])
    
    num_uniques = sum([len(x) in {2,4,3,7} for x in output])
    
    
    def wire_to_digit(sequence, decoder):
        decoded = "".join(sorted({decoder[s] for s in sequence}))
        if decoded == "ABCEFG":
            return "0"
        elif decoded == "CF":
            return "1"
        elif decoded == "ACDEG":
            return "2"
        elif decoded == "ACDFG":
            return "3"
        elif decoded == "BCDF":
            return "4"
        elif decoded == "ABDFG":
            return "5"
        elif decoded == "ABDEFG":
            return "6"
        elif decoded == "ACF":
            return "7"
        elif decoded == "ABCDEFG":
            return "8"
        elif decoded == "ABCDFG":
            return "9"
        else:
            raise Exception(f"{decoded}")
        
    def decode(sequences, decoder):
        num = ""
        for sequence in sequences:
            num += wire_to_digit(sequence, decoder)
        return int(num)
    
    output_nums = []
    for data_entry in data:
        reading = list(set(["".join(sorted(x)) for x in sum(data_entry, [])]))
        reading = [set(x) for x in reading]
        for r in reading:
            parts = len(r)
            if parts == 2:
                CF = r
            elif parts == 4:
                BCDF = r
            elif parts == 3:
                ACF = r
            elif parts == 7:
                ABCDEFG = r
             
        connected = {}
        
        A = (ACF-CF).pop()
        connected[A] = "A"
        
        ACEFG = ACF | (ABCDEFG-BCDF-ACF)
        B = [(x-ACEFG).pop() for x in reading if len(x-ACEFG) == 1 and len(x) == 6][0]
        connected[B] = "B"
        
        D = (BCDF-CF-{B}).pop()
        connected[D] = "D"
        
        ABCDF = ACF | {B, D}
        G =  [(x-ABCDF).pop() for x in reading if len(x-ABCDF) == 1][0]
        connected[G] = "G"
        
        ABCDFG = ABCDF | {G}
        E = (ABCDEFG-ABCDFG).pop()
        connected[E] = "E"
        
        ABDG = {A, B, D, G}
        F =  [(x-ABDG).pop() for x in reading if len(x-ABDG) == 1][0]
        connected[F] = "F"
        
        C = (CF - {F}).pop()
        connected[C] = "C"
    
        output_nums.append(decode(data_entry[1], connected))
        
    output_sum = sum(output_nums)
    
    
    return num_uniques, output_sum

#%%
# Day 9: Smoke Basin

@time_this_func
def day9():
    floor = []
    with open("input9.txt") as f:
        for l in f:
            floor.append([int(x) for x in l.strip()])
    floor = np.array(floor)
    
    floor = np.pad(floor, 1, constant_values = 9)
    
    def is_lowest(i,j):
        curr = floor[(i,j)]
        if curr >= floor[i-1,j] or curr >= floor[i+1,j] or curr >= floor[i,j-1] or curr >= floor[i,j+1]:
            return False
        else:
            return True
    
    lowests = []
    lowest_locs = []
    for i in range(floor.shape[0]):
        for j in range(floor.shape[1]):
            if is_lowest(i,j):
                lowests.append(floor[i,j])
                lowest_locs.append((i,j))
                
    risk_sum = sum([x+1 for x in lowests])
    
    
    def measure_basin(lowest_loc):
        visited = {lowest_loc}
        paths = [[lowest_loc]]
        while True:
            new_paths = []
            for path in paths:
                i,j = path[-1]
                height = floor[(i,j)]
                next_locs = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
                for next_loc in next_locs:
                    if floor[next_loc] > height and floor[next_loc] != 9 and next_loc not in visited:
                        new_paths.append(path + [next_loc])
                        visited.add(next_loc)
                
            if len(new_paths) == 0:
                return len(visited)
                
            paths = new_paths
                
    basin_sizes = [measure_basin(x) for x in lowest_locs]
    
    basin_prod = prod(sorted(basin_sizes)[-3:])
    
    
    return risk_sum, basin_prod

#%%
# Day 10: Syntax Scoring

@time_this_func
def day10():
    lines = []
    with open("input10.txt") as f:
        for l in f:
            lines.append(l.strip())
    
    def check_line(line):
        openers = "[(<{"
        opens = []
        for c in line:
            if c in openers:
                opens.append(c)
            else:
                if openers.index(opens[-1]) != "])>}".index(c):
                    return ("Corrupted", c)
                else:
                    del opens[-1]
    
        return ("Incomplete", "".join(opens))
    
    illegals = ""
    still_opens = []
    for line in lines:
        result = check_line(line)
        if result[0] == "Corrupted":
            illegals += result[1]
        else:
            still_opens.append(result[1])
            
    points = {")": 3, "]": 57, "}": 1197, ">": 25137}
    
    total_corrupted_points = sum([points[x] for x in illegals])
    
    def score(still_open):
        to_complete = still_open[::-1]
        to_complete = to_complete.replace("[","]")
        to_complete = to_complete.replace("(",")")
        to_complete = to_complete.replace("<",">")
        to_complete = to_complete.replace("{","}")
        
        points = {")": 1, "]": 2, "}": 3, ">": 4}
        score = 0
        for c in to_complete:
            score *= 5
            score += points[c]
        return score
    
    scores = [score(x) for x in still_opens]
    
    median_score = int(np.median(scores))
    
    return total_corrupted_points, median_score

#%%
# Day 11: Dumbo Octopus

@time_this_func
def day11(visualize = False):
    octopuses = []
    with open("input11.txt") as f:
        for l in f:
            octopuses.append([int(x) for x in l.strip()])
    octopuses = np.array(octopuses)
    
    def flash_neighbors(i,j):
        for r in range(i-1,i+2):
            for c in range(j-1,j+2):
                if r not in range(octopuses.shape[0]) or c not in range(octopuses.shape[1]):
                    continue
                octopuses[r,c] += 1
    
    flashes_100 = 0
    s = 0
    if visualize:
        history = [octopuses.copy()]
        
    while True:
        s += 1
        octopuses += 1
        already_flashed = np.zeros(octopuses.shape)
        while True:
            changed = False
            for i in range(octopuses.shape[0]):
                for j in range(octopuses.shape[1]):
                    if octopuses[i,j] >= 10 and not already_flashed[i,j]:
                        flash_neighbors(i,j)
                        already_flashed[i,j] = True
                        changed = True
                        
                        if visualize:
                            history.append(octopuses.copy())
            if not changed:
                break
        
        if s <= 100:
            flashes_100 += np.sum(octopuses >= 10)
            
        if np.sum(octopuses >= 10) == prod(octopuses.shape):
            break
            
        octopuses[octopuses >= 10] = 0
        
    first_full_flash_step = s
        
    if visualize:
        fig, ax = plt.subplots()
        ims = []
        for state in history:
            im = ax.imshow(state, cmap = "gray", animated=True, vmin = 9, vmax = 10)
            ax.set_xticks([])
            ax.set_yticks([])
            ims.append([im])
        
        ani = animation.ArtistAnimation(fig, ims, interval=5, blit=True, repeat_delay=10)
     
    if not visualize:
        return flashes_100, first_full_flash_step
    else:
        return flashes_100, first_full_flash_step, ani
    
#%%
# Day 12: Passage Pathing

@time_this_func
def day12():
    linked = {}
    with open("input12.txt") as f:
        for l in f:
            new = l.strip().split("-")
            for x,y in [(0,1), (1,0)]:
                if new[x] == "end" or new[y] == "start":
                    continue
                
                if new[x] not in linked:
                    linked[new[x]] = {new[y]}
                else:
                    linked[new[x]].add(new[y])
                    
    num_paths_to_end1 = 0
    paths = [['start']]
    while True:
        new_paths = []
        for path in paths:
            latest = path[-1]
            next_caves = linked[latest]
            for next_cave in next_caves:
                if next_cave.isupper() or next_cave not in path:
                    new_path = path + [next_cave]
                    if next_cave == "end":
                        num_paths_to_end1 += 1
                    else:
                        new_paths.append(new_path)
        
        if len(new_paths) == 0:
            break
        
        paths = new_paths
    
    
    num_paths_to_end2 = 0
    last_locs = ['start']
    num_visits = [{}]
    while True:
        new_last_locs = []
        new_num_visits = []
        for last_loc, num_visit in zip(last_locs, num_visits):
            next_caves = linked[last_loc]
            for next_cave in next_caves:
                new_num_visit = num_visit.copy()
                if next_cave.islower():
                    if next_cave in new_num_visit:
                        new_num_visit[next_cave] += 1
                    else:
                        new_num_visit[next_cave] = 1
                        
                    if sum([x > 1 for x in new_num_visit.values()]) > 1 or \
                        True in {x > 2 for x in new_num_visit.values()}:
                        continue
                            
                if next_cave == "end":
                    num_paths_to_end2 += 1
                else:
                    new_last_locs.append(next_cave)
                    new_num_visits.append(new_num_visit)
        
        if len(new_last_locs) == 0:
            break
        
        last_locs = new_last_locs
        num_visits = new_num_visits
    
    
    return num_paths_to_end1, num_paths_to_end2
    
#%%
# Day 13: Transparent Origami

@time_this_func
def day13(visualize = False):
    with open("input13.txt") as f:
        raw = f.read().strip()
        
    dot_coords_raw, fold_instructions_raw = raw.split("\n\n")
    
    dot_coords = [[int(x) for x in dot_coord_raw.split(",")] for dot_coord_raw in dot_coords_raw.split()]
    
    folds = []
    for fold_instruction in fold_instructions_raw.split("\n"):
        fold = fold_instruction.split()[-1].split("=")
        folds.append([fold[0], int(fold[1])])
        
    max_x = max([x[0] for x in dot_coords])
    max_y = max([x[1] for x in dot_coords])
    
    dots = np.zeros([max_y+1,max_x+1], dtype = bool)
    
    for dot_coord in dot_coords:
        dots[tuple(dot_coord)[::-1]] = True
    
    if visualize:
        history = [dots.copy()]
    
    for f, fold in enumerate(folds):
        fold_line = fold[1]
        if fold[0] == "x":
            to_fold_left = np.fliplr(dots[:,fold_line+1:])
            folded_onto = dots[:,:fold_line]
            to_add = np.zeros([folded_onto.shape[0], abs(to_fold_left.shape[1]-folded_onto.shape[1])], dtype = bool)
            if folded_onto.shape[1] < to_fold_left.shape[1]:
                folded_onto = np.hstack([to_add, folded_onto])
            elif to_fold_left.shape[1] < folded_onto.shape[1]:
                to_fold_left = np.hstack([to_add, to_fold_left])
                                      
            dots = to_fold_left + folded_onto
        else:
            to_fold_up = np.flipud(dots[fold_line+1:,:])
            folded_onto = dots[:fold_line,:]
            to_add = np.zeros([abs(to_fold_left.shape[0]-folded_onto.shape[0]), folded_onto.shape[1]], dtype = bool)
            if folded_onto.shape[0] < to_fold_up.shape[0]:
                folded_onto = np.vstack([to_add, folded_onto])
            elif to_fold_up.shape[0] < folded_onto.shape[0]:
                to_fold_up = np.vstack([to_add, to_fold_up])
                
            dots = to_fold_up + folded_onto
            
        if visualize:
            history.append(dots.copy())
            
        if f == 0:
            dots_after_first_fold = np.sum(dots)
    
    if not visualize:
        plt.imshow(dots)
        plt.xticks([])
        plt.yticks([])
        
        return dots_after_first_fold
    else:
        fig, ax = plt.subplots()
        def animate(state):
            im = ax.imshow(state)
            ax.set_xticks([])
            ax.set_yticks([])
            return im
        
        ani = animation.FuncAnimation(fig, animate, frames = history, interval=1000, repeat_delay=2000)
        
        return dots_after_first_fold, ani
    
#%%
# Day 14: Extended Polymerization

@time_this_func
def day14():
    with open("input14.txt") as f:
        raw = f.read().strip()
        
    polymer, reactions_raw = raw.split("\n\n")
    
    reactions = {}
    for reaction_raw in reactions_raw.split("\n"):
        start_pair, inserted = reaction_raw.split(" -> ")
        reactions[start_pair] = inserted
    
    conversions = {k: [k[0]+v, v+k[1]] for k,v in reactions.items()}
    
    counts = dict(zip(reactions,[0]*len(reactions)))
    for pair in [polymer[i:i+2] for i in range(len(polymer)-1)]:
        counts[pair] += 1
    
    letter_counts = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",[0]*26))
    for letter in polymer:
        letter_counts[letter] += 1
    
    for s in range(40):  
        new_counts = dict(zip(reactions,[0]*len(reactions)))
        for pair in counts:
            for made_pair in conversions[pair]:
                new_counts[made_pair] += counts[pair]
            letter_counts[reactions[pair]] += counts[pair]
        
        counts = new_counts
                
        if s == 10-1:
            non_zero_letter_counts = [v for v in letter_counts.values() if v > 0]
            count_range10 = max(non_zero_letter_counts) - min(non_zero_letter_counts)
            
    non_zero_letter_counts = [v for v in letter_counts.values() if v > 0]
    count_range40 = max(non_zero_letter_counts) - min(non_zero_letter_counts)
    
    return count_range10, count_range40

#%%
# Day 15: Chiton

@time_this_func
def day15():
    print("645 s runtime (ouch) ...")
    
    chitons = []
    with open("input15.txt") as f:
        for l in f:
            chitons.append([int(x) for x in l.strip()])
    chitons = np.array(chitons)
        
    chitons_5x = np.hstack([chitons, chitons + 1, chitons + 2, chitons + 3, chitons + 4])
    chitons_5x[chitons_5x == 10] = 1
    chitons_5x[chitons_5x == 11] = 2
    chitons_5x[chitons_5x == 12] = 3
    chitons_5x[chitons_5x == 13] = 4
    chitons_5x = np.vstack([chitons_5x, chitons_5x + 1, chitons_5x + 2, chitons_5x + 3, chitons_5x + 4])
    chitons_5x[chitons_5x == 10] = 1
    chitons_5x[chitons_5x == 11] = 2
    chitons_5x[chitons_5x == 12] = 3
    chitons_5x[chitons_5x == 13] = 4
    chitons_5x = chitons_5x.astype(int)
    
    def least_risky_to_bottom_right(chitons_map):
        valid_locs = set(product(range(chitons_map.shape[0]), repeat = 2))
        
        lowest_risk_to = {(0,0): 0}
        
        locs = [(0,0)]
        risks = [0]
        end_loc = (chitons_map.shape[0]-1, chitons_map.shape[1]-1)
        while True:
            new_locs = []
            new_risks = []
            for loc, risk in zip(locs, risks):
                i,j = loc
                next_locs = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
                for next_loc in next_locs:            
                    if next_loc not in valid_locs:
                        continue
                    
                    risk_to_next_loc = risk + chitons_map[next_loc]
                    
                    if next_loc in lowest_risk_to and risk_to_next_loc >= lowest_risk_to[next_loc]:
                        continue

                    lowest_risk_to[next_loc] = risk_to_next_loc
                    
                    if next_loc != end_loc:
                        new_locs.append(next_loc)
                        new_risks.append(risk_to_next_loc)
            
            if len(new_locs) == 0:
                break
            
            locs = new_locs
            risks = new_risks    
        
        least_risk = lowest_risk_to[(chitons_map.shape[0]-1, chitons_map.shape[1]-1)]
        return least_risk
    
    least_risky1 = least_risky_to_bottom_right(chitons)
    least_risky2 = least_risky_to_bottom_right(chitons_5x)

    return least_risky1, least_risky2

#%%
# Day 16: Packet Decoder

@time_this_func
def day16():
    with open("input16.txt") as f:
        packet = f.read().strip()
        
    def hex_to_bit(hex_packet):
        from_hex = {"0": "0000",
                    "1": "0001",
                    "2": "0010",
                    "3": "0011",
                    "4": "0100",
                    "5": "0101",
                    "6": "0110",
                    "7": "0111",
                    "8": "1000",
                    "9": "1001",
                    "A": "1010",
                    "B": "1011",
                    "C": "1100",
                    "D": "1101",
                    "E": "1110",
                    "F": "1111"}
        
        return "".join([from_hex[x] for x in hex_packet])
        
    bin_packet = hex_to_bit(packet)
    
    versions = []
    
    def analyze(i):
        version = int(bin_packet[i:i+3],2)
        versions.append(version)
        i += 3
        type_ID = int(bin_packet[i:i+3],2)
        i += 3
        
        if type_ID != 4:
            length_type_id = bin_packet[i]
            i += 1 
            
            op_nums = []
            if length_type_id == "0":
                num_subpacket_bits = int(bin_packet[i:i+15],2)
                i += 15
                
                from_i = i
                while i < from_i + num_subpacket_bits:
                    i, op_num = analyze(i)
                    op_nums.append(op_num)
    
            else: 
                num_subpackets = int(bin_packet[i:i+11],2)
                i += 11
                
                for _ in range(num_subpackets):
                    i, op_num = analyze(i)
                    op_nums.append(op_num)
            
            if type_ID == 0:
                num = sum(op_nums)
            elif type_ID == 1:
                num = prod(op_nums)
            elif type_ID == 2:
                num = min(op_nums)
            elif type_ID == 3:
                num = max(op_nums)
            elif type_ID == 5:
                num = int(op_nums[0] > op_nums[1])
            elif type_ID == 6:
                num = int(op_nums[0] < op_nums[1])
            elif type_ID == 7:
                num = int(op_nums[0] == op_nums[1])            
        
        else:
            bin_num = bin_packet[i:i+5]
            i += 5
            while bin_num[-5] != "0":
                bin_num += bin_packet[i:i+5]
                i += 5
            num = int("".join([bin_num[j+1:j+5] for j in range(0,len(bin_num),5)]),2)
            
        return i, num
        
    _, final_num = analyze(0)
        
    versions_sum = sum(versions)
    
    return versions_sum, final_num
        
#%%
# Day 17: Trick Shot

@time_this_func
def day17():
    with open("input17.txt") as f:
        raw = f.read().strip().split()
    
    x_target = [int(x) for x in raw[2][2:-1].split("..")]
    y_target = [int(y) for y in raw[3][2:].split("..")]
    target = set()
    for x in range(x_target[0], x_target[1]+1):
        for y in range(y_target[0], y_target[1]+1):
            target.add((x,y))
    
    max_x = x_target[1]
    min_y = y_target[0]
    
    def shoot(v_x, v_y):
        position = np.array([0,0])
        velocity = np.array([v_x,v_y])
        
        y_history = [position[1]]
        while position[1] >= min_y:
            position += velocity
            velocity[0] -= 1*np.sign(velocity[0])
            velocity[1] -= 1
    
            y_history.append(position[1])
            
            if tuple(position) in target:
                return max(y_history)
        
        return "miss"
    
    highest_y = 0
    valid_velocities = 0
    for vy in range(min_y, -min_y):
        for vx in range(max_x+1):
            result = shoot(vx, vy)
            
            if type(result) != str:
                if result > highest_y:
                    highest_y = result
                valid_velocities += 1
                
    return highest_y, valid_velocities

#%%
# Day 18: Snailfish

@time_this_func
def day18():
    snail_nums_orig = []
    with open("input18.txt") as f:
        for l in f:
            snail_nums_orig.append(l.strip())
    
    def explode(added):
        explodable = False
        depth = 0
        for i, c in enumerate(added):
            if c == "[":
                depth += 1
                if depth >= 5:
                    explodable = True
                    break
            elif c == "]":
                depth -= 1
                
        if explodable:
            while added[i] != "]":
                if added[i] == "[":
                    to_explode_from = i
                i += 1
            to_explode_to = i
            to_explode = eval(added[to_explode_from:to_explode_to+1])
            
            regular_to_left = None
            i = to_explode_from
            while i in range(len(added)):
                if added[i].isnumeric():
                    regular_to_left = [i]
                    while added[i-1].isnumeric():
                        i -= 1
                    regular_to_left = [i] + regular_to_left
                    
                    regular_to_left_num = int(added[regular_to_left[0]:regular_to_left[1]+1])
                    
                    break
                i -= 1
                
            regular_to_right = None
            i = to_explode_to
            while i in range(len(added)):
                if added[i].isnumeric():
                    regular_to_right = [i]
                    while added[i+1].isnumeric():
                        i += 1
                    regular_to_right.append(i)
                    
                    regular_to_right_num = int(added[regular_to_right[0]:regular_to_right[1]+1])
                    
                    break
                i += 1
            
            new_added = ""
            if regular_to_left == None:
                new_added += added[:to_explode_from]
            else:
                new_added += added[:regular_to_left[0]] + str(regular_to_left_num + to_explode[0]) + added[regular_to_left[1]+1:to_explode_from]
                
            new_added += "0"
            
            if regular_to_right == None:
                new_added += added[to_explode_to+1:]
            else:
                new_added += added[to_explode_to+1:regular_to_right[0]] + str(regular_to_right_num + to_explode[1]) + added[regular_to_right[1]+1:]
                
            added = new_added
        
        return added
        
    def split(added):
        splittable = False
        num = ["",[]]
        for i, c in enumerate(added):
            if c.isnumeric():
                num[0] += c
                num[1].append(i)
            else:
                if num[0] != "":
                    int_num = int(num[0])
                    if int_num >= 10:
                        splittable = True
                        to_split_from = num[1][0]
                        to_split_to = num[1][-1]
                        break
                    num = ["",[]]
        
        if splittable:
            added = added[:to_split_from] + "[" + str(int_num//2) + "," + str(int(int_num/2+0.5)) + "]" + added[to_split_to+1:]
            
        return added
    
    def reduce(added):
        last = ""
        while last != added:
            last = added
            added = explode(added)
            if last != added:
                continue
            added = split(added)
            
        return added
    
    def magnitude(snail_num):
        if type(snail_num) == list:
            mag0 = 3*magnitude(snail_num[0])
            mag1 = 2*magnitude(snail_num[1])
            return mag0 + mag1       
        else:
            return snail_num
    
    snail_nums = snail_nums_orig.copy()
    
    running_sum = snail_nums.pop(0)
    while len(snail_nums) != 0:
        snail_num1 = running_sum
        snail_num2 = snail_nums.pop(0)
        running_sum = reduce("[" + snail_num1 + "," + snail_num2 + "]")
        
    final_magnitude = magnitude(eval(running_sum))
    
    
    max_magnitude_from_two = 0
    for s1 in range(len(snail_nums_orig)):
        for s2 in range(len(snail_nums_orig)):
            if s1 == s2:
                continue
            
            snail_num_sum = reduce("[" + snail_nums_orig[s1] + "," + snail_nums_orig[s2] + "]")
            s1s2_magnitude = magnitude(eval(snail_num_sum))
            
            if s1s2_magnitude > max_magnitude_from_two:
                max_magnitude_from_two = s1s2_magnitude
                
    
    return final_magnitude, max_magnitude_from_two

#%%
# Day 19: Beacon Scanner

@time_this_func
def day19():
    with open("input19.txt") as f:
        scanners_raw = f.read().strip().split("\n\n")
        
    scanners_beacons = [[eval(x) for x in scanner.split("\n")[1:]] for scanner in scanners_raw]
    
    def from_perspective(base_p, p2):
        return tuple([p2[i]-base_p[i] for i in range(3)])
     
    def get_dist(p1, p2):
        return sum([abs(p1[i]-p2[i]) for i in range(3)])
    
    def get_rotation(points, version):
        mat1 = [[1,0,0],[0,1,0],[0,0,1]]
        mat2 = [[0,0,1],[0,1,0],[-1,0,0]]
        mat3 = [[-1,0,0],[0,1,0],[0,0,-1]]
        mat4 = [[0,0,-1],[0,1,0],[1,0,0]]
        mat5 = [[0,-1,0],[1,0,0],[0,0,1]]
        mat6 = [[0,0,1],[1,0,0],[0,1,0]]
        mat7 = [[0,1,0],[1,0,0],[0,0,-1]]
        mat8 = [[0,0,-1],[1,0,0],[0,-1,0]]
        mat9 = [[0,1,0],[-1,0,0],[0,0,1]]
        mat10 = [[0,0,1],[-1,0,0],[0,-1,0]]
        mat11 = [[0,-1,0],[-1,0,0],[0,0,-1]]
        mat12 = [[0,0,-1],[-1,0,0],[0,1,0]]
        mat13 = [[1,0,0],[0,0,-1],[0,1,0]]
        mat14 = [[0,1,0],[0,0,-1],[-1,0,0]]
        mat15 = [[-1,0,0],[0,0,-1],[0,-1,0]]
        mat16 = [[0,-1,0],[0,0,-1],[1,0,0]]
        mat17 = [[1,0,0],[0,-1,0],[0,0,-1]]
        mat18 = [[0,0,-1],[0,-1,0],[-1,0,0]]
        mat19 = [[-1,0,0],[0,-1,0],[0,0,1]]
        mat20 = [[0,0,1],[0,-1,0],[1,0,0]]
        mat21 = [[1,0,0],[0,0,1],[0,-1,0]]
        mat22 = [[0,-1,0],[0,0,1],[-1,0,0]]
        mat23 = [[-1,0,0],[0,0,1],[0,1,0]]
        mat24 = [[0,1,0],[0,0,1],[1,0,0]]
        
        mats = [mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8, mat9, mat10, mat11, mat12,
                mat13, mat14, mat15, mat16, mat17, mat18, mat19, mat20, mat21, mat22, mat23, mat24]
        
        ready_mats = [np.array(x) for x in mats]
        
        if version == "fingerprints":
            ready_fingerprints = [[np.array(b).reshape(3,1) for b in beacon] for beacon in points]
            
            for ready_mat in ready_mats:
                new_fingerprints = []
                for beacon in ready_fingerprints:
                    new_beacon_fingerprint = []
                    for ready_fingerprint in beacon:
                        new_beacon_fingerprint.append(tuple((ready_mat@ready_fingerprint).ravel()))
                    new_fingerprints.append(tuple(new_beacon_fingerprint))
                    
                yield new_fingerprints
        
        else:
            ready_points = [np.array(p).reshape(3,1) for p in points]
            
            for ready_mat in ready_mats:
                new_points = []
                for ready_point in ready_points:
                    new_points.append(tuple((ready_mat@ready_point).ravel()))
                
                yield new_points
        
    def get_fingerprints(beacons, num_neighbors = 1):   #1 is enough. Also works with 2.
        fingerprints = []
        for base_beacon in beacons:
            fingerprint = []
            for other_beacon in beacons:
                if base_beacon == other_beacon:
                    continue
                fingerprint.append(from_perspective(base_beacon, other_beacon))
            
            fingerprint = tuple(sorted(fingerprint, key = lambda x: get_dist((0,0,0), x))[:num_neighbors])
            fingerprints.append(fingerprint)
        
        return fingerprints
    
    def shift_point(point_to_shift, shift):
        new_point = [point_to_shift[i]+shift[i] for i in range(3)]
        return tuple(new_point)
    
    scanners_fingerprints = list()
    for beacons in scanners_beacons:
        scanners_fingerprints.append(get_fingerprints(beacons))
        
    joined_beacons = scanners_beacons[0]
    joined_fingerprints = scanners_fingerprints[0]
    joined_scanner_inds = {0}
    joined_scanners = [(0,0,0)]
    while len(joined_scanner_inds) < len(scanners_fingerprints):
        for scanner_i, fingerprints in enumerate(scanners_fingerprints):
            if scanner_i in joined_scanner_inds:
                continue
            
            overlap = False
            for rotated_fingerprints, rotated_beacons in zip(get_rotation(fingerprints, "fingerprints"), 
                                                            get_rotation(scanners_beacons[scanner_i], "points")):
                in_common = []
                not_in_common = []
                for beacon_i, fingerprint in enumerate(rotated_fingerprints):
                    if fingerprint in joined_fingerprints:
                        in_common.append(beacon_i)
                    else:
                        not_in_common.append(beacon_i)
                        
                if len(in_common) >= 12:
                    overlap = True
                    break
            
            if overlap:
                joined_fingerprints += [rotated_fingerprints[x] for x in not_in_common]
                joined_scanner_inds.add(scanner_i)
                
                fingerprint_in_common = rotated_fingerprints[in_common[0]]
                for beacon_i, joined_fingerprint in enumerate(joined_fingerprints):
                    if fingerprint_in_common == joined_fingerprint:
                        offset = [joined_beacons[beacon_i][i]-rotated_beacons[in_common[0]][i] for i in range(3)]
                        break

                joined_beacons += [shift_point(rotated_beacons[x], offset) for x in not_in_common]
    
                joined_scanners.append(tuple(offset))
    
    num_beacons = len(joined_fingerprints)
    
    max_dist = 0
    for scanner1 in joined_scanners:
        for scanner2 in joined_scanners:
            if scanner1 == scanner2:
                continue
            
            dist = get_dist(scanner1, scanner2)
            if dist > max_dist:
                max_dist = dist
                
    return num_beacons, max_dist

#%%
# Day 20: Trench Map

@time_this_func
def day20():
    with open("input20.txt") as f:
        raw = f.read().strip().split("\n\n")
        
    enhancement = raw[0]
        
    pixels_raw = raw[1].split()
    light_pixels = set()
    for i, row in enumerate(pixels_raw):
        for j, c in enumerate(row):
            if c == "#":
                light_pixels.add((i,j))
                
    def bounding_box():
        xs = [x[0] for x in light_pixels]
        ys = [x[1] for x in light_pixels]
        
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        
        return min_x, max_x, min_y, max_y
    
    for step in range(50):
        if step%2 == 0:
            outer_pixels = "0"
        else:
            outer_pixels = "1"
            
        boundaries = bounding_box()
        
        bound_xs = set(range(boundaries[0], boundaries[1]+1))
        bound_ys = set(range(boundaries[2], boundaries[3]+1))
        
        new_light_pixels = set()
        for i in range(boundaries[0]-1,boundaries[1]+2):
            for j in range(boundaries[2]-1,boundaries[3]+2):
                signature = ""
                for x in range(i-1,i+2):
                    for y in range(j-1,j+2):
                        if x not in bound_xs or y not in bound_ys:
                            signature += outer_pixels
                        else:
                            if (x,y) in light_pixels:
                                signature += "1"
                            else:
                                signature += "0"
                enhancement_ind = int(signature,2)
                if enhancement[enhancement_ind] == "#":
                    new_light_pixels.add((i,j))
        light_pixels = new_light_pixels
        
        if step == 1:
            num_light_pixels2 = len(light_pixels)
    
    num_light_pixels50 = len(light_pixels)
    
    return num_light_pixels2, num_light_pixels50

#%%
# Day 21: Dirac Dice

@time_this_func
def day21():
    with open("input21.txt") as f:
        player_locs_orig = [int(x.split()[-1]) for x in f.read().strip().split("\n")]
    
    def deterministic_dice():
        while True:
            for d in range(1,101):
                yield d
                
    player_locs = player_locs_orig.copy()
    player_scores = [0,0]
    dice_roll = deterministic_dice()
    player = -1
    num_rolls = 0
    while max(player_scores) < 1000:
        player = (player + 1)%2
        
        player_locs[player] += next(dice_roll) + next(dice_roll) + next(dice_roll)
        num_rolls += 3
        while player_locs[player] > 10:
            player_locs[player] -= 10
            
        player_scores[player] += player_locs[player]
    
    loser_score_mult_dice_rolls = min(player_scores)*num_rolls
    
    
    possible_rolls = [sum(x) for x in product([1,2,3], repeat = 3)]
    possible_rolls_counts = {x:possible_rolls.count(x) for x in set(possible_rolls)}
                
    state_leads_to_wins = {}
    def play(player_scores, player_locs, player):
        if (player_scores, player_locs, player) in state_leads_to_wins:       
            return state_leads_to_wins[(player_scores, player_locs, player)]
    
        else:
            leads_to_wins = np.array([0,0], dtype = np.int64)
            new_player = (player + 1)%2
            
            for roll, count in possible_rolls_counts.items():
                player_loc = player_locs[player] + roll
                if player_loc > 10:
                    player_loc %= 10
                
                player_score = player_scores[player] + player_loc
                if player_score >= 21:
                    leads_to_wins[player] += count
                else:
                    new_player_scores = list(player_scores)
                    new_player_scores[player] = player_score
                    new_player_scores = tuple(new_player_scores)
                    
                    new_player_locs = list(player_locs)
                    new_player_locs[player] = player_loc
                    new_player_locs = tuple(new_player_locs)
                    
                    leads_to_wins += play(new_player_scores, new_player_locs, new_player)*count
                    
            state_leads_to_wins[(player_scores, player_locs, player)] = leads_to_wins
    
            return leads_to_wins
                
    total_wins = play((0,0), tuple(player_locs_orig), 0)
    
    most_universe_wins = max(total_wins)
    
    
    return loser_score_mult_dice_rolls, most_universe_wins

#%%
# Day 22: Reactor Reboot

@time_this_func
def day22():
    print("1,322 s runtime (ouch) ...")
    
    steps = []
    with open("input22.txt") as f:
        for l in f:
            on_or_off, region = l.strip().split()
            region_x, region_y, region_z = region.split(",")
            region_x = sorted([int(x) for x in region_x[2:].split("..")])
            region_y = sorted([int(y) for y in region_y[2:].split("..")])
            region_z = sorted([int(z) for z in region_z[2:].split("..")])
            steps.append((on_or_off, (region_x, region_y, region_z)))
    
    class cube():
        def __init__(self, x_region, y_region, z_region):
            self.x_region = tuple(x_region)
            self.y_region = tuple(y_region)
            self.z_region = tuple(z_region)
            
        def num_points(self):
            return (self.x_region[1]+1-self.x_region[0]) * (self.y_region[1]+1-self.y_region[0]) * (self.z_region[1]+1-self.z_region[0])
            
        def in_50_50_range(self):
            return self.x_region[0] >= -50 and self.x_region[1] <= 50 and \
                    self.y_region[0] >= -50 and self.y_region[1] <= 50 and \
                    self.z_region[0] >= -50 and self.z_region[1] <= 50
        
        def __hash__(self):
            return hash((self.x_region, self.y_region, self.z_region))
        
        def __repr__(self):
            return str(f"x:{self.x_region}, y:{self.y_region}, z:{self.z_region}")
        
        def __str__(self):
            return self.__repr__()
        
        def intersect(self, other_cube):
            if self.x_region[1] < other_cube.x_region[0] or self.x_region[0] > other_cube.x_region[1] or \
                self.y_region[1] < other_cube.y_region[0] or self.y_region[0] > other_cube.y_region[1] or \
                self.z_region[1] < other_cube.z_region[0] or self.z_region[0] > other_cube.z_region[1]:
                return ("No Intersection", [])
            
            if self.x_region[0] >= other_cube.x_region[0] and self.x_region[1] <= other_cube.x_region[1] and \
                self.y_region[0] >= other_cube.y_region[0] and self.y_region[1] <= other_cube.y_region[1] and \
                self.z_region[0] >= other_cube.z_region[0] and self.z_region[1] <= other_cube.z_region[1]:
                return ("Encompassed", [])
            
            if other_cube.x_region[0] < self.x_region[0] and \
                other_cube.x_region[1] >= self.x_region[0] and \
                other_cube.x_region[1] <= self.x_region[1]:
                x_intersections = [None, other_cube.x_region[1]]
            elif other_cube.x_region[0] >= self.x_region[0] and other_cube.x_region[0] <= self.x_region[1] and \
                other_cube.x_region[1] >= self.x_region[0] and other_cube.x_region[1] <= self.x_region[1]:
                x_intersections = other_cube.x_region
            elif other_cube.x_region[0] >= self.x_region[0] and \
                other_cube.x_region[0] <= self.x_region[1] and \
                other_cube.x_region[1] > self.x_region[1]:
                x_intersections = [other_cube.x_region[0], None]
            else:
                x_intersections = set()
                                
            if other_cube.y_region[0] < self.y_region[0] and \
                other_cube.y_region[1] >= self.y_region[0] and \
                other_cube.y_region[1] <= self.y_region[1]:
                y_intersections = [None, other_cube.y_region[1]]
            elif other_cube.y_region[0] >= self.y_region[0] and other_cube.y_region[0] <= self.y_region[1] and \
                other_cube.y_region[1] >= self.y_region[0] and other_cube.y_region[1] <= self.y_region[1]:
                y_intersections = other_cube.y_region
            elif other_cube.y_region[0] >= self.y_region[0] and \
                other_cube.y_region[0] <= self.y_region[1] and \
                other_cube.y_region[1] > self.y_region[1]:
                y_intersections = [other_cube.y_region[0], None]
            else:
                y_intersections = set()
                
            if other_cube.z_region[0] < self.z_region[0] and \
                other_cube.z_region[1] >= self.z_region[0] and \
                other_cube.z_region[1] <= self.z_region[1]:
                z_intersections = [None, other_cube.z_region[1]]
            elif other_cube.z_region[0] >= self.z_region[0] and other_cube.z_region[0] <= self.z_region[1] and \
                other_cube.z_region[1] >= self.z_region[0] and other_cube.z_region[1] <= self.z_region[1]:
                z_intersections = other_cube.z_region
            elif other_cube.z_region[0] >= self.z_region[0] and \
                other_cube.z_region[0] <= self.z_region[1] and \
                other_cube.z_region[1] > self.z_region[1]:
                z_intersections = [other_cube.z_region[0], None]
            else:
                z_intersections = set()
                
            intersections = [x_intersections, y_intersections, z_intersections]
    
            return ("Intersected", intersections)
        
        def split(self, split_lines):
            remaining_cube = self
            new_cubes = set()
            for dim, split_nums in enumerate(split_lines):
                if len(split_nums) == 0:
                    continue
                    
                elif None in split_nums:
                    if split_nums[0] == None:
                        split_num = split_nums[1]
                        if dim == 0:
                            new_cubes.add(cube((split_num+1,remaining_cube.x_region[1]),remaining_cube.y_region,remaining_cube.z_region))
                            remaining_cube = cube((remaining_cube.x_region[0],split_num),remaining_cube.y_region,remaining_cube.z_region)
                        elif dim == 1:
                            new_cubes.add(cube(remaining_cube.x_region,(split_num+1,remaining_cube.y_region[1]),remaining_cube.z_region))
                            remaining_cube = cube(remaining_cube.x_region,(remaining_cube.y_region[0],split_num),remaining_cube.z_region)
                        else:
                            new_cubes.add(cube(remaining_cube.x_region,remaining_cube.y_region,(split_num+1,remaining_cube.z_region[1])))
                            remaining_cube = cube(remaining_cube.x_region,remaining_cube.y_region,(remaining_cube.z_region[0],split_num))
    
                    else:
                        split_num = split_nums[0]
                        if dim == 0:
                            new_cubes.add(cube((remaining_cube.x_region[0],split_num-1),remaining_cube.y_region,remaining_cube.z_region))
                            remaining_cube = cube((split_num,remaining_cube.x_region[1]),remaining_cube.y_region,remaining_cube.z_region)
                        elif dim == 1:
                            new_cubes.add(cube(remaining_cube.x_region,(remaining_cube.y_region[0],split_num-1),remaining_cube.z_region))
                            remaining_cube = cube(remaining_cube.x_region,(split_num,remaining_cube.y_region[1]),remaining_cube.z_region)
                        else:
                            new_cubes.add(cube(remaining_cube.x_region,remaining_cube.y_region,(remaining_cube.z_region[0],split_num-1)))
                            remaining_cube = cube(remaining_cube.x_region,remaining_cube.y_region,(split_num,remaining_cube.z_region[1]))
                        
                else:
                    if dim == 0:
                        new_cubes.add(cube((remaining_cube.x_region[0],split_nums[0]-1),remaining_cube.y_region,remaining_cube.z_region))
                        new_cubes.add(cube((split_nums[1]+1,remaining_cube.x_region[1]),remaining_cube.y_region,remaining_cube.z_region))
                        remaining_cube = cube((split_nums[0],split_nums[1]),remaining_cube.y_region,remaining_cube.z_region)
                    elif dim == 1:
                        new_cubes.add(cube(remaining_cube.x_region,(remaining_cube.y_region[0],split_nums[0]-1),remaining_cube.z_region))
                        new_cubes.add(cube(remaining_cube.x_region,(split_nums[1]+1,remaining_cube.y_region[1]),remaining_cube.z_region))
                        remaining_cube = cube(remaining_cube.x_region,(split_nums[0],split_nums[1]),remaining_cube.z_region)
                    else:
                        new_cubes.add(cube(remaining_cube.x_region,remaining_cube.y_region,(remaining_cube.z_region[0],split_nums[0]-1)))
                        new_cubes.add(cube(remaining_cube.x_region,remaining_cube.y_region,(split_nums[1]+1,remaining_cube.z_region[1])))
                        remaining_cube = cube(remaining_cube.x_region,remaining_cube.y_region,(split_nums[0],split_nums[1]))
    
            return new_cubes
    
    def get_not_intersecting(basis_cube, comparison_cube):
        split_info = basis_cube.intersect(comparison_cube)
            
        if split_info[0] == "No Intersection":
            return {basis_cube}
        
        if split_info[0] == "Encompassed":
            return set()
        
        no_intersection_cubes = basis_cube.split(split_info[1])
        return no_intersection_cubes
    
    def plot_cube(c, color = "k", axis = None):
        cx = list(c.x_region)
        cy = list(c.y_region)
        cz = list(c.z_region)
        
        if axis == None:
            ax = plt.axes(projection ='3d')
        else:
            ax = axis
            
        ax.plot3D([cx[0]]*5, [cy[0], cy[0], cy[1], cy[1], cy[0]],  [cz[0], cz[1], cz[1], cz[0], cz[0]], c = color)
        ax.plot3D([cx[1]]*5, [cy[0], cy[0], cy[1], cy[1], cy[0]],  [cz[0], cz[1], cz[1], cz[0], cz[0]], c = color)
        ax.plot3D([cx[0], cx[0], cx[1], cx[1], cx[0]], [cy[0]]*5,  [cz[0], cz[1], cz[1], cz[0], cz[0]], c = color)
        ax.plot3D([cx[0], cx[0], cx[1], cx[1], cx[0]], [cy[1]]*5,  [cz[0], cz[1], cz[1], cz[0], cz[0]], c = color)
    
    cubes = [cube(*x[1]) for x in steps]
    
    first_on_cube = [i for i, x in enumerate(steps) if x[0] == "on"][0]
    on_cubes = {cubes[first_on_cube]}
    for step, next_cube in zip(steps[first_on_cube+1:],cubes[first_on_cube+1:]):   
        if step[0] == "on":
            cubes_to_add = {next_cube}
            for already_on_cube in on_cubes:
                new_cubes_to_add = set()
                for cube_to_add in cubes_to_add:
                    new_cubes_to_add = new_cubes_to_add | get_not_intersecting(cube_to_add, already_on_cube)
                cubes_to_add = new_cubes_to_add
                
            on_cubes = on_cubes | cubes_to_add
            
        elif step[0] == "off":
            new_on_cubes = set()
            for on_cube in on_cubes:
                new_on_cubes = new_on_cubes | get_not_intersecting(on_cube, next_cube)
            
            on_cubes = new_on_cubes
            
    small_num_on = sum([x.num_points() for x in on_cubes if x.in_50_50_range()])
            
    final_num_on = sum([x.num_points() for x in on_cubes])
    
    return small_num_on, final_num_on

#%%
# Day 23: Amphipod

@time_this_func
def day23(visualize = False):
    from copy import deepcopy
    
    with open("input23.txt") as f:
        rows = f.read().split("\n")
        row_len = len(rows[0])
        state = [[x for x in list(row)+[" "]*(row_len-len(row))] for row in rows[:-1]]
    part1_start_map = np.array(state)
    
    if visualize:
        def make_map_plottable(x):
            if x == " " or x == ".":
                return 0
            if x == "#":
                return 1
            if x == "A":
                return 3
            if x == "B":
                return 5
            if x == "C":
                return 7
            if x == "D":
                return 9
        
        num_map = np.vectorize(make_map_plottable)
        
        def layout_to_map(layout, empty_map):
            full_map = empty_map.copy()
            types = {0:"A", 1:"B", 2:"C", 3:"D"}
            for t, amphipod_group in enumerate(layout):
                for amphipod in amphipod_group:
                    full_map[amphipod] = types[t]
            return full_map
        
        def layout_to_plottable(layout, empty_map):
            full_map = layout_to_map(layout, empty_map)
            plottable_map = num_map(full_map)
            return plottable_map
            
    
    def solve(original_map):
        amphipods = [[],[],[],[]]
        spaces = set()
        
        for i in range(original_map.shape[0]):
            for j in range(original_map.shape[1]):
                if original_map[i,j] in {"A","B","C","D"}:
                    if original_map[i,j] == "A":
                        amphipods[0].append((i,j))
                    elif original_map[i,j] == "B":
                        amphipods[1].append((i,j))
                    elif original_map[i,j] == "C":
                        amphipods[2].append((i,j))
                    elif original_map[i,j] == "D":
                        amphipods[3].append((i,j))
                    spaces.add((i,j))
                elif original_map[i,j] == ".":
                    spaces.add((i,j))
        
        empty_map = original_map.copy()
        for i,j in sum(amphipods, []):
            empty_map[i,j] = "."
            
        def path_from_to(loc1, loc2):
            paths = [[loc1]]
            visited = {loc1}
            while True:
                new_paths = []
                for path in paths:
                    i,j = path[-1]
                    next_locs = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
                    for next_loc in next_locs:
                        if next_loc in visited or empty_map[next_loc] == "#":
                            continue
                        
                        if next_loc == loc2:
                            return path[1:] + [next_loc]
                        
                        new_paths.append(path + [next_loc])
                        visited.add(next_loc)
                paths = new_paths
                
        paths_from_to = {}
        for space1 in spaces:
            for space2 in spaces:
                if space1 == space2:
                    continue
                paths_from_to[(space1,space2)] = path_from_to(space1, space2)
                
        def can_go_to(loc1, loc2, amphipods):    
            path_spaces = set(paths_from_to[(loc1, loc2)])
            amphipod_locs = set(sum(amphipods, []))
            return (path_spaces.isdisjoint(amphipod_locs), len(path_spaces))
        
        def hash_state(amphipods):
            As = tuple(sorted(sorted(amphipods[0], key = lambda x: x[0]), key = lambda x: x[1]))
            Bs = tuple(sorted(sorted(amphipods[1], key = lambda x: x[0]), key = lambda x: x[1]))
            Cs = tuple(sorted(sorted(amphipods[2], key = lambda x: x[0]), key = lambda x: x[1]))
            Ds = tuple(sorted(sorted(amphipods[3], key = lambda x: x[0]), key = lambda x: x[1]))
            
            return hash((As, Bs, Cs, Ds))
        
        burrows = {0:[(x,3) for x in range(2,original_map.shape[0]-1)], \
                   1:[(x,5) for x in range(2,original_map.shape[0]-1)], \
                   2:[(x,7) for x in range(2,original_map.shape[0]-1)], \
                   3:[(x,9) for x in range(2,original_map.shape[0]-1)]}
        burrow_sets = {k:set(v) for k,v in burrows.items()}
        
        states = {hash_state(amphipods): 0}
        final_hash = hash_state([burrows[0],burrows[1],burrows[2],burrows[3]])
        states[final_hash] = np.inf
        
        unhash = {final_hash: [burrows[0],burrows[1],burrows[2],burrows[3]], \
                  hash_state(amphipods): amphipods}
        
        layouts = [amphipods]
        energies = [0]
        paths = [[hash_state(amphipods)]]
        full_paths = []
        while True:
            new_layouts = []
            new_energies = []
            new_paths = []
            for layout, energy, path in zip(layouts, energies, paths):
                
                if states[hash_state(layout)] < energy or energy > states[final_hash]:
                    continue
                
                for t, amphipod_group in enumerate(layout):
                    
                    if set(amphipod_group) == burrow_sets[t]:
                        continue
                    
                    for a, amphipod in enumerate(amphipod_group):
                        
                        if amphipod == burrows[t][1]:
                            continue
                        
                        other_amphipod_locs = set(sum([o for i,o in enumerate(layout) if i != t], []))
                        if other_amphipod_locs.isdisjoint(burrow_sets[t]):
                            can_go_spaces = burrow_sets[t]
                        else:
                            can_go_spaces = set()
                            
                        if amphipod[0] != 1:
                            can_go_spaces = can_go_spaces | {(1,1),(1,2),(1,4),(1,6),(1,8),(1,10),(1,11)}
                            
                        can_go_spaces = can_go_spaces - set(sum(layout,[]))
        
                        if burrows[t][1] in can_go_spaces:
                            can_go_spaces.discard(burrows[t][0])
                        
                        for space in can_go_spaces:                    
                            new_layout = deepcopy(layout)
                            new_layout[t][a] = space
                            new_hash = hash_state(new_layout)
                            can_it_go = can_go_to(amphipod, space, layout)
                            new_energy = energy + pow(10,t)*can_it_go[1]
                            
                            if can_it_go[0] and (new_hash not in states or new_energy < states[new_hash]):
                                states[new_hash] = new_energy
                                new_layouts.append(new_layout)
                                new_energies.append(new_energy)
                                if visualize:
                                    stops = paths_from_to[(amphipod, space)]
                                    stop_hashes = []
                                    for stop in stops:
                                        stop_layout = deepcopy(layout)
                                        stop_layout[t][a] = stop
                                        stop_hash = hash_state(stop_layout)
                                        unhash[stop_hash] = stop_layout
                                        stop_hashes.append(stop_hash)
                                    new_paths.append(path + stop_hashes)
                                else:
                                    new_paths.append(path + [new_hash])
                                
                                if new_hash == final_hash:
                                    if visualize:
                                        full_paths.append([path + stop_hashes, new_energy])
                                    else:
                                        full_paths.append([path + [new_hash], new_energy])
                                
            if len(new_layouts) == 0:
                break
            
            layouts = new_layouts
            energies = new_energies
            paths = new_paths
            
        best = sorted(full_paths, key = lambda x: x[1])[0]
            
        return best, unhash, empty_map
    
    best1, unhash1, empty_map1 = solve(part1_start_map)
    lowest_energy1 = best1[1]
    
    part2_insert = np.array([[" "," ","#","D","#","C","#","B","#","A","#"," "," "], \
                              [" "," ","#","D","#","B","#","A","#","C","#"," "," "]])
        
    part2_start_map = np.vstack([part1_start_map[:3,:], part2_insert, part1_start_map[3:,:]])
    
    best2, unhash2, empty_map2 = solve(part2_start_map)
    lowest_energy2 = best2[1]
    
    if visualize:
        history1 = [layout_to_plottable(unhash1[x], empty_map1) for x in best1[0]]
        history2 = [layout_to_plottable(unhash2[x], empty_map2) for x in best2[0]]
        
        if visualize:
            set_interval = 100
            
            fig, ax = plt.subplots(1,2)
            ims = []
            axes = ax.ravel()
            for i in range(0,max(len(history1),len(history2))):
                if i in range(len(history1)):
                   im1 = axes[0].imshow(history1[i], animated=True)
                   axes[0].set_xticks([])
                   axes[0].set_yticks([])
                if i in range(len(history2)):
                   im2 = axes[1].imshow(history2[i], animated=True)
                   axes[1].set_xticks([])
                   axes[1].set_yticks([])
                   
                ims.append([im1, im2])
            
            ims += [[im1, im2]]*(2000//set_interval)
            
            
            ani = animation.ArtistAnimation(fig, ims, interval=set_interval, blit=True, repeat_delay=1000)
    
    if visualize:
        return lowest_energy1, lowest_energy2, ani
    else:
        return lowest_energy1, lowest_energy2
    
#%%

#The difference here is pruning all branches that include a non-optimal state
#anywhere along them. Didn't speed up run-time, though.

@time_this_func
def day23_alt(visualize = False):    
    from copy import deepcopy
    
    with open("input23.txt") as f:
        rows = f.read().split("\n")
        row_len = len(rows[0])
        state = [[x for x in list(row)+[" "]*(row_len-len(row))] for row in rows[:-1]]
    part1_start_map = np.array(state)
    
    if visualize:
        def make_map_plottable(x):
            if x == " " or x == ".":
                return 0
            if x == "#":
                return 1
            if x == "A":
                return 3
            if x == "B":
                return 5
            if x == "C":
                return 7
            if x == "D":
                return 9
        
        num_map = np.vectorize(make_map_plottable)
        
        def layout_to_map(layout, empty_map):
            full_map = empty_map.copy()
            types = {0:"A", 1:"B", 2:"C", 3:"D"}
            for t, amphipod_group in enumerate(layout):
                for amphipod in amphipod_group:
                    full_map[amphipod] = types[t]
            return full_map
        
        def layout_to_plottable(layout, empty_map):
            full_map = layout_to_map(layout, empty_map)
            plottable_map = num_map(full_map)
            return plottable_map
            
    
    def solve(original_map):
        amphipods = [[],[],[],[]]
        spaces = set()
        
        for i in range(original_map.shape[0]):
            for j in range(original_map.shape[1]):
                if original_map[i,j] in {"A","B","C","D"}:
                    if original_map[i,j] == "A":
                        amphipods[0].append((i,j))
                    elif original_map[i,j] == "B":
                        amphipods[1].append((i,j))
                    elif original_map[i,j] == "C":
                        amphipods[2].append((i,j))
                    elif original_map[i,j] == "D":
                        amphipods[3].append((i,j))
                    spaces.add((i,j))
                elif original_map[i,j] == ".":
                    spaces.add((i,j))
        
        empty_map = original_map.copy()
        for i,j in sum(amphipods, []):
            empty_map[i,j] = "."
            
        def path_from_to(loc1, loc2):
            paths = [[loc1]]
            visited = {loc1}
            while True:
                new_paths = []
                for path in paths:
                    i,j = path[-1]
                    next_locs = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
                    for next_loc in next_locs:
                        if next_loc in visited or empty_map[next_loc] == "#":
                            continue
                        
                        if next_loc == loc2:
                            return path[1:] + [next_loc]
                        
                        new_paths.append(path + [next_loc])
                        visited.add(next_loc)
                paths = new_paths
                
        paths_from_to = {}
        for space1 in spaces:
            for space2 in spaces:
                if space1 == space2:
                    continue
                paths_from_to[(space1,space2)] = path_from_to(space1, space2)
                
        def can_go_to(loc1, loc2, amphipods):    
            path_spaces = set(paths_from_to[(loc1, loc2)])
            amphipod_locs = set(sum(amphipods, []))
            return (path_spaces.isdisjoint(amphipod_locs), len(path_spaces))
        
        def hash_state(amphipods):
            As = tuple(sorted(sorted(amphipods[0], key = lambda x: x[0]), key = lambda x: x[1]))
            Bs = tuple(sorted(sorted(amphipods[1], key = lambda x: x[0]), key = lambda x: x[1]))
            Cs = tuple(sorted(sorted(amphipods[2], key = lambda x: x[0]), key = lambda x: x[1]))
            Ds = tuple(sorted(sorted(amphipods[3], key = lambda x: x[0]), key = lambda x: x[1]))
            
            return hash((As, Bs, Cs, Ds))
        
        burrows = {0:[(x,3) for x in range(2,original_map.shape[0]-1)], \
                   1:[(x,5) for x in range(2,original_map.shape[0]-1)], \
                   2:[(x,7) for x in range(2,original_map.shape[0]-1)], \
                   3:[(x,9) for x in range(2,original_map.shape[0]-1)]}
        burrow_sets = {k:set(v) for k,v in burrows.items()}
        
        states = {hash_state(amphipods): 0}
        final_hash = hash_state([burrows[0],burrows[1],burrows[2],burrows[3]])
        states[final_hash] = np.inf
        
        unhash = {final_hash: [burrows[0],burrows[1],burrows[2],burrows[3]], \
                  hash_state(amphipods): amphipods}
        
        layouts = [amphipods]
        energies = [0]
        paths = [[hash_state(amphipods)]]
        full_paths = []
        while True:
            new_layouts = []
            new_energies = []
            new_paths = []
            for layout, energy, path in zip(layouts, energies, paths):
                
                if sum([states[x] for x in path]) < energy or energy > states[final_hash]:
                    continue
                
                for t, amphipod_group in enumerate(layout):
                    
                    if set(amphipod_group) == burrow_sets[t]:
                        continue
                    
                    for a, amphipod in enumerate(amphipod_group):
                        
                        if amphipod == burrows[t][1]:
                            continue
                        
                        other_amphipod_locs = set(sum([o for i,o in enumerate(layout) if i != t], []))
                        if other_amphipod_locs.isdisjoint(burrow_sets[t]):
                            can_go_spaces = burrow_sets[t]
                        else:
                            can_go_spaces = set()
                            
                        if amphipod[0] != 1:
                            can_go_spaces = can_go_spaces | {(1,1),(1,2),(1,4),(1,6),(1,8),(1,10),(1,11)}
                            
                        can_go_spaces = can_go_spaces - set(sum(layout,[]))
        
                        if burrows[t][1] in can_go_spaces:
                            can_go_spaces.discard(burrows[t][0])
                        
                        for space in can_go_spaces:                    
                            new_layout = deepcopy(layout)
                            new_layout[t][a] = space
                            new_hash = hash_state(new_layout)
                            can_it_go = can_go_to(amphipod, space, layout)
                            new_energy = energy + pow(10,t)*can_it_go[1]
                            
                            if can_it_go[0] and (new_hash not in states or new_energy < states[new_hash]):
                                states[new_hash] = new_energy
                                new_layouts.append(new_layout)
                                new_energies.append(new_energy)
                                new_paths.append(path + [new_hash])
                                unhash[new_hash] = new_layout
                                
                                if new_hash == final_hash:
                                    full_paths.append([path + [new_hash], new_energy])
                        
            if len(new_layouts) == 0:
                break
            
            layouts = new_layouts
            energies = new_energies
            paths = new_paths
            
        best = sorted(full_paths, key = lambda x: x[1])[0]
        
        if visualize:
            best_path_no_stops = best[0]
            best_path_with_stops = [unhash[best_path_no_stops[0]]]
            for i, major_stop in enumerate(best_path_no_stops[1:]):
                major_layout_from = unhash[best_path_no_stops[i]]
                major_layout_to = unhash[major_stop]
                flat_major_layout_from = set(sum(major_layout_from, []))
                flat_major_layout_to = set(sum(major_layout_to, []))
                
                found_from = False
                found_to = False
                for t in range(len(major_layout_from)):
                    for a in range(len(major_layout_from[0])):
                        if major_layout_from[t][a] not in flat_major_layout_to:
                            from_loc = major_layout_from[t][a]
                            from_loc_inds = (t,a)
                            found_from = True
                        if major_layout_to[t][a] not in flat_major_layout_from:
                            to_loc = major_layout_to[t][a]
                            found_to = True
                    if found_from and found_to:
                        break
                    
                stops = paths_from_to[(from_loc, to_loc)]
                for stop in stops:
                    new_layout = deepcopy(major_layout_from)
                    new_layout[from_loc_inds[0]][from_loc_inds[1]] = stop
                    best_path_with_stops.append(new_layout)
                    
            best[0] = best_path_with_stops              
            
        return best, unhash, empty_map
    
    best1, unhash1, empty_map1 = solve(part1_start_map)
    lowest_energy1 = best1[1]
    
    part2_insert = np.array([[" "," ","#","D","#","C","#","B","#","A","#"," "," "], \
                              [" "," ","#","D","#","B","#","A","#","C","#"," "," "]])
        
    part2_start_map = np.vstack([part1_start_map[:3,:], part2_insert, part1_start_map[3:,:]])
    
    best2, unhash2, empty_map2 = solve(part2_start_map)
    lowest_energy2 = best2[1]
    
    if visualize:
        history1 = [layout_to_plottable(x, empty_map1) for x in best1[0]]
        history2 = [layout_to_plottable(x, empty_map2) for x in best2[0]]
        
        if visualize:
            fig, ax = plt.subplots(1,2)
            ims = []
            axes = ax.ravel()
            for i in range(0,max(len(history1),len(history2))):
                if i in range(len(history1)):
                   im1 = axes[0].imshow(history1[i], animated=True)
                   axes[0].set_xticks([])
                   axes[0].set_yticks([])
                if i in range(len(history2)):
                   im2 = axes[1].imshow(history2[i], animated=True)
                   axes[1].set_xticks([])
                   axes[1].set_yticks([])
                   
                ims.append([im1, im2])
            
            ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
            
    if visualize:
        return lowest_energy1, lowest_energy2, ani
    else:
        return lowest_energy1, lowest_energy2
    

#%%
# Day 24: Arithmetic Logic Unit

@time_this_func
def day24():
    sections = []
    with open("input24.txt") as f:
        for l in f:
            new = l.strip().split()
            if new[0] == "inp":
                sections.append([])
            else:
                sections[-1].append(new)
                
    relevant = [[int(x[i][2]) for i in [3,4,14]] for x in sections]
    
    def get_z(d, a1, a2, input_num, last_z):   
        x_ = ((last_z%26 + a1) == input_num) == 0
        y_ = (input_num + a2) * x_
        return ((last_z//d) * ((25 * x_) + 1)) + y_
          
    new_correct_zs = {0}
    corrects = []
    for s, mod_div_add in enumerate(relevant[::-1]):
        print(s)
        correct_zs = new_correct_zs
        
        new_correct_zs = set()
        corrects.append(set())
        
        if s == 13:
            z_range = [0]
        else:
            z_range = range(4_500_000)
        for z in z_range:
            for num in range(1,10):
                
                out_z = get_z(*mod_div_add,num,z)
                
                if out_z in correct_zs:
    
                    corrects[-1].add((num,z,out_z))
                    new_correct_zs.add(z)
    
    correct_in_order = corrects[::-1]

    def make_layer(first_layer, second_layer = None):
        layer_dict = {}
        if second_layer != None:
            second_layer_input_zs = {z[1] for z in second_layer}
        else:
            second_layer_input_zs = {0}
        for num, z_in, z_out in first_layer:
            
            if z_out in second_layer_input_zs:
                if z_in not in layer_dict:
                    layer_dict[z_in] = {num:z_out}
                else:
                    layer_dict[z_in][num] = z_out
        return layer_dict
    
    layer_dicts = []
    for i in range(len(correct_in_order)):
        if i+1 in range(len(correct_in_order)):
            layer_dicts.append(make_layer(correct_in_order[i], correct_in_order[i+1]))
        else:
            layer_dicts.append(make_layer(correct_in_order[i]))
    
    nums = [9]*14
    while True:
        last = 0
        for i, layer_dict in enumerate(layer_dicts):
            if last not in layer_dict or nums[i] not in layer_dict[last]:
                if nums[i] == 1:
                    nums[i] = 9
                    nums[i-1] -= 1
                else:
                    nums[i] -= 1
                break
            
            else:
                last = layer_dict[last][nums[i]]
                
        else:
            break
        
    highest_num = int("".join([str(x) for x in nums]))
    
    
    nums = [1]*14
    while True:
        last = 0
        for i, layer_dict in enumerate(layer_dicts):
            if last not in layer_dict or nums[i] not in layer_dict[last]:
                if nums[i] == 9:
                    nums[i] = 1
                    nums[i-1] += 1
                else:
                    nums[i] += 1
                break
            
            else:
                last = layer_dict[last][nums[i]]
                
        else:
            break
        
    lowest_num = int("".join([str(x) for x in nums]))
    
    
    return highest_num, lowest_num
            
#%%
# Day 25: Sea Cucumber

@time_this_func
def day25(visualize = False):
    floor = []
    with open("input25.txt") as f:
        for l in f:
            floor.append([1 if x == ">" else 2 if x == "v" else 0 for x in l.strip()])
    floor = np.array(floor)
    
    if visualize:
        history = []
        
    step = 0        
    while True:
        if visualize:
            history.append(floor.copy())
            
        step += 1
        
        easties = floor == 1
        
        occupied = floor != 0
        east_occupied = np.hstack([occupied[:,1:], occupied[:,0].reshape(-1,1)])
        
        easties_want = np.hstack([easties[:,-1].reshape(-1,1), easties[:,:-1]])
        
        easties_that_moved = easties_want * (floor == 0)
        easties_that_stayed = easties * east_occupied
        new_easties = 1*(easties_that_moved + easties_that_stayed)
        
        southies = floor == 2
        floor_temp = new_easties + (2*southies)
                                                 
        occupied = floor_temp != 0
        south_occupied = np.vstack([occupied[1:,:], occupied[0,:].reshape(1,-1)])
        
        southies_want = np.vstack([southies[-1,:].reshape(1,-1), southies[:-1,:]])
        
        southies_that_moved = southies_want * (floor_temp == 0)
        southies_that_stayed = southies * south_occupied
        new_southies = (southies_that_moved + southies_that_stayed)*2
        
        new_floor = new_easties + new_southies
        
        if np.array_equal(floor, new_floor):
            break
        
        floor = new_floor
        
        
    if visualize:
        interval_time = 35
        
        fig, ax = plt.subplots()
        ims = []
        for i, state in enumerate(history):
            im = ax.imshow(state, animated=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ims.append([im])
            
        ims += [[im]]*(2000//interval_time)
        
        ani = animation.ArtistAnimation(fig, ims, interval=interval_time, blit=True, repeat_delay=100)
    
    
    if not visualize:
        return step
    else:
        return step, ani