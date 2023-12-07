#%%
import numpy as np
from copy import deepcopy

#%%
# Day 1: Inverse Captcha

def day1():
    with open("input1.txt") as f:
        numbers = f.read().strip()
        
    repeats = []
    for i in range(len(numbers)):
        if numbers[i] == numbers[(i+1)%len(numbers)]:
            repeats.append(int(numbers[i]))
        
    sum_repeats_next = sum(repeats)
    
    
    repeats = []
    for i in range(len(numbers)):
        if numbers[i] == numbers[(i + len(numbers)//2)%len(numbers)]:
            repeats.append(int(numbers[i]))
            
    sum_repeats_across = sum(repeats)
    
    
    return sum_repeats_next, sum_repeats_across

#%%
# Day 2: Corruption Checksum

def day2():
    rows = []
    with open("input2.txt") as f:
        for l in f:
            rows.append([int(x) for x in l.split()])
            
    checksum = sum([max(x)-min(x) for x in rows])
    
    
    divided_evenly = []
    for row in rows:
        for el in row:
            dividers = [True if el%x == 0 and el != x else False for x in row]
            if True in dividers:
                divided_evenly.append(el/row[dividers.index(True)])
                break
                
    sum_divided_evenly = int(sum(divided_evenly))
    
    
    return checksum, sum_divided_evenly

#%%
# Day 3: Spiral Memory

def day3():
    data_num = 361527
    
    square_size = 1
    while data_num > square_size**2:
        square_size += 2
    
    def around_square(square_size, seek_num):
        loc = [square_size//2,square_size//2]
        num = square_size**2
        if num == seek_num:
            return loc
        
        for _ in range(square_size-1):
            loc[1] -= 1
            num -= 1
            if num == seek_num:
                return loc
        for _ in range(square_size-1):
            loc[0] -= 1
            num -= 1
            if num == seek_num:
                return loc   
        for _ in range(square_size-1):
            loc[1] += 1
            num -= 1
            if num == seek_num:
                return loc
        for _ in range(square_size-2):
            loc[0] += 1
            num -= 1
            if num == seek_num:
                return loc
    
    loc = around_square(square_size, data_num)
    dist_to_center = sum([abs(x) for x in loc])
    
    
    def get_value(loc):
        value = 0
        if (loc[0]-1,loc[1]-1) in values.keys():
            value += values[(loc[0]-1,loc[1]-1)]
        if (loc[0]-1,loc[1]) in values.keys():
            value += values[(loc[0]-1,loc[1])]
        if (loc[0]-1,loc[1]+1) in values.keys():
            value += values[(loc[0]-1,loc[1]+1)]
        if (loc[0],loc[1]-1) in values.keys():
            value += values[(loc[0],loc[1]-1)]
        if (loc[0],loc[1]+1) in values.keys():
            value += values[(loc[0],loc[1]+1)]
        if (loc[0]+1,loc[1]-1) in values.keys():
            value += values[(loc[0]+1,loc[1]-1)]
        if (loc[0]+1,loc[1]) in values.keys():
            value += values[(loc[0]+1,loc[1])]
        if (loc[0]+1,loc[1]+1) in values.keys():
            value += values[(loc[0]+1,loc[1]+1)]   
        return value
            
    values = {}
    values[(0,0)] = 1
    values[(0,1)] = 1
    last = (0,1)
    while max(values.values()) < data_num:
        if (last[0],last[1]-1) in values.keys() and (last[0]-1,last[1]) not in values.keys():
            last = (last[0]-1,last[1])
            values[last] = get_value(last)
        elif (last[0]+1,last[1]) in values.keys():
            last = (last[0],last[1]-1)
            values[last] = get_value(last)
        elif (last[0],last[1]+1) in values.keys():
            last = (last[0]+1,last[1])
            values[last] = get_value(last)
        elif (last[0]-1,last[1]) in values.keys():
            last = (last[0],last[1]+1)
            values[last] = get_value(last)
            
    first_over = max(values.values())
    
        
    return dist_to_center, first_over

#%%
# Day 4: High-Entropy Passphrases

def day4():
    passphrases = []
    with open("input4.txt") as f:
        for l in f:
            passphrases.append(l.split())
            
    valid_passphrases = []
    for passphrase in passphrases:
        valid = True
        for word in passphrase:
            if passphrase.count(word) > 1:
                valid = False
                break
        if valid:
            valid_passphrases.append(passphrase)
    
    num_valid_no_repeats = len(valid_passphrases)
    
    
    passphrases = [["".join(sorted(x)) for x in y] for y in passphrases]
    valid_passphrases = []
    for passphrase in passphrases:
        valid = True
        for word in passphrase:
            if passphrase.count(word) > 1:
                valid = False
                break
        if valid:
            valid_passphrases.append(passphrase)
            
    num_valid_no_anagrams = len(valid_passphrases)
    
    
    return num_valid_no_repeats, num_valid_no_anagrams

#%%
# Day 5: A Maze of Twisty Trampolines, All Alike

def day5():
    jumps = []
    with open("input5.txt") as f:
        for l in f:
            jumps.append(int(l.strip()))
            
    jumps_og = jumps.copy()
    
    i = 0
    steps = 0
    while i in range(len(jumps)):
        steps += 1
        jump = jumps[i]
        jumps[i] += 1
        i += jump
    steps1 = steps
    
    
    jumps = jumps_og
    i = 0
    steps = 0
    while i in range(len(jumps)):
        steps += 1
        jump = jumps[i]
        if jump >= 3:
            jumps[i] -= 1
        else:
            jumps[i] += 1
        i += jump
    steps2 = steps
    
    
    return steps1, steps2

#%%
# Day 6: Memory Reallocation

def day6():
    with open("input6.txt") as f:
        banks = [int(x) for x in f.read().split()]
        
    states = [banks.copy()]
    redistributions = 0
    while True:
        max_i = banks.index(max(banks))
        to_redistribute = banks[max_i]
        banks[max_i]= 0
        for i in range(1,to_redistribute+1):
            banks[(max_i+i)%len(banks)] += 1
        redistributions += 1
            
        if banks in states:      
            break
        
        states.append(banks.copy())
        
        
    first_occur = states.index(banks)
    second_occur = len(states)
    cycle_len = second_occur - first_occur
    
    
    return redistributions, cycle_len
    
#%%
# Day 7: Recursive Circus

def day7():
    towers = {}
    with open("input7.txt") as f:
        for l in f:
            new = l.strip().split(" -> ")
            name_and_weight = new[0].split(" (")
            name = name_and_weight[0]
            weight = int(name_and_weight[1][:-1])
            if len(new) > 1:
                supports = new[1].split(", ")
            else:
                supports = []
            towers[name] = {"weight": weight, "supports":supports}
    
    supports_others = [x for x in towers.keys() if towers[x]["supports"]]
    supported = [towers[x]["supports"] for x in towers.keys() if towers[x]["supports"]]
    supported = sum(supported, [])
    
    for supporter in supports_others:
        if supporter not in supported:
            break
    base = supporter
    
    
    def get_weight(name):
        if towers[name]["supports"] == []:
            return towers[name]["weight"]
        else:
            return towers[name]["weight"] + sum([get_weight(x) for x in towers[name]["supports"]])
        
    def unbalanced_one(name):
        options = towers[name]["supports"]
        weights = [get_weight(x) for x in options]
        found = False
        for wt in weights:
            if weights.count(wt) == 1:
                found = True
                break
        
        if found:
            return options[weights.index(wt)]
        else:
            return None
        
    curr_tower = base
    while unbalanced_one(unbalanced_one(curr_tower)) != None:
        curr_tower = unbalanced_one(curr_tower)
        
    unbalanced_weights = [get_weight(x) for x in towers[curr_tower]["supports"]]
    correct = unbalanced_weights[1]
    for i, wt in enumerate(unbalanced_weights):
        if unbalanced_weights.count(wt) == 1:
            incorrect = wt
            break
        else:
            correct = wt
            
    correction = correct - incorrect
    unbalanced_tower = towers[curr_tower]["supports"][i]
    corrected_weight = towers[unbalanced_tower]["weight"] + correction
    
    
    return base, corrected_weight

#%%
# Day 8: I Heard You Like Registers

def day8():
    operations = []
    registers = {}
    with open("input8.txt") as f:
        for l in f:
            new = l.strip().split(" if ")
            to_do = new[0].split()
            to_do[-1] = int(to_do[-1])
            operations.append([to_do, new[1]])
            if to_do[0] not in registers.keys():
                registers[to_do[0]] = 0
    
    def evaluate_if(if_statement):
        register = if_statement.split()[0]
        return eval(if_statement,{register:registers[register]})
    
    highest_ever_val = 0
    for op in operations:
        if evaluate_if(op[1]):
            if op[0][1] == "inc":
                registers[op[0][0]] += op[0][2]
            elif op[0][1] == "dec":
                registers[op[0][0]] -= op[0][2]
            highest_ever_val = max(highest_ever_val, max(registers.values()))
    
    max_ending_val = max(registers.values())
    
    return max_ending_val, highest_ever_val
            
#%%
# Day 9: Stream Processing

def day9():
    with open("input9.txt") as f:
        stream = f.read().strip()  
     
    no_garbage = ""
    garbage_len = 0
    ignore_next = False
    is_garbage = False
    for c in stream:
        if ignore_next:
            ignore_next = False
            continue
        if c == "!":
            ignore_next = True
            continue
        if c == "<" and not is_garbage:
            is_garbage = True
            continue
        if c == ">" and is_garbage:
            is_garbage = False
            continue
        
        if not is_garbage:
            no_garbage += c
        else:
            garbage_len += 1
            
    layer_counts = {}
    num_open = 0
    for c in no_garbage:
        if c == "{":
            num_open += 1
        elif c == "}":
            num_open -= 1
            if num_open not in layer_counts.keys():
                layer_counts[num_open] = 1
            else:
                layer_counts[num_open] += 1
                
    total_score = sum([(l+1)*c for l,c in layer_counts.items()])
    
    return total_score, garbage_len

#%%
# Day 10: Knot Hash

def day10():
    with open("input10.txt") as f:
        lengths = [int(x) for x in f.read().strip().split(",")]
        
    skip = 0
    num_nums = 256
    nums = list(range(num_nums))
    i = 0
    
    for l in lengths:
        old_nums = nums.copy()
        back_flip = (i+l-1)%num_nums
        front_flip = i
        for _ in range(l//2):
            nums[front_flip] = old_nums[back_flip]
            nums[back_flip] = old_nums[front_flip]
            front_flip = (front_flip + 1)%num_nums
            back_flip = (back_flip - 1)%num_nums
        i = (i + l + skip)%num_nums
        skip += 1
        
    first_two_mult = nums[0] * nums[1]
    
    
    with open("input10.txt") as f:
        lengths = [ord(x) for x in f.read().strip()]
    lengths = lengths + [17, 31, 73, 47, 23]
    
    skip = 0
    num_nums = 256
    nums = list(range(num_nums))
    i = 0
    
    for _1 in range(64):
        for l in lengths:
            old_nums = nums.copy()
            back_flip = (i+l-1)%num_nums
            front_flip = i
            for _2 in range(l//2):
                nums[front_flip] = old_nums[back_flip]
                nums[back_flip] = old_nums[front_flip]
                front_flip = (front_flip + 1)%num_nums
                back_flip = (back_flip - 1)%num_nums
            i = (i + l + skip)%num_nums
            skip += 1
    
    final_hex = ""
    for i in range(0, num_nums, 16):
        block = nums[i:i+16]
        output_num = 0
        for b in block:
            output_num = output_num ^ b
            
        next_hex = hex(output_num)[2:]
        if len(next_hex) == 1:
            next_hex = "0" + next_hex
        final_hex += next_hex
        
        
    return first_two_mult, final_hex
    
#%%
# Day 11: Hex Ed

def day11():
    with open("input11.txt") as f:
        moves = f.read().strip().split(",")
        
    loc = [0,0]
    most_steps_away = 0
    for move in moves:
        if move == "nw":
            loc[0] -= 0.5
            loc[1] -= 1
        elif move == "n":
            loc[0] -= 1
        elif move == "ne":
            loc[0] -= 0.5
            loc[1] += 1
        if move == "sw":
            loc[0] += 0.5
            loc[1] -= 1
        elif move == "s":
            loc[0] += 1
        elif move == "se":
            loc[0] += 0.5
            loc[1] += 1
        most_steps_away = max(most_steps_away, int(loc[1] + loc[0] - loc[1]/2))
        
    final_steps_away =  int(loc[1] + loc[0] - loc[1]/2)
    
    return final_steps_away, most_steps_away

#%%
# Day 12: Digital Plumber

def day12():
    connections = {}
    with open("input12.txt") as f:
        for l in f:
            new = l.strip().split(" <-> ")
            from_num = int(new[0])
            to_nums = [int(x) for x in new[1].split(", ")]
            connections[from_num] = to_nums
            
    def all_connected_to(num, dont_look_back_to = []):
        connected_to = connections[num].copy()
        if num in connected_to and len(connected_to) == 1:
            return [num]
        
        for visited in dont_look_back_to:
            if visited in connected_to:
                connected_to.remove(visited)
            
        if len(connected_to) == 0:
            return [num]
        
        new_connections = []
        for connection in connected_to:
            new_connections += all_connected_to(connection, dont_look_back_to + [num])
        
        return list(set([num] + new_connections))
    
    num_zero_connects_to = len(all_connected_to(0))
    
    
    num_groups = 0
    grouped = []
    for num in connections.keys():
        if num in grouped:
            continue
        else:
            num_groups += 1
            grouped += all_connected_to(num)
            
    
    return num_zero_connects_to, num_groups

#%%
# Day 13: Packet Scanners

def day13():
    walls_raw = {}
    with open("input13.txt") as f:
        for l in f:
            new = [int(x) for x in l.strip().split(": ")]
            walls_raw[new[0]] = new[1]
    
    num_walls = max(walls_raw.keys())+1
    walls_orig = []
    for i in range(num_walls):
        if i in walls_raw.keys():
            walls_orig.append([0]*walls_raw[i])
            walls_orig[-1][0] = 1
        else:
            walls_orig.append([])
    
    class wall_class():
        def __init__(self, walls):
            self.walls = walls
            self.moving_down = [True]*len(walls)
            
        def move(self):
            for i,w in enumerate(self.walls):
                if len(w) <= 1:
                    continue
                
                curr_i = w.index(1)
                if self.moving_down[i]:
                    w[curr_i] = 0
                    w[curr_i + 1] = 1
                    if curr_i + 1 == len(w)-1:
                        self.moving_down[i] = False
                else:
                    w[curr_i] = 0
                    w[curr_i - 1] = 1
                    if curr_i - 1 == 0:
                        self.moving_down[i] = True
    
    walls = wall_class(walls_orig)
             
    caught = []
    for pos in range(num_walls):
        if len(walls.walls[pos]) == 0:
            caught.append(False)
        elif walls.walls[pos][0] == 1:
            caught.append(True)
        else:
            caught.append(False)
        walls.move()
        
    severity = sum([i*len(walls.walls[i]) for i in range(num_walls) if caught[i]])
    
    
    wall_periods = []
    for w in walls_orig:
        if len(w) == 0:
            wall_periods.append(np.inf)
        else:
            wall_periods.append(len(w)*2 - 2)
    wall_periods = np.array(wall_periods)
    
    will_enter = np.array(range(num_walls))
    
    wait = 0
    while True:
        wait += 1
        attempt = (will_enter+wait)%wall_periods
        if 0 not in attempt:
            break
        
        
    #part one also more succinctly calculated by:
    #sum([i*len(walls_orig[i]) for i in range(num_walls) if (will_enter%wall_periods)[i] == 0])
    
    return severity, wait

#%%
# Day 14: Disk Defragmentation

def day14():
    starter = "hfdlxzhv"
    
    def knot_hash(string):
        lengths = [ord(x) for x in string] + [17, 31, 73, 47, 23]
        
        skip = 0
        num_nums = 256
        nums = list(range(num_nums))
        i = 0
        
        for _1 in range(64):
            for l in lengths:
                old_nums = nums.copy()
                back_flip = (i+l-1)%num_nums
                front_flip = i
                for _2 in range(l//2):
                    nums[front_flip] = old_nums[back_flip]
                    nums[back_flip] = old_nums[front_flip]
                    front_flip = (front_flip + 1)%num_nums
                    back_flip = (back_flip - 1)%num_nums
                i = (i + l + skip)%num_nums
                skip += 1
        
        final_hex = ""
        for i in range(0, num_nums, 16):
            block = nums[i:i+16]
            output_num = 0
            for b in block:
                output_num = output_num ^ b
                
            next_hex = hex(output_num)[2:]
            if len(next_hex) == 1:
                next_hex = "0" + next_hex
            final_hex += next_hex
            
        return final_hex
    
    knot_hashes = []
    for n in range(128):
        knot_hashes.append(knot_hash(starter+"-"+str(n)))
        
    def hex_to_bin(char):
        hex_ = int(char, 16)
        to_bin = bin(hex_)[2:]
        to_bin = "0"*(4-len(to_bin)) + to_bin
        return to_bin
    
    disc = []
    for knot_hash in knot_hashes:
        disc.append("".join([hex_to_bin(x) for x in knot_hash]))
        
    total_used = sum([x.count("1") for x in disc])
    
    
    disc = np.array([[int(x) for x in y] for y in disc])
    
    all_used = []
    for i in range(disc.shape[0]):
        for j in range(disc.shape[1]):
            if disc[i,j] == 1:
                all_used.append((i,j))
                
    def get_group(loc, all_used = all_used):
        paths = [[loc]]
        in_group = set([loc])
        while True:
            new_paths = []
            all_in_paths = sum(paths, [])
            for path in paths:
                latest = path[-1]
                if (latest[0]-1, latest[1]) in all_used and (latest[0]-1, latest[1]) not in all_in_paths:
                    new_paths.append(path + [(latest[0]-1, latest[1])])
                if (latest[0]+1, latest[1]) in all_used and (latest[0]+1, latest[1]) not in all_in_paths:
                    new_paths.append(path + [(latest[0]+1, latest[1])])
                if (latest[0], latest[1]-1) in all_used and (latest[0], latest[1]-1) not in all_in_paths:
                    new_paths.append(path + [(latest[0], latest[1]-1)])
                if (latest[0], latest[1]+1) in all_used and (latest[0], latest[1]+1) not in all_in_paths:
                    new_paths.append(path + [(latest[0], latest[1]+1)])
                    
            if len(new_paths) == 0:
                break
            
            in_group = in_group | set(sum(new_paths, []))
            
            last_stops = []
            paths = []
            for p in new_paths:
                if p[-1] not in last_stops:
                    last_stops.append(p[-1])
                    paths.append(p)
            
        return list(in_group)
    
    grouped = []
    num_regions = 0
    for u in all_used:
        if u not in grouped:
            grouped += get_group(u)
            num_regions += 1
            
    
    return total_used, num_regions

#%%
# Day 15: Dueling Generators

def day15():
    with open("input15.txt") as f:
        for l in f:
            if "Generator A" in l:
                genA_start = int(l.split()[-1])
            else:
                genB_start = int(l.split()[-1])
                
    genA_factor = 16807
    genB_factor = 48271
                
    def generator(gen_start, gen_factor):
        last = gen_start
        while True:
            current = (last*gen_factor)%2147483647
            yield current
            last = current
            
    genA = generator(genA_start, genA_factor)
    genB = generator(genB_start, genB_factor)
    
    matching = 0
    for i in range(40000000):
        if bin(next(genA))[-16:] == bin(next(genB))[-16:]:
            matching += 1
    
    
    genA = generator(genA_start, genA_factor)
    genB = generator(genB_start, genB_factor)  
    
    matching2 = 0
    for i in range(5000000):
        while True:
            a = next(genA)
            if a%4 == 0:
                break
        while True:
            b = next(genB)
            if b%8 == 0:
                break
        
        if bin(a)[-16:] == bin(b)[-16:]:
            matching2 += 1
            
            
    return matching, matching2

#%%
# Day 16: Permutation Promenade

def day16():
    with open("input16.txt") as f:
        operations = f.read().strip().split(",")
        
    ops = []
    for op in operations:
        if op[0] == "s":
            ops.append([op[0],int(op[1:])])
        elif op[0] == "x":
            ops.append([op[0], [int(x) for x in op[1:].split("/")]])
        elif op[0] == "p":
            ops.append([op[0], op[1:].split("/")])
            
    orig_order = [x for x in "abcdefghijklmnop"]
            
    def dance(order):
        for op in ops:
            if op[0] == "s":
                order = order[-op[1]:] + order[:-op[1]]
            elif op[0] == "x":
                old_order = order.copy()
                order[op[1][0]] = old_order[op[1][1]]
                order[op[1][1]] = old_order[op[1][0]]
            elif op[0] == "p":
                ind1 = order.index(op[1][0])
                ind2 = order.index(op[1][1])
                old_order = order.copy()
                order[ind1] = old_order[ind2]
                order[ind2] = old_order[ind1]
        return order
    
    final_order_after_1 = "".join(dance(orig_order.copy()))
    
    
    order = orig_order.copy()
    rounds = 0
    while True:
        rounds += 1
        order = dance(order)
                
        if order == orig_order:
            break
    
    rounds_to_reset = rounds
    rounds_to_equal_1_bil = 1000000000%rounds_to_reset
    
    order = orig_order.copy()
    for _ in range(rounds_to_equal_1_bil):
        order = dance(order)
                
    final_order_after_1_bil = "".join(order)
    
    
    return final_order_after_1, final_order_after_1_bil

#%%
# Day 17: Spinlock

def day17():
    num_steps = 370
    
    spinlock = [0]
    i = 0
    for n in range(1,2017+1):
        i = (i+num_steps)%len(spinlock) + 1
        spinlock.insert(i,n)
    
    after_2017_initially = spinlock[(spinlock.index(2017)+1)%len(spinlock)]
    
    
    length = 1
    i = 0
    for n in range(1,50000000+1):
        i = (i+num_steps)%length + 1
        length += 1
        
        if i == 1:
            after_0 = n
            
            
    return after_2017_initially, after_0
        
#%%
# Day 18: Duet

def day18():
    operations = []
    registers = {}
    with open("input18.txt") as f:
        for l in f:
            new = l.split()
            for i in range(1,len(new)):
                if new[i].isalpha():
                    if new[i] not in registers.keys():
                        registers[new[i]] = 0
                else:
                    new[i] = int(new[i])     
            operations.append(new)
    
    last_played = None
    i = 0
    while i in range(len(operations)):
        op = operations[i]
        if op[0] == "snd":
            if type(op[1]) == int:
                last_played = op[1]
            else:
                last_played = registers[op[1]]
                
        elif op[0] == "set":
            if type(op[2]) == int:
                registers[op[1]] = op[2]
            else:
                registers[op[1]] = registers[op[2]]
                
        elif op[0] == "add":
            if type(op[2]) == int:
                registers[op[1]] += op[2]
            else:
                registers[op[1]] += registers[op[2]]
        
        elif op[0] == "mul":
            if type(op[2]) == int:
                registers[op[1]] *= op[2]
            else:
                registers[op[1]] *= registers[op[2]]
                
        elif op[0] == "mod":
            if type(op[2]) == int:
                registers[op[1]] %= op[2]
            else:
                registers[op[1]] %= registers[op[2]]
                
        elif op[0] == "rcv":
            if type(op[1]) == int and op[1] != 0:
                recovered = last_played
                break
            elif registers[op[1]] != 0:
                recovered = last_played
                break
            
        elif op[0] == "jgz":
            if (type(op[1]) == int and op[1] > 0) or registers[op[1]] > 0:
                if type(op[2]) == int:
                    i += op[2] - 1
                else:
                    i += registers[op[2]] - 1
        i += 1
        
    
    registers1 = dict(zip(registers.keys(),[0]*len(registers)))
    registers2 = dict(zip(registers.keys(),[0]*len(registers)))
    registers2["p"] = 1
    send_queue1 = []
    send_queue2 = []
    sent1 = 0
    sent2 = 0
    i1 = 0
    i2 = 0
    active1 = True
    active2 = True
    
    while ((active1 or len(send_queue2) > 0) and i1 in range(len(operations))) \
        or (active2 or len(send_queue1) > 0) and i2 in range(len(operations)):
        
        op1 = operations[i1]
        op2 = operations[i2]
        
        if (active1 or len(send_queue2) > 0) and i1 in range(len(operations)):
            if op1[0] == "snd":
                sent1 += 1
                if type(op1[1]) == int:
                    send_queue1.append(op1[1])
                else:
                    send_queue1.append(registers1[op1[1]])
                    
            elif op1[0] == "set":
                if type(op1[2]) == int:
                    registers1[op1[1]] = op1[2]
                else:
                    registers1[op1[1]] = registers1[op1[2]]
                    
            elif op1[0] == "add":
                if type(op1[2]) == int:
                    registers1[op1[1]] += op1[2]
                else:
                    registers1[op1[1]] += registers1[op1[2]]
            
            elif op1[0] == "mul":
                if type(op1[2]) == int:
                    registers1[op1[1]] *= op1[2]
                else:
                    registers1[op1[1]] *= registers1[op1[2]]
                    
            elif op1[0] == "mod":
                if type(op1[2]) == int:
                    registers1[op1[1]] %= op1[2]
                else:
                    registers1[op1[1]] %= registers1[op1[2]]
                    
            elif op1[0] == "rcv":
                if len(send_queue2) > 0:
                    registers1[op1[1]] = send_queue2.pop(0)
                    active1 = True
                else:
                    active1 = False
                
            elif op1[0] == "jgz":
                if (type(op1[1]) == int and op1[1] > 0) or registers1[op1[1]] > 0:
                    if type(op1[2]) == int:
                        i1 += op1[2] - 1
                    else:
                        i1 += registers1[op1[2]] - 1
            
            if active1:
                i1 += 1
            
        if (active2 or len(send_queue1) > 0) and i2 in range(len(operations)):
            if op2[0] == "snd":
                sent2 += 1
                if type(op2[1]) == int:
                    send_queue2.append(op2[1])
                else:
                    send_queue2.append(registers2[op2[1]])
                    
            elif op2[0] == "set":
                if type(op2[2]) == int:
                    registers2[op2[1]] = op2[2]
                else:
                    registers2[op2[1]] = registers2[op2[2]]
                    
            elif op2[0] == "add":
                if type(op2[2]) == int:
                    registers2[op2[1]] += op2[2]
                else:
                    registers2[op2[1]] += registers2[op2[2]]
            
            elif op2[0] == "mul":
                if type(op2[2]) == int:
                    registers2[op2[1]] *= op2[2]
                else:
                    registers2[op2[1]] *= registers2[op2[2]]
                    
            elif op2[0] == "mod":
                if type(op2[2]) == int:
                    registers2[op2[1]] %= op2[2]
                else:
                    registers2[op2[1]] %= registers2[op2[2]]
                    
            elif op2[0] == "rcv":
                if len(send_queue1) > 0:
                    registers2[op2[1]] = send_queue1.pop(0)
                    active2 = True
                else:
                    active2 = False
                
            elif op2[0] == "jgz":
                if (type(op2[1]) == int and op2[1] > 0) or registers2[op2[1]] > 0:
                    if type(op2[2]) == int:
                        i2 += op2[2] - 1
                    else:
                        i2 += registers2[op2[2]] - 1
                
            if active2:
                i2 += 1
                
                
    return recovered, sent2

#%%
# Day 19: A Series of Tubes

def day19():
    overview = []
    with open("input19.txt") as f:
        for l in f:
            overview.append([x for x in l[:-1]])
    start_col = overview[0].index("|")
    overview = np.array(overview)
    
    def is_path(location):
        return overview[(location[0], location[1])] in ["|", "-", "+"] or overview[(location[0], location[1])].isalpha()
    
    visited = ""
    loc = [0, start_col]
    going = "S"
    num_steps = 1
    while True:
        if going == "S":
            if is_path((loc[0]+1, loc[1])):
                loc[0] += 1
            elif is_path((loc[0], loc[1]-1)):
                loc[1] -= 1
                going = "W"
            elif is_path((loc[0], loc[1]+1)):
                loc[1] += 1
                going = "E"
            else:
                break
        elif going == "N":
            if is_path((loc[0]-1, loc[1])):
                loc[0] -= 1
            elif is_path((loc[0], loc[1]-1)):
                loc[1] -= 1
                going = "W"
            elif is_path((loc[0], loc[1]+1)):
                loc[1] += 1
                going = "E"
            else:
                break
        elif going == "W":
            if is_path((loc[0], loc[1]-1)):
                loc[1] -= 1
            elif is_path((loc[0]-1, loc[1])):
                loc[0] -= 1
                going = "N"
            elif is_path((loc[0]+1, loc[1])):
                loc[0] += 1
                going = "S"
            else:
                break
        elif going == "E":
            if is_path((loc[0], loc[1]+1)):
                loc[1] += 1
            elif is_path((loc[0]-1, loc[1])):
                loc[0] -= 1
                going = "N"
            elif is_path((loc[0]+1, loc[1])):
                loc[0] += 1
                going = "S"
            else:
                break
    
        if overview[tuple(loc)].isalpha():
            visited += overview[tuple(loc)]
            
        num_steps += 1
            
    return visited, num_steps

#%%
# Day 20: Particle Swarm

def day20():
    class particle():
        def __init__(self, pos, vel, acc):
            self.pos = pos
            self.vel = vel
            self.acc = acc
        
        def move(self):
            self.vel = [self.vel[i] + self.acc[i] for i in range(len(self.vel))]
            self.pos = [self.pos[i] + self.vel[i] for i in range(len(self.pos))]
        
        def dist(self):
            return sum([abs(x) for x in self.pos])
    
    particles = []
    with open("input20.txt") as f:
        for l in f:
            new = l.strip().split(", ")
            pos = [int(x) for x in new[0][3:-1].split(",")]
            vel = [int(x) for x in new[1][3:-1].split(",")]
            acc = [int(x) for x in new[2][3:-1].split(",")]
            particles.append(particle(pos, vel, acc))
    particles_orig = deepcopy(particles)
            
    for _ in range(350):
        for p in particles:
            p.move()
            
    closest_particle = [x.dist() for x in particles].index(min([x.dist() for x in particles]))
    
    
    particles = particles_orig
            
    for _ in range(40):
        for p in particles:
            p.move()
        poses = [x.pos for x in particles]
        particles = [particles[i] for i in range(len(particles)) if poses.count(poses[i]) == 1]
        
    surviving_particles = len(particles)
        
    
    return closest_particle, surviving_particles

#%%
# Day 21: Fractal Art

def day21():
    from_pat = []
    to_pat = []
    with open("input21.txt") as f:
        for l in f:
            new = l.strip().split(" => ")
            from_pat.append(new[0])
            to_pat.append(new[1])
    
    def string_to_array(string):
        pat = string.split("/")
        pat = [[1 if x == "#" else 0 for x in y] for y in pat]
        return np.array(pat)
    
    def array_to_string(numpy_array):
        string = ""
        for i in range(len(numpy_array)):
            string += "".join(["#" if x == 1 else "." for x in numpy_array[i,:]]) + "/"
        return string[:-1]
    
    def all_forms(string):
        pat = string_to_array(string)
        all_pats = []
        for rotations in range(4):
            pati = np.rot90(pat, rotations)
            all_pats += [pati, np.flipud(pati), np.flipud(pati)]
        return set([array_to_string(x) for x in all_pats])
    
    for i in range(len(from_pat)):
        from_pat[i] = all_forms(from_pat[i])
    
    image = np.array([[0,1,0],[0,0,1],[1,1,1]])
    for r in range(18):
        size = image.shape[0]
        
        if size%2 == 0:
            minis = []
            for i in range(0,size,2):
                for j in range(0,size,2):
                    minis.append(array_to_string(image[i:i+2,j:j+2]))
                    
            replacements = []
            for mini in minis:
                for rep_i in range(len(from_pat)):
                    if mini in from_pat[rep_i]:
                        break
                replacements.append(string_to_array(to_pat[rep_i]))
            
            rows = []
            for i in range(0,len(replacements),size//2):
                rows.append(np.hstack(replacements[i:i+size//2]))
            
            image = np.vstack(rows)
                
        elif size%3 == 0:
            minis = []
            for i in range(0,size,3):
                for j in range(0,size,3):
                    minis.append(array_to_string(image[i:i+3,j:j+3]))
                    
            replacements = []
            for mini in minis:
                for rep_i in range(len(from_pat)):
                    if mini in from_pat[rep_i]:
                        break
                replacements.append(string_to_array(to_pat[rep_i]))
                
            rows = []
            for i in range(0,len(replacements),size//3):
                rows.append(np.hstack(replacements[i:i+size//3]))
            
            image = np.vstack(rows)
            
        if r == 5-1:
            num_on_after_5 = np.sum(image)
    
    num_on_after_18 = np.sum(image)
    
    return num_on_after_5, num_on_after_18

#%%
# Day 22: Sporifica Virus

def day22():
    nodes = []
    with open("input22.txt") as f:
        for l in f:
            new = [1 if y == "#" else 0 for y in [x for x in l.strip()]]
            nodes.append(new)
    nodes = np.array(nodes)
    
    node_dict_orig = {}
    for i in range(nodes.shape[0]):
        for j in range(nodes.shape[1]):
            node_dict_orig[(i,j)] = nodes[i,j]
    
    def turn_left(was_facing):
        if was_facing == "N":
            return "W"
        elif was_facing == "W":
            return "S"
        elif was_facing == "S":
            return "E"
        elif was_facing == "E":
            return "N"
    
    def turn_right(was_facing):
        if was_facing == "N":
            return "E"
        elif was_facing == "E":
            return "S"
        elif was_facing == "S":
            return "W"
        elif was_facing == "W":
            return "N"
    
    node_dict = node_dict_orig.copy()
    facing = "N"
    loc = [nodes.shape[0]//2, nodes.shape[1]//2]
    
    infections = 0
    for _ in range(10000):
        if tuple(loc) in node_dict.keys() and node_dict[tuple(loc)] == 1:
            facing = turn_right(facing)
            node_dict[tuple(loc)] = 0
        else:
            facing = turn_left(facing)
            node_dict[tuple(loc)] = 1
            infections += 1
            
        if facing == "N":
            loc[0] -= 1
        elif facing == "S":
            loc[0] +=1
        elif facing == "W":
            loc[1] -= 1
        elif facing == "E":
            loc[1] += 1
            
    num_infections1 = infections
    
    
    node_dict = node_dict_orig.copy()
    facing = "N"
    loc = [nodes.shape[0]//2, nodes.shape[1]//2]
    
    infections = 0
    for _ in range(10000000):
        if tuple(loc) in node_dict.keys() and node_dict[tuple(loc)] == 1: #infected
            facing = turn_right(facing)
            node_dict[tuple(loc)] = -1
        elif tuple(loc) in node_dict.keys() and node_dict[tuple(loc)] == 0.5: #weakened
            infections += 1
            node_dict[tuple(loc)] = 1
        elif  tuple(loc) in node_dict.keys() and node_dict[tuple(loc)] == -1: #flagged
            node_dict[tuple(loc)] = 0
            facing = turn_right(turn_right(facing))
        else: #0 clean
            facing = turn_left(facing)
            node_dict[tuple(loc)] = 0.5
            
        if facing == "N":
            loc[0] -= 1
        elif facing == "S":
            loc[0] +=1
        elif facing == "W":
            loc[1] -= 1
        elif facing == "E":
            loc[1] += 1
    
    num_infections2 = infections
    
    
    return num_infections1, num_infections2

#%%
# Day 23: Coprocessor Conflagration

def day23():
    operations = []
    registers = dict(zip("abcdefgh",[0]*8))
    with open("input23.txt") as f:
        for l in f:
            new = l.split()
            if new[1].isnumeric() or "-" in new[1]:
                new[1] = int(new[1])
            if new[2].isnumeric() or "-" in new[2]:
                new[2] = int(new[2])
            operations.append(new)
            
    i = 0
    mul_count = 0
    while i in range(len(operations)):
        op = operations[i]
        
        if op[0] == "set":
            if type(op[2]) == int:
                registers[op[1]] = op[2]
            else:
                registers[op[1]] = registers[op[2]]
        
        elif op[0] == "sub":
            if type(op[2]) == int:
                registers[op[1]] -= op[2]
            else:
                registers[op[1]] -= registers[op[2]]   
                
        elif op[0] == "mul":
            mul_count += 1
            if type(op[2]) == int:
                registers[op[1]] *= op[2]
            else:
                registers[op[1]] *= registers[op[2]]  
                
        elif op[0] == "jnz":
            jump = False
            if (type(op[1]) == int and op[1] != 0) or (type(op[1]) == str and registers[op[1]] != 0):
                jump = True
            
            if jump:
                if type(op[2]) == int:
                    i += op[2] - 1
                else:
                    i += registers[op[2]] - 1
    
        i += 1
        
        
    #Needed a lot of help on this. Come back to it? Manual interpretation of assembly code needed.
    start_b = operations[0][2]*operations[4][2] - operations[5][2]
    start_c = start_b - operations[7][2]
    
    def is_prime(n):
        for f in range(2,n//2):
            if n%f == 0:
                return False
        return True
    
    h = 0
    for b in range(start_b, start_c + 1, 17):
        if not is_prime(b):
            h += 1
            
    
    return mul_count, h

#%%
# Day 24: Electromagnetic Moat

def day24():
    pieces = []
    with open("input24.txt") as f:
        for l in f:
            pieces.append(tuple([int(x) for x in l.strip().split("/")]))
            
    bridges = []
    for p in pieces:
        if p[0] == 0:
            bridges.append([p])
        elif p[1] == 0:
            bridges.append([p[::-1]])
            
    strongest = 0
    while True:
        new_bridges = []
        for bridge in bridges:
            latest = bridge[-1][1]
            for piece in pieces:
                if latest in piece and piece not in bridge and piece[::-1] not in bridge:
                    if piece[1] == latest:
                        new_bridges.append(bridge + [piece[::-1]])
                    else:
                        new_bridges.append(bridge + [piece])
        
        if len(new_bridges) != 0:
            bridges = new_bridges
            strongest = max(strongest, max([np.sum(x) for x in bridges]))
        else:
            break
    
    strongest_longest = max([np.sum(x) for x in bridges])
    
    return strongest, strongest_longest

#%%
# Day 25: The Halting Problem

def day25():
    instructions = {}
    with open("input25.txt") as f:
        current_state = None
        current_value = None
        if_zero = None
        if_one = None
        for l in f:
            if current_state == None:
                if "Begin in state" in l:
                    start_state = l.strip()[-2:-1]
                elif "Perform" in l:
                    num_steps = int(l.split()[-2])
                    current_state = "fake"
            
            elif "In state" in l:
                if current_state != "fake":
                    instructions[current_state] = {0: if_zero, 1: if_one}
                current_state = l.strip()[-2:-1]
            
            elif "If the current value is" in l:
                current_value = int(l.strip()[-2:-1])
                if current_value == 0:
                    if_zero = []
                elif current_value == 1:
                    if_one = []
                
            elif "Write the value" in l:
                if current_value == 0:
                    if_zero.append(int(l.strip()[-2:-1]))
                elif current_value == 1:
                    if_one.append(int(l.strip()[-2:-1]))
            
            elif "Move one slot" in l:
                if current_value == 0:
                    if_zero.append(l.strip().split()[-1][:-1])
                elif current_value == 1:
                    if_one.append(l.strip().split()[-1][:-1])     
                    
            elif "Continue with state" in l:
                if current_value == 0:
                    if_zero.append(l.strip()[-2:-1])
                elif current_value == 1:
                    if_one.append(l.strip()[-2:-1])
                    
        instructions[current_state] = {0: if_zero, 1: if_one}
    
    values = {}
    loc = 0
    state = start_state
    for _ in range(num_steps):
        if loc not in values.keys() or values[loc] == 0:
            curr_value = 0
        else:
            curr_value = 1

        values[loc] = instructions[state][curr_value][0]
        if instructions[state][curr_value][1] == "left":
            loc -= 1
        else:
            loc += 1
        state =  instructions[state][curr_value][2]
            
    checksum = sum(values.values())
    
    return checksum