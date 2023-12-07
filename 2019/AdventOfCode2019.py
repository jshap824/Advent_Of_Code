#%% 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation  

#%%
#Timer Wrapper Function

def time_this_func(func):
    from time import time
    def timed_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        print(f"{time()-t1:0.3f} s runtime")
        return result
    return timed_func

#%%
# Day 1: The Tyranny of the Rocket Equation

@time_this_func
def day1():
    with open("input1.txt") as f:
        modules = tuple(int(x) for x in f.readlines())
        
    def fuel(module_mass):
        return module_mass//3-2
    
    fuel_sum = sum([fuel(x) for x in modules])
    
    
    def recursive_fuel(mass):
        next_mass = fuel(mass)
        if next_mass > 0:
            return next_mass + recursive_fuel(next_mass)
        else:
            return 0
        
    recursive_fuel_sum = sum([recursive_fuel(x) for x in modules])
    
    
    return fuel_sum, recursive_fuel_sum

#%%
# Day 2: 1202 Program Alarm

@time_this_func
def day2():
    with open("input2.txt") as f:
        ints = tuple(int(x) for x in f.read().split(","))
        
    def get_output(ints_tuple, noun, verb):
        ints_list = list(ints_tuple)
        ints_list[1] = noun
        ints_list[2] = verb
        for i in range(0,len(ints_list),4):
            if ints_list[i] == 1:
                ints_list[ints_list[i+3]] = ints_list[ints_list[i+1]] + ints_list[ints_list[i+2]]
            elif ints_list[i] == 2:
                ints_list[ints_list[i+3]] = ints_list[ints_list[i+1]] * ints_list[ints_list[i+2]]
            elif ints_list[i] == 99:
                break
            else:
                raise Exception("Invalid op code")
        return ints_list[0]
        
    output_1202 = get_output(ints, 12, 2)
    
    
    goal = 19690720
    found_goal = False
    for noun in range(0,100):
        for verb in range(0,100):
            output = get_output(ints, noun, verb)
            if output == goal:
                found_goal = True
                break
        if found_goal:
            break
        
    goal_noun_verb = 100*noun + verb

    
    return output_1202, goal_noun_verb

#%%
# Day 3: Crossed Wires

@time_this_func
def day3():
    wire_directions = []
    with open("input3.txt") as f:
        for l in f:
            new = tuple((x[0], int(x[1:])) for x in l.strip().split(","))
            wire_directions.append(new)
            
    def step_counter():
        s = 0
        while True:
            s += 1
            yield s
            
    wire_map = {(0,0): 1}
    step = step_counter()
    loc = [0,0]
    for direc in wire_directions[0]:
        if direc[0] == "U":
            for _ in range(direc[1]):
                loc[0] -= 1
                wire_map[tuple(loc)] = [1, next(step)]
        elif direc[0] == "D":
            for _ in range(direc[1]):
                loc[0] += 1
                wire_map[tuple(loc)] = [1, next(step)]
        elif direc[0] == "L":
            for _ in range(direc[1]):
                loc[1] -= 1
                wire_map[tuple(loc)] = [1, next(step)]
        elif direc[0] == "R":
            for _ in range(direc[1]):
                loc[1] += 1
                wire_map[tuple(loc)] = [1, next(step)]
           
    intersections = []     
    intersection_steps = []       
           
    step = step_counter()
    loc = [0,0]
    for direc in wire_directions[1]:
        if direc[0] == "U":
            for _ in range(direc[1]):
                loc[0] -= 1
                if tuple(loc) in wire_map and wire_map[tuple(loc)][0] == 1:
                    intersections.append(tuple(loc))
                    intersection_steps.append(wire_map[tuple(loc)][1] + next(step))
                    wire_map[tuple(loc)][0] += 2
                else:
                    wire_map[tuple(loc)] = (2, next(step))
        elif direc[0] == "D":
            for _ in range(direc[1]):
                loc[0] += 1
                if tuple(loc) in wire_map and wire_map[tuple(loc)][0] == 1:
                    intersections.append(tuple(loc))
                    intersection_steps.append(wire_map[tuple(loc)][1] + next(step))
                    wire_map[tuple(loc)][0] += 2
                else:
                    wire_map[tuple(loc)] = (2, next(step))
        elif direc[0] == "L":
            for _ in range(direc[1]):
                loc[1] -= 1
                if tuple(loc) in wire_map and wire_map[tuple(loc)][0] == 1:
                    intersections.append(tuple(loc))
                    intersection_steps.append(wire_map[tuple(loc)][1] + next(step))
                    wire_map[tuple(loc)][0] += 2
                else:
                    wire_map[tuple(loc)] = (2, next(step))
        elif direc[0] == "R":
            for _ in range(direc[1]):
                loc[1] += 1
                if tuple(loc) in wire_map and wire_map[tuple(loc)][0] == 1:
                    intersections.append(tuple(loc))
                    intersection_steps.append(wire_map[tuple(loc)][1] + next(step))
                    wire_map[tuple(loc)][0] += 2
                else:
                    wire_map[tuple(loc)] = (2, next(step))
                    
    min_intersection_dist = min([abs(x[0]) + abs(x[1]) for x in intersections])
    
    min_intersection_step_sum = min(intersection_steps)
    
    return min_intersection_dist, min_intersection_step_sum

#%%
# Day 4: Secure Container

@time_this_func
def day4():
    num_options = range(136818, 685979+1)
    
    valid_passwords1 = []
    valid_passwords2 = []
    for num in num_options:
        str_num = [x for x in str(num)]
        
        cond1 = False
        cond2 = True
        cond3 = False
        for i in range(1,len(str_num)):
            if str_num[i-1] == str_num[i]:
                cond1 = True
                if (i-1 == -1 or str_num[i-2] != str_num[i]) and (i+1 == len(str_num) or str_num[i+1] != str_num[i]):
                    cond3 = True
            if str_num[i-1] > str_num[i]:
                cond2 = False
                break
            
        if cond1 and cond2:
            valid_passwords1.append(num)
            
        if cond1 and cond2 and cond3:
            valid_passwords2.append(num)
    
    num_valid_passwords1 = len(valid_passwords1)
    
    num_valid_passwords2 = len(valid_passwords2)
    
    return num_valid_passwords1, num_valid_passwords2

#%%
# Day 5: Sunny with a Chance of Asteroids

@time_this_func
def day5():
    with open("input5.txt") as f:
        ints = tuple(int(x) for x in f.read().split(","))
    
    def get_output(input_val, ints_tuple = ints):
        ints_list = list(ints_tuple)
        i = 0
        while True:
            inst = str(ints_list[i])
            opcode = int(inst[-2:])
            modes = [int(x) for x in inst[:-2][::-1]]
            
            if opcode == 1:
                modes = modes + [0]*(3-len(modes))
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                else:
                    input1 = ints_list[i+1]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                else:
                    input2 = ints_list[i+2]
                    
                ints_list[ints_list[i+3]] = input1 + input2
                    
                i += 4
                
            if opcode == 2:
                modes = modes + [0]*(3-len(modes))
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                else:
                    input1 = ints_list[i+1]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                else:
                    input2 = ints_list[i+2]
                    
                ints_list[ints_list[i+3]] = input1 * input2
                    
                i += 4
            
            elif opcode == 3:
                ints_list[ints_list[i+1]] = input_val
            
                i += 2
            
            elif opcode == 4:
                modes = modes + [0]*(1-len(modes))
                
                if modes[0] == 0:
                    output = ints_list[ints_list[i+1]]
                else:
                    output = ints_list[i+1]
                    
                i += 2
                
            elif opcode == 5:
                modes = modes + [0]*(2-len(modes))
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                else:
                    input1 = ints_list[i+1]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                else:
                    input2 = ints_list[i+2]
                    
                if input1 != 0:
                    i = input2
                else:
                    i += 3
                    
            elif opcode == 6:
                modes = modes + [0]*(2-len(modes))
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                else:
                    input1 = ints_list[i+1]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                else:
                    input2 = ints_list[i+2]
                    
                if input1 == 0:
                    i = input2
                else:
                    i += 3
                    
            elif opcode == 7:
                modes = modes + [0]*(3-len(modes))
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                else:
                    input1 = ints_list[i+1]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                else:
                    input2 = ints_list[i+2]
                    
                if input1 < input2:
                    ints_list[ints_list[i+3]] = 1
                else:
                    ints_list[ints_list[i+3]] = 0
                
                i += 4
                
            elif opcode == 8:
                modes = modes + [0]*(3-len(modes))
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                else:
                    input1 = ints_list[i+1]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                else:
                    input2 = ints_list[i+2]
                    
                if input1 == input2:
                    ints_list[ints_list[i+3]] = 1
                else:
                    ints_list[ints_list[i+3]] = 0
                
                i += 4      
                
            elif opcode == 99:
                break
            
        return output
    
    
    return get_output(1), get_output(5)

#%%
# Day 6: Universal Orbit Map

@time_this_func
def day6():
    orbit_pairs = {}
    bodies = set()
    with open("input6.txt") as f:
        for l in f:
            new = l.strip().split(")")
            orbit_pairs[new[1]] = new[0]
            bodies.add(new[0])
            bodies.add(new[1])
            
    def count_orbits(first_orbiter):
        num_orbits = 0
        orbiter = first_orbiter
        while orbiter != "COM":
            num_orbits += 1
            orbiter = orbit_pairs[orbiter]
        return num_orbits
    
    total_orbits = 0
    for body in bodies:
        total_orbits += count_orbits(body)
        
    
    keys = list(orbit_pairs)
    values = list(orbit_pairs.values())
    links_to = {}
    for body in bodies:
        links_to[body] = []
        for key, value in zip(keys, values):
            if key == body:
                links_to[body].append(value)
            elif value == body:
                links_to[body].append(key)
        
    start = orbit_pairs["YOU"]
    finish = orbit_pairs["SAN"]
    
    paths = [[start]]
    while True:
        new_paths = []
        previously_visited = sum(paths, [])
        for path in paths:
            latest = path[-1]
            for option in links_to[latest]:
                if option not in (previously_visited + sum(new_paths, [])):
                    new_paths.append(path + [option])
        
        paths = new_paths
        if finish in [x[-1] for x in paths]:
            break
    
    orbits_apart = len(paths[0])-1
    
    
    return total_orbits, orbits_apart

#%%
# Day 7: Amplification Circuit

@time_this_func
def day7():
    from itertools import permutations
    
    with open("input7.txt") as f:
        ints = tuple(int(x) for x in f.read().split(","))
    
    def get_output(input_vals, ints_tuple = ints):
        output = None
        ints_list = list(ints_tuple)
        i = 0
        while True:
            inst = str(ints_list[i])
            opcode = int(inst[-2:])
            modes = [int(x) for x in inst[:-2][::-1]]
            
            if opcode == 1:
                modes = modes + [0]*(3-len(modes))
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                else:
                    input1 = ints_list[i+1]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                else:
                    input2 = ints_list[i+2]
                    
                ints_list[ints_list[i+3]] = input1 + input2
                    
                i += 4
                
            if opcode == 2:
                modes = modes + [0]*(3-len(modes))
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                else:
                    input1 = ints_list[i+1]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                else:
                    input2 = ints_list[i+2]
                    
                ints_list[ints_list[i+3]] = input1 * input2
                    
                i += 4
            
            elif opcode == 3:
                ints_list[ints_list[i+1]] = input_vals.pop(0)
            
                i += 2
            
            elif opcode == 4:
                modes = modes + [0]*(1-len(modes))
                
                if modes[0] == 0:
                    output = ints_list[ints_list[i+1]]
                else:
                    output = ints_list[i+1]
                    
                i += 2
                
            elif opcode == 5:
                modes = modes + [0]*(2-len(modes))
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                else:
                    input1 = ints_list[i+1]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                else:
                    input2 = ints_list[i+2]
                    
                if input1 != 0:
                    i = input2
                else:
                    i += 3
                    
            elif opcode == 6:
                modes = modes + [0]*(2-len(modes))
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                else:
                    input1 = ints_list[i+1]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                else:
                    input2 = ints_list[i+2]
                    
                if input1 == 0:
                    i = input2
                else:
                    i += 3
                    
            elif opcode == 7:
                modes = modes + [0]*(3-len(modes))
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                else:
                    input1 = ints_list[i+1]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                else:
                    input2 = ints_list[i+2]
                    
                if input1 < input2:
                    ints_list[ints_list[i+3]] = 1
                else:
                    ints_list[ints_list[i+3]] = 0
                
                i += 4
                
            elif opcode == 8:
                modes = modes + [0]*(3-len(modes))
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                else:
                    input1 = ints_list[i+1]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                else:
                    input2 = ints_list[i+2]
                    
                if input1 == input2:
                    ints_list[ints_list[i+3]] = 1
                else:
                    ints_list[ints_list[i+3]] = 0
                
                i += 4      
                
            elif opcode == 99:
                break
            
        return output
        
    possible_orders = list(permutations(range(5), 5))
    
    max_a5_out = 0
    for order in possible_orders:
        a1_out = get_output([order[0], 0])
        a2_out = get_output([order[1], a1_out])
        a3_out = get_output([order[2], a2_out])
        a4_out = get_output([order[3], a3_out])
        a5_out = get_output([order[4], a4_out])
        
        if a5_out > max_a5_out:
            max_a5_out = a5_out
    
    max_a5_out1 = max_a5_out
    
    
    def next_op(i, ints_list, input_vals):
        output = None
        inst = str(ints_list[i])
        opcode = int(inst[-2:])
        modes = [int(x) for x in inst[:-2][::-1]]
        
        if opcode == 1:
            modes = modes + [0]*(3-len(modes))
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            else:
                input1 = ints_list[i+1]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            else:
                input2 = ints_list[i+2]
                
            ints_list[ints_list[i+3]] = input1 + input2
                
            i += 4
            
        if opcode == 2:
            modes = modes + [0]*(3-len(modes))
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            else:
                input1 = ints_list[i+1]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            else:
                input2 = ints_list[i+2]
                
            ints_list[ints_list[i+3]] = input1 * input2
                
            i += 4
        
        elif opcode == 3:
            if len(input_vals) == 0:
                return i, output
            
            ints_list[ints_list[i+1]] = input_vals.pop(0)
        
            i += 2
        
        elif opcode == 4:
            modes = modes + [0]*(1-len(modes))
            
            if modes[0] == 0:
                output = ints_list[ints_list[i+1]]
            else:
                output = ints_list[i+1]
                
            i += 2
            
        elif opcode == 5:
            modes = modes + [0]*(2-len(modes))
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            else:
                input1 = ints_list[i+1]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            else:
                input2 = ints_list[i+2]
                
            if input1 != 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 6:
            modes = modes + [0]*(2-len(modes))
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            else:
                input1 = ints_list[i+1]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            else:
                input2 = ints_list[i+2]
                
            if input1 == 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 7:
            modes = modes + [0]*(3-len(modes))
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            else:
                input1 = ints_list[i+1]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            else:
                input2 = ints_list[i+2]
                
            if input1 < input2:
                ints_list[ints_list[i+3]] = 1
            else:
                ints_list[ints_list[i+3]] = 0
            
            i += 4
            
        elif opcode == 8:
            modes = modes + [0]*(3-len(modes))
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            else:
                input1 = ints_list[i+1]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            else:
                input2 = ints_list[i+2]
                
            if input1 == input2:
                ints_list[ints_list[i+3]] = 1
            else:
                ints_list[ints_list[i+3]] = 0
            
            i += 4      
            
        elif opcode == 99:
            return -1, output
        
        return i, output
    
    possible_orders = list(permutations(range(5,10), 5))
    
    max_a5_out = 0
    for order in possible_orders:
        a1_active = True
        a2_active = True
        a3_active = True
        a4_active = True
        a5_active = True
        
        a1_i = 0
        a2_i = 0
        a3_i = 0
        a4_i = 0
        a5_i = 0
        
        a1_ints_list = list(ints)
        a2_ints_list = list(ints)
        a3_ints_list = list(ints)
        a4_ints_list = list(ints)
        a5_ints_list = list(ints)
        
        a1_input_vals = [order[0], 0]
        a2_input_vals = [order[1]]
        a3_input_vals = [order[2]]
        a4_input_vals = [order[3]]
        a5_input_vals = [order[4]]
        
        while a1_active or a2_active or a3_active or a4_active or a5_active:
            if a1_active:
                a1_i, a1_out = next_op(a1_i, a1_ints_list, a1_input_vals)
                if a1_out != None:
                    a2_input_vals.append(a1_out)
                if a1_i < 0:
                    a1_active = False
                    
            if a2_active:
                a2_i, a2_out = next_op(a2_i, a2_ints_list, a2_input_vals)
                if a2_out != None:
                    a3_input_vals.append(a2_out)
                if a2_i < 0:
                    a2_active = False
                    
            if a3_active:
                a3_i, a3_out = next_op(a3_i, a3_ints_list, a3_input_vals)
                if a3_out != None:
                    a4_input_vals.append(a3_out)
                if a3_i < 0:
                    a3_active = False
        
            if a4_active:
                a4_i, a4_out = next_op(a4_i, a4_ints_list, a4_input_vals)
                if a4_out != None:
                    a5_input_vals.append(a4_out)
                if a4_i < 0:
                    a4_active = False
                    
            if a5_active:
                a5_i, a5_out = next_op(a5_i, a5_ints_list, a5_input_vals)
                if a5_out != None:
                    a1_input_vals.append(a5_out)
                    last_a5_out = a5_out
                if a5_i < 0:
                    a5_active = False
                    
        if last_a5_out > max_a5_out:
            max_a5_out = last_a5_out        
                
    max_a5_out2 = max_a5_out
    
    
    return max_a5_out1, max_a5_out2

#%%
# Day 8: Space Image Format

@time_this_func
def day8():
    with open("input8.txt") as f:
        digits = tuple(int(x) for x in f.read().strip())
        
    width = 25
    height = 6
    
    layer_digits = []
    for i in range(0,len(digits),width*height):
        layer_digits.append(digits[i:i+width*height])
        
    min_num_zeros = np.inf
    for layer in layer_digits:
        if layer.count(0) < min_num_zeros:
            min_num_zeros = layer.count(0)
            ones_times_twos = layer.count(1)*layer.count(2)
            
    
    layers = []
    for layer in layer_digits:
        layers.append(np.array(layer).reshape(height,width))
        
    final_image = np.ones([height, width])*-1
    for i in range(height):
        for j in range(width):
            layer_ind = 0
            while layers[layer_ind][i,j] == 2:
                layer_ind += 1
            final_image[i,j] = layers[layer_ind][i,j]
            
    plt.imshow(final_image)
    
    
    return ones_times_twos

#%%
# Day 9: Sensor Boost

@time_this_func
def day9():
    with open("input9.txt") as f:
        ints = tuple(int(x) for x in f.read().split(","))
    
    def get_output(input_vals, ints_tuple = ints):
        ints_list = list(ints_tuple)
        
        if type(input_vals) == int:
            input_vals = [input_vals]
        
        relative_base = 0
        i = 0
        while True:
            inst = str(ints_list[i])
            opcode = int(inst[-2:])
            modes = [int(x) for x in inst[:-2][::-1]]
            
            modes = modes + [0]*(3-len(modes)) 
            
            max_accessible_ind = max([(i+1) * (modes[0] == 1), \
                                      (i+2) * (modes[1] == 1), \
                                      (i+3) * (modes[2] == 1), \
                                      (ints_list[i+1]) * (modes[0] == 0), \
                                      (ints_list[i+2]) * (modes[1] == 0 and opcode in {1,2,5,6,7,8}), \
                                      (ints_list[i+3]) * (modes[2] == 0 and opcode in {1,2,7,8}), \
                                      (relative_base + ints_list[i+1]) * (modes[0] == 2), \
                                      (relative_base + ints_list[i+2]) * (modes[1] == 2), \
                                      (relative_base + ints_list[i+3]) * (modes[2] == 2)])
              
            if max_accessible_ind > len(ints_list)-1:
                ints_list += [0]*(max_accessible_ind-(len(ints_list)-1))     
            
            if opcode == 1: 
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                elif modes[0] == 1:
                    input1 = ints_list[i+1]
                elif modes[0] == 2:
                    input1 = ints_list[relative_base + ints_list[i+1]]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                elif modes[1] == 1:
                    input2 = ints_list[i+2]
                elif modes[1] == 2:
                    input2 = ints_list[relative_base + ints_list[i+2]]
                    
                if modes[2] == 0:
                    write_to_ind = ints_list[i+3]
                elif modes[2] == 2:
                    write_to_ind = relative_base + ints_list[i+3]
                    
                ints_list[write_to_ind] = input1 + input2
                    
                i += 4
                
            if opcode == 2:
    
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                elif modes[0] == 1:
                    input1 = ints_list[i+1]
                elif modes[0] == 2:
                    input1 = ints_list[relative_base + ints_list[i+1]]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                elif modes[1] == 1:
                    input2 = ints_list[i+2]
                elif modes[1] == 2:
                    input2 = ints_list[relative_base + ints_list[i+2]]
                    
                if modes[2] == 0:
                    write_to_ind = ints_list[i+3]
                elif modes[2] == 2:
                    write_to_ind = relative_base + ints_list[i+3]
                    
                ints_list[write_to_ind] = input1 * input2
                    
                i += 4
            
            elif opcode == 3:
                
                if modes[0] == 0:
                    ints_list[ints_list[i+1]] = input_vals.pop(0)
                elif modes[0] == 2:
                    ints_list[ints_list[i+1]+relative_base] = input_vals.pop(0)
            
                i += 2
            
            elif opcode == 4:
                
                if modes[0] == 0:
                    output = ints_list[ints_list[i+1]]
                elif modes[0] == 1:
                    output = ints_list[i+1]
                elif modes[0] == 2:
                    output = ints_list[ints_list[i+1]+relative_base]
                    
                i += 2
                
            elif opcode == 5:
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                elif modes[0] == 1:
                    input1 = ints_list[i+1]
                elif modes[0] == 2:
                    input1 = ints_list[relative_base + ints_list[i+1]]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                elif modes[1] == 1:
                    input2 = ints_list[i+2]
                elif modes[1] == 2:
                    input2 = ints_list[relative_base + ints_list[i+2]]
                    
                if input1 != 0:
                    i = input2
                else:
                    i += 3
                    
            elif opcode == 6:
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                elif modes[0] == 1:
                    input1 = ints_list[i+1]
                elif modes[0] == 2:
                    input1 = ints_list[relative_base + ints_list[i+1]]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                elif modes[1] == 1:
                    input2 = ints_list[i+2]
                elif modes[1] == 2:
                    input2 = ints_list[relative_base + ints_list[i+2]]
                    
                if input1 == 0:
                    i = input2
                else:
                    i += 3
                    
            elif opcode == 7:
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                elif modes[0] == 1:
                    input1 = ints_list[i+1]
                elif modes[0] == 2:
                    input1 = ints_list[relative_base + ints_list[i+1]]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                elif modes[1] == 1:
                    input2 = ints_list[i+2]
                elif modes[1] == 2:
                    input2 = ints_list[relative_base + ints_list[i+2]]
                    
                if modes[2] == 0:
                    write_to_ind = ints_list[i+3]
                elif modes[2] == 2:
                    write_to_ind = relative_base + ints_list[i+3]
                    
                if input1 < input2:
                    ints_list[write_to_ind] = 1
                else:
                    ints_list[write_to_ind] = 0
                
                i += 4
                
            elif opcode == 8:
                
                if modes[0] == 0:
                    input1 = ints_list[ints_list[i+1]]
                elif modes[0] == 1:
                    input1 = ints_list[i+1]
                elif modes[0] == 2:
                    input1 = ints_list[relative_base + ints_list[i+1]]
                if modes[1] == 0:
                    input2 = ints_list[ints_list[i+2]]
                elif modes[1] == 1:
                    input2 = ints_list[i+2]
                elif modes[1] == 2:
                    input2 = ints_list[relative_base + ints_list[i+2]]
                    
                if modes[2] == 0:
                    write_to_ind = ints_list[i+3]
                elif modes[2] == 2:
                    write_to_ind = relative_base + ints_list[i+3]
                    
                if input1 == input2:
                    ints_list[write_to_ind] = 1
                else:
                    ints_list[write_to_ind] = 0
                
                i += 4  
                
            elif opcode == 9:
    
                if modes[0] == 0:
                    relative_base += ints_list[ints_list[i+1]]
                elif modes[0] == 1:
                    relative_base += ints_list[i+1]
                elif modes[0] == 2:
                    relative_base += ints_list[relative_base + ints_list[i+1]]
                
                i += 2
                
            elif opcode == 99:
                break
            
        return output
    
    BOOST_keyword = get_output(1)
    
    distress_coordinates = get_output(2)
    
    return BOOST_keyword, distress_coordinates

#%%
# Day 10: Monitoring Station

@time_this_func
def day10():
    space = []
    asteroid_locs = []
    with open("input10.txt") as f:
        row = 0
        for l in f:
            new = [1 if x == "#" else 0 for x in l.strip()]
            for col in range(len(new)):
                if new[col] == 1:
                    asteroid_locs.append((row, col))
            space.append(new)
            row += 1
    space = np.array(space)
    
    def can_see(p1, p2):
        if p1 == p2:
            return False
        
        if abs(p1[1]-p2[1]) == 1 or abs(p1[0]-p2[0]) == 1:
            return True
        
        sorted_xs, sorted_ys_by_x = zip(*sorted(zip([p1[0], p2[0]], [p1[1], p2[1]])))
        if sorted_xs[0] == sorted_xs[1]:
            for y in range(min(sorted_ys_by_x)+1, max(sorted_ys_by_x)):
                if space[sorted_xs[0], y] == 1:
                    return False
        else:
            slope = (sorted_ys_by_x[1]-sorted_ys_by_x[0])/(sorted_xs[1]-sorted_xs[0])
            for x in range(sorted_xs[0]+1,sorted_xs[1]):
                y = sorted_ys_by_x[0] + slope*(x-sorted_xs[0])
                if y%1 == 0 and space[x,int(y)] == 1:
                    return False
        return True
    
    view_counts = []
    for asteroid in asteroid_locs:
        views = sum([can_see(asteroid, x) for x in asteroid_locs])
        view_counts.append(views)
        
    most_views = max(view_counts)
    
    
    base = asteroid_locs[np.argmax(view_counts)]
    
    seeable = [x for x in asteroid_locs if can_see(base, x)]
    
    def sight_angle_from_12(other):
        ray1 = [0-base[0], 0]
        ray2 = [other[0]-base[0], other[1]-base[1]]
        
        dot = np.dot(ray1, ray2)
        ray1_mag = np.sqrt(sum([x**2 for x in ray1]))
        ray2_mag = np.sqrt(sum([x**2 for x in ray2]))
        angle = np.arccos(dot/(ray1_mag*ray2_mag)) * (180/np.pi)
        
        if other[1] < base[1]:
            return 360 - angle
        else:
            return angle
        
    seeable = sorted(seeable, key = lambda x: sight_angle_from_12(x))
    
    vaporized_200 = seeable[199][1]*100 + seeable[199][0]
    
    
    return most_views, vaporized_200

#%%
# Day 11: Space Police

@time_this_func
def day11():
    with open("input11.txt") as f:
        ints = tuple(int(x) for x in f.read().split(","))
    
    def next_op(i, ints_list, input_vals, relative_base):
        output = None
        inst = str(ints_list[i])
        opcode = int(inst[-2:])
        modes = [int(x) for x in inst[:-2][::-1]]
        
        modes = modes + [0]*(3-len(modes)) 
        
        max_accessible_ind = max([(i+1) * (modes[0] == 1), \
                                  (i+2) * (modes[1] == 1), \
                                  (i+3) * (modes[2] == 1), \
                                  (ints_list[i+1]) * (modes[0] == 0), \
                                  (ints_list[i+2]) * (modes[1] == 0 and opcode in {1,2,5,6,7,8}), \
                                  (ints_list[i+3]) * (modes[2] == 0 and opcode in {1,2,7,8}), \
                                  (relative_base + ints_list[i+1]) * (modes[0] == 2), \
                                  (relative_base + ints_list[i+2]) * (modes[1] == 2), \
                                  (relative_base + ints_list[i+3]) * (modes[2] == 2)])
          
        if max_accessible_ind > len(ints_list)-1:
            ints_list += [0]*(max_accessible_ind-(len(ints_list)-1))      
        
        if opcode == 1: 
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            ints_list[write_to_ind] = input1 + input2
                
            i += 4
            
        if opcode == 2:
    
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
              
            ints_list[write_to_ind] = input1 * input2
                
            i += 4
        
        elif opcode == 3:
            
            if modes[0] == 0:
                ints_list[ints_list[i+1]] = input_vals.pop(0)
            elif modes[0] == 2:
                ints_list[ints_list[i+1]+relative_base] = input_vals.pop(0)
        
            i += 2
        
        elif opcode == 4:
            
            if modes[0] == 0:
                output = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                output = ints_list[i+1]
            elif modes[0] == 2:
                output = ints_list[ints_list[i+1]+relative_base]
                
            i += 2
            
        elif opcode == 5:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 != 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 6:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 == 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 7:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 < input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4
            
        elif opcode == 8:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 == input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4  
            
        elif opcode == 9:
    
            if modes[0] == 0:
                relative_base += ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                relative_base += ints_list[i+1]
            elif modes[0] == 2:
                relative_base += ints_list[relative_base + ints_list[i+1]]
            
            i += 2
            
        elif opcode == 99:
            i = -1
        
        return i, output, relative_base
    
    def turn(left_or_right, facing):
        if left_or_right == "left":
            if facing == "N":
                return "W"
            elif facing == "W":
                return "S"
            elif facing == "S":
                return "E"
            elif facing == "E":
                return "N"
        elif left_or_right == "right":
            if facing == "N":
                return "E"
            elif facing == "E":
                return "S"
            elif facing == "S":
                return "W"
            elif facing == "W":
                return "N"
            
    def paint_hull(starting_panel_color):
        ints_list = list(ints)
        loc = (0,0)
        painted = set()
        curr_color = {loc: starting_panel_color}
        facing = "N"
        i = 0
        relative_base = 0
        while True:
            if loc not in curr_color or curr_color[loc] == 0:
                input_vals = [0]
            else:
                input_vals = [1]
                
            output_vals = []
            while len(output_vals) != 2:
                i, output, relative_base =  next_op(i, ints_list, input_vals, relative_base)
                
                if output != None:
                    output_vals.append(output)
                
                if i < 0:
                    break
            
            if i < 0:
                break
            
            curr_color[loc] = output_vals[0]
            painted.add(loc)
            
            if output_vals[1] == 0:
                facing = turn("left", facing)
            elif output_vals[1] == 1:
                facing = turn("right", facing)
                
            if facing == "N":
                loc = (loc[0]-1, loc[1])
            elif facing == "W":
                loc = (loc[0], loc[1]-1)
            elif facing == "S":
                loc = (loc[0]+1, loc[1])
            elif facing == "E":
                loc = (loc[0], loc[1]+1)
                
        num_panels_painted = len(painted)
        
        return num_panels_painted, curr_color
    
    
    num_panels_painted, _ = paint_hull(0)
    
    
    _, hull_message_dict = paint_hull(1)
    
    min_row = min([x[0] for x in hull_message_dict])
    max_row = max([x[0] for x in hull_message_dict])
    min_col = min([x[1] for x in hull_message_dict])
    max_col = max([x[1] for x in hull_message_dict])
    
    message = np.zeros([max_row - min_row + 1, max_col - min_col + 1])
    for loc in hull_message_dict:
        if hull_message_dict[loc] == 1:
            message[loc] = 1
            
    plt.imshow(message)
    
    
    return num_panels_painted

#%%
# Day 12: The N-Body Problem

@time_this_func
def day12():
    from itertools import combinations
    from math import gcd
    
    class planet():
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.vx = 0
            self.vy = 0
            self.vz = 0
    
    moons = []
    with open("input12.txt") as f:
        for l in f:
            new = l.strip()[1:-1].split(", ")
            x = int(new[0][2:])
            y = int(new[1][2:])
            z = int(new[2][2:])
            moons.append(planet(x,y,z))
            
    def energy(moon):
        potential = abs(moon.x) + abs(moon.y) + abs(moon.z)
        kinetic = abs(moon.vx) + abs(moon.vy) + abs(moon.vz)
        return potential*kinetic
            
    pairs = list(combinations(range(len(moons)),2))
    x_starts = tuple((m.x, m.vx) for m in moons)
    y_starts = tuple((m.y, m.vy) for m in moons)
    z_starts = tuple((m.z, m.vz) for m in moons)
    
    x_repeated = False
    y_repeated = False
    z_repeated = False
    s = 0
    while x_repeated*y_repeated*z_repeated == 0:
        for pair in pairs:
            if moons[pair[0]].x < moons[pair[1]].x:
                moons[pair[0]].vx += 1
                moons[pair[1]].vx -= 1
            elif moons[pair[0]].x > moons[pair[1]].x:
                moons[pair[0]].vx -= 1
                moons[pair[1]].vx += 1
                
            if moons[pair[0]].y < moons[pair[1]].y:
                moons[pair[0]].vy += 1
                moons[pair[1]].vy -= 1
            elif moons[pair[0]].y > moons[pair[1]].y:
                moons[pair[0]].vy -= 1
                moons[pair[1]].vy += 1  
                
            if moons[pair[0]].z < moons[pair[1]].z:
                moons[pair[0]].vz += 1
                moons[pair[1]].vz -= 1
            elif moons[pair[0]].z > moons[pair[1]].z:
                moons[pair[0]].vz -= 1
                moons[pair[1]].vz += 1
            
        for i in range(len(moons)):
            moons[i].x += moons[i].vx
            moons[i].y += moons[i].vy
            moons[i].z += moons[i].vz
            
        s += 1
            
        if  tuple((m.x, m.vx) for m in moons) == x_starts and not x_repeated:
            x_repeat_cycle_len = s
            x_repeated = True
        if  tuple((m.y, m.vy) for m in moons) == y_starts and not y_repeated:
            y_repeat_cycle_len = s
            y_repeated = True
        if  tuple((m.z, m.vz) for m in moons) == z_starts and not z_repeated:
            z_repeat_cycle_len = s
            z_repeated = True
    
        if s == 1000:
            total_energy = sum(energy(x) for x in moons)
            
    def least_common_multiple(a):
      lcm = a[0]
      for i in range(1,len(a)):
        lcm = lcm*a[i]//gcd(lcm, a[i])
      return lcm
    
    all_cycle_lens = [x_repeat_cycle_len, y_repeat_cycle_len, z_repeat_cycle_len]
    
    cycle_len = least_common_multiple(all_cycle_lens)
    
    return total_energy, cycle_len

#%%
# Day 13: Care Package

@time_this_func
def day13(visualize = False):
    with open("input13.txt") as f:
        ints = tuple(int(x) for x in f.read().split(","))
    
    def next_op(i, ints_list, input_vals, relative_base):
        needs_input = False
        output = None
        inst = str(ints_list[i])
        opcode = int(inst[-2:])
        
        modes = [int(x) for x in inst[:-2][::-1]]      
        modes = modes + [0]*(3-len(modes)) 
        
        max_accessible_ind = max([(i+1) * (modes[0] == 1), \
                                  (i+2) * (modes[1] == 1), \
                                  (i+3) * (modes[2] == 1), \
                                  (ints_list[i+1]) * (modes[0] == 0), \
                                  (ints_list[i+2]) * (modes[1] == 0 and opcode in {1,2,5,6,7,8}), \
                                  (ints_list[i+3]) * (modes[2] == 0 and opcode in {1,2,7,8}), \
                                  (relative_base + ints_list[i+1]) * (modes[0] == 2), \
                                  (relative_base + ints_list[i+2]) * (modes[1] == 2), \
                                  (relative_base + ints_list[i+3]) * (modes[2] == 2)])
          
        if max_accessible_ind > len(ints_list)-1:
            ints_list += [0]*(max_accessible_ind-(len(ints_list)-1))      
        
        if opcode == 1: 
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            ints_list[write_to_ind] = input1 + input2
                
            i += 4
            
        if opcode == 2:
    
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
              
            ints_list[write_to_ind] = input1 * input2
                
            i += 4
        
        elif opcode == 3:
            
            if len(input_vals) == 0:
                needs_input = True
                return i, output, relative_base, needs_input
            
            if modes[0] == 0:
                ints_list[ints_list[i+1]] = input_vals.pop(0)
            elif modes[0] == 2:
                ints_list[ints_list[i+1]+relative_base] = input_vals.pop(0)
        
            i += 2
        
        elif opcode == 4:
            
            if modes[0] == 0:
                output = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                output = ints_list[i+1]
            elif modes[0] == 2:
                output = ints_list[ints_list[i+1]+relative_base]
                
            i += 2
            
        elif opcode == 5:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 != 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 6:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 == 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 7:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 < input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4
            
        elif opcode == 8:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 == input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4  
            
        elif opcode == 9:
    
            if modes[0] == 0:
                relative_base += ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                relative_base += ints_list[i+1]
            elif modes[0] == 2:
                relative_base += ints_list[relative_base + ints_list[i+1]]
            
            i += 2
            
        elif opcode == 99:
            i = -1
        
        return i, output, relative_base, needs_input
    
    ints_list = list(ints)
    i = 0
    relative_base = 0
    input_vals = []
    outputs = []
    while True:
        i, output, relative_base, _ = next_op(i, ints_list, input_vals, relative_base)
        
        if output != None:
            outputs.append(output)
            
        if i < 0:
            break
    
    elements = {}
    for i in range(0,len(outputs),3):
        from_left = outputs[i]
        from_top = outputs[i+1]
        el_type = outputs[i+2]
            
        elements[(from_top, from_left)] = el_type
    
    num_blocks = list(elements.values()).count(2)
    
    
    all_locs = list(elements)
    all_ys = [x[0] for x in all_locs]
    all_xs = [x[1] for x in all_locs]
    
    min_y = min(all_ys)
    max_y = max(all_ys)
    min_x = min(all_xs)
    max_x = max(all_xs)
    
    def get_frame():
        frame = np.zeros([max_y-min_y+1, max_x-min_x+1])
        for e in elements:
            frame[e] = elements[e]
            
        return frame
        
    def get_x(get):
        locs = list(elements)
        elems = list(elements.values())
        
        if get == "ball":
            x_loc = locs[elems.index(4)][1]
        elif get == "paddle":
            x_loc = locs[elems.index(3)][1]
        
        return x_loc
        
    ints_list = list(ints)
    ints_list[0] = 2
    i = 0
    relative_base = 0
    input_vals = []
    outputs = []
    elements = {}
    score = 0

    history = []
    start_recording = False
    while True:
        i, output, relative_base, needs_input = next_op(i, ints_list, input_vals, relative_base)
        
        if needs_input:
            start_recording = True
            if get_x("ball") < get_x("paddle"):
                input_vals = [-1]
            elif get_x("ball") > get_x("paddle"):
                input_vals = [1]
            else:
                input_vals = [0]
        
        if output != None:
            outputs.append(output)
            
            if len(outputs) == 3:
                from_left = outputs[0]
                from_top = outputs[1]
                el_type_or_score = outputs[2]
                
                if from_left == -1 and from_top == 0:
                    score = el_type_or_score
                else:
                    elements[(from_top, from_left)] = el_type_or_score
                    
                    if start_recording and visualize:
                        if 3 in elements.values() and 4 in elements.values():
                            history.append(get_frame())
                
                outputs = []
                
        if i < 0:
            break
    
    if visualize:
        fig, ax = plt.subplots()
        ims = []
        for i in range(len(history)):
            im = ax.imshow(history[i], animated=True, vmin = 0, vmax = 4)
            ims.append([im])
        
        ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=10)
        
    
    if not visualize:
        return num_blocks, score
    else:
        return num_blocks, score, ani

#%%
# Day 14: Space Stoichiometry

@time_this_func
def day14():
    reactions = {}
    with open("input14.txt") as f:
        for l in f:
            new = l.strip().split(" => ")
            reactants = [tuple(x.split()) for x in new[0].split(", ")]
            reactants = tuple((int(x[0]), x[1]) for x in reactants)
            product = tuple(new[1].split())
            product = (int(product[0]), product[1])
            reactions[product] = reactants
            
    num_products = tuple(x[0] for x in reactions)
    products = tuple(x[1] for x in reactions)
    
    def reaction_details(end_product):
        ind = products.index(end_product)
        num_end_product = num_products[ind]
        key = (num_end_product, end_product)
        reactants_info = reactions[key]
        return reactants_info, num_end_product
    
    def get_ore_input(product, num_product_wanted):
        while extras[product] > 0:
            if num_product_wanted == 0:
                break
            extras[product] -= 1
            num_product_wanted -= 1
        
        reactants_info, num_product_produced_per_reactants = reaction_details(product)
        num_reactions_needed = int(np.ceil(num_product_wanted/num_product_produced_per_reactants))
        
        extras[product] += (num_reactions_needed*num_product_produced_per_reactants)-num_product_wanted
        
        ore_needed = 0
        for num_reactant, reactant in reactants_info:
            if reactant == "ORE":
                ore_needed += num_reactions_needed*num_reactant
            else:
                ore_needed += get_ore_input(reactant, num_reactions_needed*num_reactant)
                
        return ore_needed
         
    extras = dict(zip(products, [0]*len(products)))       
    ore_for_1_fuel = get_ore_input("FUEL", 1)
    
    
    intervals_up_then_down = [1000000,100000,10000,1000,100,1] #must be even number, end on 1
    num_fuel = 1
    for i, interval in enumerate(intervals_up_then_down):
        extras = dict(zip(products, [0]*len(products)))
        if i%2 == 0:
            while get_ore_input("FUEL", num_fuel) <= 1_000_000_000_000:
                num_fuel += interval
        
        else:
            while get_ore_input("FUEL", num_fuel) > 1_000_000_000_000:
                num_fuel -= interval
                
    fuel_from_1T_ore = num_fuel
                
    
    return ore_for_1_fuel, fuel_from_1T_ore

#%%
# Day 15: Oxygen System

@time_this_func
def day15(visualize = False):
    with open("input15.txt") as f:
        ints = tuple(int(x) for x in f.read().split(","))
    
    def next_op(i, ints_list, input_vals, relative_base):
        needs_input = False
        output = None
        inst = str(ints_list[i])
        opcode = int(inst[-2:])
        
        modes = [int(x) for x in inst[:-2][::-1]]      
        modes = modes + [0]*(3-len(modes)) 
        
        max_accessible_ind = max([(i+1) * (modes[0] == 1), \
                                  (i+2) * (modes[1] == 1), \
                                  (i+3) * (modes[2] == 1), \
                                  (ints_list[i+1]) * (modes[0] == 0), \
                                  (ints_list[i+2]) * (modes[1] == 0 and opcode in {1,2,5,6,7,8}), \
                                  (ints_list[i+3]) * (modes[2] == 0 and opcode in {1,2,7,8}), \
                                  (relative_base + ints_list[i+1]) * (modes[0] == 2), \
                                  (relative_base + ints_list[i+2]) * (modes[1] == 2), \
                                  (relative_base + ints_list[i+3]) * (modes[2] == 2)])
          
        if max_accessible_ind > len(ints_list)-1:
            ints_list += [0]*(max_accessible_ind-(len(ints_list)-1))      
        
        if opcode == 1: 
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            ints_list[write_to_ind] = input1 + input2
                
            i += 4
            
        if opcode == 2:
    
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
              
            ints_list[write_to_ind] = input1 * input2
                
            i += 4
        
        elif opcode == 3:
            
            if len(input_vals) == 0:
                needs_input = True
                return i, output, relative_base, needs_input
            
            if modes[0] == 0:
                ints_list[ints_list[i+1]] = input_vals.pop(0)
            elif modes[0] == 2:
                ints_list[ints_list[i+1]+relative_base] = input_vals.pop(0)
        
            i += 2
        
        elif opcode == 4:
            
            if modes[0] == 0:
                output = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                output = ints_list[i+1]
            elif modes[0] == 2:
                output = ints_list[ints_list[i+1]+relative_base]
                
            i += 2
            
        elif opcode == 5:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 != 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 6:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 == 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 7:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 < input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4
            
        elif opcode == 8:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 == input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4  
            
        elif opcode == 9:
    
            if modes[0] == 0:
                relative_base += ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                relative_base += ints_list[i+1]
            elif modes[0] == 2:
                relative_base += ints_list[relative_base + ints_list[i+1]]
            
            i += 2
            
        elif opcode == 99:
            i = -1
        
        return i, output, relative_base, needs_input
    
    
    def check_next(input_vals):
        i = 0
        relative_base = 0
        ints_list = list(ints)
        outputs = []
        while True:
            i, output, relative_base, needs_input = next_op(i, ints_list, input_vals, relative_base)
            
            if needs_input:
                break
            
            if output != None:
                outputs.append(output)
                
        if 0 in outputs[:-1]:
            raise Exception("Didn't make it to end of path")
            
        return outputs[-1]
                
    def inputs_from_path(path_of_locs):
        if len(path_of_locs) <= 1:
            return []
        
        inputs = []
        for i, loc in enumerate(path_of_locs[1:]):
            most_recent = path_of_locs[i]
            if loc[0] == most_recent[0] - 1:
                inputs.append(1)
            elif loc[0] == most_recent[0] + 1:
                inputs.append(2)
            elif loc[1] == most_recent[1] - 1:
                inputs.append(3)
            elif loc[1] == most_recent[1] + 1:
                inputs.append(4)
    
        return inputs
    
    paths = [[(0,0)]]
    all_visited = set()
    found_oxygen = False
    while True:
        new_paths = []
        visited = sum(paths, [])
        all_visited = all_visited | set(visited)
        for path in paths:
            latest = path[-1]
            if check_next(inputs_from_path(path + [(latest[0]-1, latest[1])])) != 0 and \
                (latest[0]-1, latest[1]) not in visited + sum(new_paths, []):
                new_paths.append(path + [(latest[0]-1, latest[1])])
                
            if check_next(inputs_from_path(path + [(latest[0]+1, latest[1])])) != 0 and \
                (latest[0]+1, latest[1]) not in visited + sum(new_paths, []):
                new_paths.append(path + [(latest[0]+1, latest[1])])
                
            if check_next(inputs_from_path(path + [(latest[0], latest[1]-1)])) != 0 and \
                (latest[0], latest[1]-1) not in visited + sum(new_paths, []):
                new_paths.append(path + [(latest[0], latest[1]-1)])
                
            if check_next(inputs_from_path(path + [(latest[0], latest[1]+1)])) != 0 and \
                (latest[0], latest[1]+1) not in visited + sum(new_paths, []):
                new_paths.append(path + [(latest[0], latest[1]+1)])
                
        if len(new_paths) == 0:
            break
            
        paths = new_paths
        
        curr_output = [check_next(inputs_from_path(x)) for x in paths]
        if 2 in curr_output and not found_oxygen:
            found_oxygen = True
            oxygen_loc = paths[curr_output.index(2)][-1]
            min_steps_to_oxygen = len(paths[0])-1
            
            
    open_spaces = all_visited
    start_point = oxygen_loc
    
    if visualize:
        history = []
        
        rows = [x[0] for x in open_spaces]
        cols = [x[1] for x in open_spaces]
    
        min_rows = min(rows)
        max_rows = max(rows)
        min_cols = min(cols)
        max_cols = max(cols)
    
        space = np.zeros([max_rows-min_rows+1, max_cols-min_cols+1])
        for r,c in open_spaces:
            space[r-min_rows,c-min_cols] = 1
        space[oxygen_loc[0]-min_rows, oxygen_loc[1]-min_cols] = 2
        
        history.append(space.copy())
    
    paths = [[start_point]]
    minutes = 0
    while True:
        new_paths = []
        visited = sum(paths, [])
        for path in paths:
            latest = path[-1]
            if (latest[0]-1, latest[1]) in open_spaces and \
                (latest[0]-1, latest[1]) not in visited + sum(new_paths, []):
                new_paths.append(path + [(latest[0]-1, latest[1])])
                
            if (latest[0]+1, latest[1]) in open_spaces and \
                (latest[0]+1, latest[1]) not in visited + sum(new_paths, []):
                new_paths.append(path + [(latest[0]+1, latest[1])])
                
            if (latest[0], latest[1]-1) in open_spaces and \
                (latest[0], latest[1]-1) not in visited + sum(new_paths, []):
                new_paths.append(path + [(latest[0], latest[1]-1)])
                
            if (latest[0], latest[1]+1) in open_spaces and \
                (latest[0], latest[1]+1) not in visited + sum(new_paths, []):
                new_paths.append(path + [(latest[0], latest[1]+1)])
                
        if len(new_paths) == 0:
            break
        
        if visualize:
            for r,c in [x[-1] for x in new_paths]:
                space[r-min_rows, c-min_cols] = 2
            history.append(space.copy())
        
        minutes += 1
        paths = new_paths
        
    minutes_to_fill = minutes
    
    if visualize:
        fig, ax = plt.subplots()
        ims = []
        for i in range(len(history)):
            im = ax.imshow(history[i], animated=True)
            ims.append([im])
        
        ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True, repeat_delay=10)
        
        
    if visualize:
        return min_steps_to_oxygen, minutes_to_fill, ani
    else:
        return min_steps_to_oxygen, minutes_to_fill
    
#%%
# Day 16: Flawed Frequency Transmission

@time_this_func
def day16():
    with open("input16.txt") as f:
        start_nums = [int(x) for x in f.read().strip()]
    start_nums = np.array(start_nums)
    
    def get_pattern(ind, nums):
        length = len(nums)
        pos = ind+1
        p1 = 0*np.ones(pos)
        p2 = 1*np.ones(pos)
        p3 = 0*np.ones(pos)
        p4 = -1*np.ones(pos)
        pattern = np.hstack([p1,p2,p3,p4])
        reps = int(np.ceil((length+1)/(4*pos)))
        pattern = np.tile(pattern, reps)
        pattern = pattern[1:length+1]
        return pattern
    
    def get_next_num(ind, nums):
        pattern = get_pattern(ind, nums)
        next_num = int(str(int(sum(pattern*nums)))[-1])
        return next_num
    
    def get_next(nums):
        next_nums = []
        for i in range(len(nums)):
            next_nums.append(get_next_num(i, nums))
        return np.array(next_nums)
            
    nums = start_nums.copy()
    for _ in range(100):
        nums = get_next(nums)
    
    first_eight = int("".join([str(x) for x in nums[:8]]))
    
    
    def p2_get_next(nums):
        rev_nums = nums[::-1]
        
        rev_next_nums = np.cumsum(rev_nums)
        rev_next_nums =  [int(str(x)[-1]) for x in rev_next_nums]
        return rev_next_nums[::-1]
    
    nums = list(np.tile(start_nums.copy(), 10000))
    message_offset = int("".join([str(x) for x in nums[:7]]))
    nums = nums[message_offset:]
    for _ in range(100):
        nums = p2_get_next(nums)
        
    real_eight = int("".join([str(x) for x in nums[:8]]))
    
    
    return first_eight, real_eight

#%%
# Day 17: Set and Forget

@time_this_func
def day17():
    import re
    
    with open("input17.txt") as f:
        ints = tuple(int(x) for x in f.read().split(","))
    
    def next_op(i, ints_list, input_vals, relative_base):
        needs_input = False
        output = None
        inst = str(ints_list[i])
        opcode = int(inst[-2:])
        
        modes = [int(x) for x in inst[:-2][::-1]]      
        modes = modes + [0]*(3-len(modes)) 
        
        max_accessible_ind = max([(i+1) * (modes[0] == 1), \
                                  (i+2) * (modes[1] == 1), \
                                  (i+3) * (modes[2] == 1), \
                                  (ints_list[i+1]) * (modes[0] == 0), \
                                  (ints_list[i+2]) * (modes[1] == 0 and opcode in {1,2,5,6,7,8}), \
                                  (ints_list[i+3]) * (modes[2] == 0 and opcode in {1,2,7,8}), \
                                  (relative_base + ints_list[i+1]) * (modes[0] == 2), \
                                  (relative_base + ints_list[i+2]) * (modes[1] == 2), \
                                  (relative_base + ints_list[i+3]) * (modes[2] == 2)])
          
        if max_accessible_ind > len(ints_list)-1:
            ints_list += [0]*(max_accessible_ind-(len(ints_list)-1))      
        
        if opcode == 1: 
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            ints_list[write_to_ind] = input1 + input2
                
            i += 4
            
        if opcode == 2:
    
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
              
            ints_list[write_to_ind] = input1 * input2
                
            i += 4
        
        elif opcode == 3:
            
            if len(input_vals) == 0:
                needs_input = True
                return i, output, relative_base, needs_input
            
            if modes[0] == 0:
                ints_list[ints_list[i+1]] = input_vals.pop(0)
            elif modes[0] == 2:
                ints_list[ints_list[i+1]+relative_base] = input_vals.pop(0)
        
            i += 2
        
        elif opcode == 4:
            
            if modes[0] == 0:
                output = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                output = ints_list[i+1]
            elif modes[0] == 2:
                output = ints_list[ints_list[i+1]+relative_base]
                
            i += 2
            
        elif opcode == 5:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 != 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 6:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 == 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 7:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 < input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4
            
        elif opcode == 8:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 == input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4  
            
        elif opcode == 9:
    
            if modes[0] == 0:
                relative_base += ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                relative_base += ints_list[i+1]
            elif modes[0] == 2:
                relative_base += ints_list[relative_base + ints_list[i+1]]
            
            i += 2
            
        elif opcode == 99:
            i = -1
        
        return i, output, relative_base, needs_input
    
    i = 0
    relative_base = 0
    ints_list = list(ints)
    layout = []
    input_vals = []
    while True:
        i, output, relative_base, needs_input = next_op(i, ints_list, input_vals, relative_base)
        
        if needs_input or i == -1:
            break
        
        if output != None:
            layout.append(output)
            
    layout_str = "".join([chr(x) for x in layout])
            
    layout = np.array([[x for x in y] for y in layout_str.split()])
    layout = np.pad(layout, 1, "constant", constant_values = ".")
    
    intersections = []
    for i in range(1, layout.shape[0]-1):
        for j in range(1, layout.shape[1]-1):
            if layout[i,j] in ["<",">","^","v"]:
                loc = (i,j)
                if layout[i,j] == "<":
                    facing = "W"
                elif layout[i,j] == ">":
                    facing = "E"
                elif layout[i,j] == "^":
                    facing = "N"
                elif layout[i,j] == "<":
                    facing = "S"
                
            elif layout[i,j] == "#":
                if layout[i-1,j] == layout[i+1,j] == layout[i,j-1] == layout[i,j+1] == "#":
                    intersections.append([i-1, j-1])
                
    alignment = sum([x[0]*x[1] for x in intersections])
    
    
    full_movement = []
    last_turn = None
    steps = 0
    while True:
        if facing == "N":
            if layout[loc[0]-1, loc[1]] == "#":
                steps += 1
                loc = (loc[0]-1, loc[1])
            else:
                if last_turn != None:
                    full_movement.append([last_turn, str(steps)])
                steps = 0
                if layout[loc[0], loc[1]-1] == "#":
                    last_turn = "L"
                    facing = "W"
                elif layout[loc[0], loc[1]+1] == "#":
                    last_turn = "R"
                    facing = "E"
                else:
                    break
        
        if facing == "S":
            if layout[loc[0]+1, loc[1]] == "#":
                steps += 1
                loc = (loc[0]+1, loc[1])
            else:
                if last_turn != None:
                    full_movement.append([last_turn, str(steps)])
                steps = 0
                if layout[loc[0], loc[1]-1] == "#":
                    last_turn = "R"
                    facing = "W"
                elif layout[loc[0], loc[1]+1] == "#":
                    last_turn = "L"
                    facing = "E"
                else:
                    break
                
        if facing == "W":
            if layout[loc[0], loc[1]-1] == "#":
                steps += 1
                loc = (loc[0], loc[1]-1)
            else:
                if last_turn != None:
                    full_movement.append([last_turn, str(steps)])
                steps = 0
                if layout[loc[0]-1, loc[1]] == "#":
                    last_turn = "R"
                    facing = "N"
                elif layout[loc[0]+1, loc[1]] == "#":
                    last_turn = "L"
                    facing = "S"
                else:
                    break
                        
        if facing == "E":
            if layout[loc[0], loc[1]+1] == "#":
                steps += 1
                loc = (loc[0], loc[1]+1)
            else:
                if last_turn != None:
                    full_movement.append([last_turn, str(steps)])
                steps = 0
                if layout[loc[0]-1, loc[1]] == "#":
                    last_turn = "L"
                    facing = "N"
                elif layout[loc[0]+1, loc[1]] == "#":
                    last_turn = "R"
                    facing = "S"
                else:
                    break
    
    pattern = [None]*len(full_movement)
    pat_num = -1
    for i in range(len(pattern)):
        if pattern[i] != None:
            continue
        
        pat_num += 1
        for j in range(i, len(pattern)):
            if full_movement[j] == full_movement[i]:
                pattern[j] = pat_num
                
    pattern = "".join([str(x) for x in pattern])
    
    group_pattern = pattern
    groups = {}
    letters = "ABC"
    letter_ind = -1
    for i in range(len(group_pattern)):
        if group_pattern[i] in {"A","B","C","_"}:
            continue
        
        for group_len in range(len(group_pattern)-i, 0, -1):
            if "_" in group_pattern[i:i+group_len]:
                continue
            
            if len(re.findall(group_pattern[i:i+group_len], group_pattern)) > 1:
                letter_ind += 1
                groups[letters[letter_ind]] = full_movement[i:i+group_len]
                group_pattern = re.sub(group_pattern[i:i+group_len], letters[letter_ind]+"_"*(group_len-1), group_pattern)
                break
    
    group_pattern = ",".join([x for x in group_pattern.replace("_","")]) + "\n"
    group_A = ",".join(sum(groups["A"], [])) + "\n"
    group_B = ",".join(sum(groups["B"], [])) + "\n"
    group_C = ",".join(sum(groups["C"], [])) + "\n"
    
    full_instructions = group_pattern + group_A + group_B + group_C + "n" + "\n"
    input_vals = [ord(x) for x in full_instructions]
    
    i = 0
    relative_base = 0
    ints_list = list(ints)
    ints_list[0] = 2
    while True:
        i, output, relative_base, needs_input = next_op(i, ints_list, input_vals, relative_base)
        
        if i == -1:
            break
        
        if output != None:
            dust_collected = output
            
            
    return alignment, dust_collected

#%%
# Day 18: Many-Worlds Interpretation

@time_this_func
def day18(verbose = False):
    layout = []
    start_not_found = True
    with open("input18.txt") as f:
        for i, l in enumerate(f):
            new = l.strip()
            if start_not_found:
                for j, c in enumerate(new):
                    if c == "@":
                        loc = (i,j)
                        new = new.replace("@",".")
                        start_not_found = False
            layout.append([x for x in new])
    layout = np.array(layout)
           
    def get_options(cur_loc, cur_layout, partitioned = False):
        letter_options = []
        shortest_paths = []
    
        paths = [[cur_loc]]
        visited = {cur_loc}
        while True:
            new_paths = []
            for path in paths:
                latest = path[-1]
                next_step_options = [(latest[0]-1, latest[1]),(latest[0]+1, latest[1]),(latest[0], latest[1]-1),(latest[0], latest[1]+1)]
                for next_step_option in next_step_options:
                    
                    if not partitioned:
                        not_allowed = visited
                    else:
                        not_allowed = visited | new_blocked
                    
                    if next_step_option not in not_allowed:
                        if cur_layout[next_step_option] == ".":
                            new_paths.append(path + [next_step_option])
                            visited.add(next_step_option)
                        elif cur_layout[next_step_option].islower():
                            letter_options.append(str(cur_layout[next_step_option]))
                            shortest_paths.append(path + [next_step_option])
            
            if len(new_paths) == 0:
                return letter_options, shortest_paths
            
            paths = new_paths
            
    def retrieve_key(letter_key, path_to, cur_layout):
        cur_layout = cur_layout.copy()
        
        num_steps = len(path_to)-1
        new_loc = path_to[-1]
        cur_layout[cur_layout == letter_key.lower()] = "."
        cur_layout[cur_layout == letter_key.upper()] = "."
        
        return num_steps, new_loc, cur_layout
    
    past_layouts_and_routes = {}
    orders = ["@"]
    locs = [loc]
    layouts = [layout.copy()]
    step_nums = [0]
    
    while True:
        new_orders = []
        new_locs = []
        new_layouts = []
        new_steps = []
        
        for last_loc, last_layout, order, step_num in zip(locs, layouts, orders, step_nums):
            last_letter = order[-1]
            
            hashifiable_last_layout = "".join(last_layout.ravel())
            if hashifiable_last_layout in past_layouts_and_routes and last_letter in past_layouts_and_routes[hashifiable_last_layout]:
                next_letters = past_layouts_and_routes[hashifiable_last_layout][last_letter]
                paths_to_letters = past_layouts_and_routes[hashifiable_last_layout][last_letter].values()
                
            else:
                next_letters, paths_to_letters = get_options(last_loc, last_layout)
                if hashifiable_last_layout not in past_layouts_and_routes:
                    past_layouts_and_routes[hashifiable_last_layout] = {last_letter:dict(zip(next_letters, paths_to_letters))}
                else:
                    past_layouts_and_routes[hashifiable_last_layout][last_letter] = dict(zip(next_letters, paths_to_letters))
                
            for next_letter, path_to_letter in zip(next_letters, paths_to_letters):
                next_steps, next_loc, next_layout = retrieve_key(next_letter, path_to_letter, last_layout)
                
                new_orders.append(order + next_letter)
                new_locs.append(next_loc)
                new_layouts.append(next_layout.copy())
                new_steps.append(step_num + next_steps)
                
        if len(new_orders) == 0:
            break
        
        new_steps, new_orders, new_locs, new_layouts = zip(*sorted(zip(new_steps, new_orders, new_locs, new_layouts)))
        
        states = set()
        
        orders = []
        locs = []
        layouts = []
        step_nums = []
        for new_order, new_loc, new_layout, new_step in zip(new_orders, new_locs, new_layouts, new_steps):
            state = "".join(sorted(new_order[:-1])) + new_order[-1]
            if state not in states:
                orders.append(new_order)
                locs.append(new_loc)
                layouts.append(new_layout)
                step_nums.append(new_step)
                
                states.add(state)
                
        if verbose:
            print(f"{len(orders)} ways of quickly getting {len(orders[0])-1} keys")
        
    min_steps = min(step_nums)


    middle_row = layout.shape[0]//2
    middle_col = layout.shape[1]//2
    new_blocked = set()
    for r in range(middle_row-1, middle_row+2):
        for c in range(middle_col-1, middle_col+2):
            if ((r == middle_row) ^ (c == middle_col)) or ( r == middle_row and c == middle_col):
                new_blocked.add((r,c))
    
    loc0 = (layout.shape[0]//2-1, layout.shape[1]//2-1)
    loc1 = (layout.shape[0]//2-1, layout.shape[1]//2+1)
    loc2 = (layout.shape[0]//2+1, layout.shape[1]//2-1)
    loc3 = (layout.shape[0]//2+1, layout.shape[1]//2+1)
    
    keys0 = {str(x) for x in layout.copy()[0:layout.shape[0]//2+1,0:layout.shape[1]//2+1].ravel() if x.islower()}
    keys1 = {str(x) for x in layout.copy()[0:layout.shape[0]//2+1,layout.shape[1]//2:].ravel() if x.islower()}
    keys2 = {str(x) for x in layout.copy()[layout.shape[0]//2:,0:layout.shape[1]//2+1].ravel() if x.islower()}
    keys3 = {str(x) for x in layout.copy()[layout.shape[0]//2:,layout.shape[1]//2:].ravel() if x.islower()}
    
    letter_keys = [keys0, keys1, keys2, keys3]
    
    orders = [["0","1","2","3"]]
    locs = [[loc0,loc1,loc2,loc3]]
    layouts = [layout.copy()]
    step_nums = [[0]*4]
    
    if verbose:
        print("\n")
        
    while True:
        new_orders = []
        new_locs = []
        new_layouts = []
        new_steps = []
        
        for last_loc, last_layout, order, step_num in zip(locs, layouts, orders, step_nums):
            
            last_letter = [x[-1] for x in order]
            
            hashifiable_last_layout = "".join(last_layout.ravel())
            
            next_letters = []
            paths_to_letters = []
            for i in range(4):
                if hashifiable_last_layout in past_layouts_and_routes and last_letter[i] in past_layouts_and_routes[hashifiable_last_layout]:
                    next_lettersi = []
                    paths_to_lettersi = []
                    for next_letter_option in past_layouts_and_routes[hashifiable_last_layout][last_letter[i]]:
                        if next_letter_option in letter_keys[i]:
                            next_lettersi.append(next_letter_option)
                            paths_to_lettersi.append(past_layouts_and_routes[hashifiable_last_layout][last_letter[i]][next_letter_option])
                    
                else:
                    next_lettersi, paths_to_lettersi = get_options(last_loc[i], last_layout, True)
                    if hashifiable_last_layout not in past_layouts_and_routes:
                        past_layouts_and_routes[hashifiable_last_layout] = {last_letter[i]:dict(zip(next_lettersi, paths_to_lettersi))}
                    else:
                        past_layouts_and_routes[hashifiable_last_layout][last_letter[i]] = dict(zip(next_lettersi, paths_to_lettersi))    
                        
                next_letters.append(next_lettersi)
                paths_to_letters.append(paths_to_lettersi)
            
            for i in range(4):
                if len(next_letters[i]) == 0:
                    next_letters[i] = next_letters[i] + [last_letter[i]]
                    paths_to_letters[i] = paths_to_letters[i] + [[last_loc[i]]]
                
            added_new = False
            for next_letter0, path_to_letter0 in zip(next_letters[0], paths_to_letters[0]):
                for next_letter1, path_to_letter1 in zip(next_letters[1], paths_to_letters[1]):
                    for next_letter2, path_to_letter2 in zip(next_letters[2], paths_to_letters[2]):
                        for next_letter3, path_to_letter3 in zip(next_letters[3], paths_to_letters[3]):
                            next_steps0, next_loc0, next_layout = retrieve_key(next_letter0, path_to_letter0, last_layout)
                            next_steps1, next_loc1, next_layout = retrieve_key(next_letter1, path_to_letter1, next_layout)
                            next_steps2, next_loc2, next_layout = retrieve_key(next_letter2, path_to_letter2, next_layout)
                            next_steps3, next_loc3, next_layout = retrieve_key(next_letter3, path_to_letter3, next_layout)
                            
                            order_to_add = []
                            if next_letter0 != order[0][-1]:
                                order_to_add.append(order[0] + next_letter0)
                                added_new = True
                            else:
                                order_to_add.append(order[0])
                            if next_letter1 != order[1][-1]:
                                order_to_add.append(order[1] + next_letter1)
                                added_new = True
                            else:
                                order_to_add.append(order[1])
                            if next_letter2 != order[2][-1]:
                                order_to_add.append(order[2] + next_letter2)
                                added_new = True
                            else:
                                order_to_add.append(order[2])
                            if next_letter3 != order[3][-1]:
                                order_to_add.append(order[3] + next_letter3)
                                added_new = True
                            else:
                                order_to_add.append(order[3])  
                            
                            new_orders.append(order_to_add)
                            new_locs.append([next_loc0, next_loc1, next_loc2, next_loc3])
                            new_layouts.append(next_layout.copy())
                            new_steps.append([step_num[0] + next_steps0, step_num[1] + next_steps1, step_num[2] + next_steps2, step_num[3] + next_steps3])       
                
        if not added_new:
            break
        
        new_steps, new_orders, new_locs, new_layouts = zip(*sorted(zip(new_steps, new_orders, new_locs, new_layouts), key = lambda x: sum(x[0])))
        
        states = set()
        
        orders = []
        locs = []
        layouts = []
        step_nums = []
        for new_order, new_loc, new_layout, new_step in zip(new_orders, new_locs, new_layouts, new_steps):
            state0 = "".join(sorted(new_order[0][:-1])) + new_order[0][-1]
            state1 = "".join(sorted(new_order[1][:-1])) + new_order[1][-1]
            state2 = "".join(sorted(new_order[2][:-1])) + new_order[2][-1]
            state3 = "".join(sorted(new_order[3][:-1])) + new_order[3][-1]
            state = state0 + state1 + state2 + state3
            if state not in states:
                orders.append(new_order)
                locs.append(new_loc)
                layouts.append(new_layout)
                step_nums.append(new_step)
                
                states.add(state)
                
        if verbose:
            print(f"{len(orders)} ways of quickly getting {sum([len(x)-1 for x in orders[0]])} keys")
    
    total_steps = [sum(x) for x in step_nums]
    min_steps_2 = min(total_steps)
    
    if verbose:
        print("\n")
        
        
    return min_steps, min_steps_2

#%%
# Day 19: Tractor Beam

@time_this_func
def day19():
    with open("input19.txt") as f:
        ints = tuple(int(x) for x in f.read().split(","))
    
    def next_op(i, ints_list, input_vals, relative_base):
        needs_input = False
        output = None
        inst = str(ints_list[i])
        opcode = int(inst[-2:])
        
        modes = [int(x) for x in inst[:-2][::-1]]      
        modes = modes + [0]*(3-len(modes)) 
        
        max_accessible_ind = max([(i+1) * (modes[0] == 1), \
                                  (i+2) * (modes[1] == 1), \
                                  (i+3) * (modes[2] == 1), \
                                  (ints_list[i+1]) * (modes[0] == 0), \
                                  (ints_list[i+2]) * (modes[1] == 0 and opcode in {1,2,5,6,7,8}), \
                                  (ints_list[i+3]) * (modes[2] == 0 and opcode in {1,2,7,8}), \
                                  (relative_base + ints_list[i+1]) * (modes[0] == 2), \
                                  (relative_base + ints_list[i+2]) * (modes[1] == 2), \
                                  (relative_base + ints_list[i+3]) * (modes[2] == 2)])
          
        if max_accessible_ind > len(ints_list)-1:
            ints_list += [0]*(max_accessible_ind-(len(ints_list)-1))      
        
        if opcode == 1: 
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            ints_list[write_to_ind] = input1 + input2
                
            i += 4
            
        if opcode == 2:
    
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
              
            ints_list[write_to_ind] = input1 * input2
                
            i += 4
        
        elif opcode == 3:
            
            if len(input_vals) == 0:
                needs_input = True
                return i, output, relative_base, needs_input
            
            if modes[0] == 0:
                ints_list[ints_list[i+1]] = input_vals.pop(0)
            elif modes[0] == 2:
                ints_list[ints_list[i+1]+relative_base] = input_vals.pop(0)
        
            i += 2
        
        elif opcode == 4:
            
            if modes[0] == 0:
                output = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                output = ints_list[i+1]
            elif modes[0] == 2:
                output = ints_list[ints_list[i+1]+relative_base]
                
            i += 2
            
        elif opcode == 5:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 != 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 6:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 == 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 7:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 < input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4
            
        elif opcode == 8:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 == input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4  
            
        elif opcode == 9:
    
            if modes[0] == 0:
                relative_base += ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                relative_base += ints_list[i+1]
            elif modes[0] == 2:
                relative_base += ints_list[relative_base + ints_list[i+1]]
            
            i += 2
            
        elif opcode == 99:
            i = -1
        
        return i, output, relative_base, needs_input
    
    def is_tractored(x, y):
        input_vals = [x,y]
        i = 0
        relative_base = 0
        ints_list = list(ints)
        while True:
            i, output, relative_base, needs_input = next_op(i, ints_list, input_vals, relative_base)
            
            if i == -1:
                break
            
            if output != None:
                is_being_pulled = output
                
        return bool(is_being_pulled)
    
    space = np.zeros([50,50])
    for x in range(space.shape[1]):
        for y in range(space.shape[0]):
            if y > 1 and space[y-1,x] == False and True in space[:y-1,x]:
                continue
            if x > 1 and space[y,x-1] == False and True in space[y,:x-1]:
                continue
            
            space[y,x] = is_tractored(x, y)
            
    tractored_in_50 = int(np.sum(space))
    
    
    def scan_more(prev_space):
        prev_y = prev_space.shape[0]
        prev_x = prev_space.shape[1]
        
        first_pulled_col_in_last = np.where(prev_space[-1,:] == True)[0]
        if len(first_pulled_col_in_last) > 0:
            first_pulled_col_in_last = first_pulled_col_in_last[0]
        first_pulled_row_in_last = np.where(prev_space[:,-1] == True)[0][0]
            
        new_col = np.zeros([prev_y,1])
        for y in range(first_pulled_row_in_last,prev_y):
            new_col[y,0] = is_tractored(prev_x,y)
            
        new_space = np.hstack([prev_space,new_col])
        
        if True in prev_space[-1,:]: 
            new_row = np.zeros([1,prev_x+1])
            for x in range(first_pulled_col_in_last,prev_x+1):
                new_row[0,x] = is_tractored(x,prev_y)
            
            new_space = np.vstack([new_space,new_row])
        
        return new_space
    
    def ship_fits(check_space, ship_size = 100):        
        y = np.where(check_space[:,-1] == True)[0][0]
        if np.sum(check_space[y,:]) < ship_size:
            return (False, None)
            
        x = np.where(check_space[y,:] == 1)[0][-ship_size]
        if np.sum(check_space[y:,x]) >= ship_size:
            return (True, (x,y))
        else:
            return (False, None)
    
    ship_space = space.copy()
    while True:
        ship_space = scan_more(ship_space)
        result, upper_left_loc = ship_fits(ship_space)
        
        if result:
            break
    
    upper_left = 10000*upper_left_loc[0] + upper_left_loc[1]
    
    
    return tractored_in_50, upper_left

#%%
# Day 20: Donut Maze

@time_this_func
def day20():
    layout = []
    with open("input20.txt") as f:
        for l in f:
            layout.append([x for x in l[:-1]])
    layout = np.array(layout, dtype = object)
    
    gates = {}
    
    row = 2
    for col in range(2,128+1):
        if layout[row,col] == ".":
            layout[row,col] = layout[row-2,col] + layout[row-1,col]
            if layout[row-2,col] + layout[row-1,col] in gates:
                gates[layout[row-2,col] + layout[row-1,col]].append((row,col))
            else:
                gates[layout[row-2,col] + layout[row-1,col]] = [(row,col)]
            layout[row-1,col] = "#"
                
    row = 92
    for col in range(37,93+1):
        if layout[row,col] == ".":
            layout[row,col] = layout[row-2,col] + layout[row-1,col]
            if layout[row-2,col] + layout[row-1,col] in gates:
                gates[layout[row-2,col] + layout[row-1,col]].append((row,col))
            else:
                gates[layout[row-2,col] + layout[row-1,col]] = [(row,col)]
            layout[row-1,col] = "#"
                
    col = 2
    for row in range(2,126+1):
        if layout[row,col] == ".":
            layout[row,col] = layout[row,col-2] + layout[row,col-1]
            if layout[row,col-2] + layout[row,col-1] in gates:
                gates[layout[row,col-2] + layout[row,col-1]].append((row,col))
            else:
                gates[layout[row,col-2] + layout[row,col-1]] = [(row,col)]
            layout[row,col-1] = "#"
                
    col = 94
    for row in range(37,91+1):
        if layout[row,col] == ".":
            layout[row,col] = layout[row,col-2] + layout[row,col-1]
            if layout[row,col-2] + layout[row,col-1] in gates:
                gates[layout[row,col-2] + layout[row,col-1]].append((row,col))
            else:
                gates[layout[row,col-2] + layout[row,col-1]] = [(row,col)]
            layout[row,col-1] = "#"
                
    row = 126
    for col in range(2,128+1):
        if layout[row,col] == ".":
            layout[row,col] = layout[row+1,col] + layout[row+2,col]
            if layout[row+1,col] + layout[row+2,col] in gates:
                gates[layout[row+1,col] + layout[row+2,col]].append((row,col))
            else:
                gates[layout[row+1,col] + layout[row+2,col]] = [(row,col)]
            
            layout[row+1,col] = "#"
                
    row = 36
    for col in range(37,93+1):
        if layout[row,col] == ".":
            layout[row,col] = layout[row+1,col] + layout[row+2,col]
            if layout[row+1,col] + layout[row+2,col] in gates:
                gates[layout[row+1,col] + layout[row+2,col]].append((row,col))
            else:
                gates[layout[row+1,col] + layout[row+2,col]] = [(row,col)]
            layout[row+1,col] = "#"
                
    col = 128
    for row in range(2,128+1):
        if layout[row,col] == ".":
            layout[row,col] = layout[row,col+1] + layout[row,col+2]
            if layout[row,col+1] + layout[row,col+2] in gates:
                gates[layout[row,col+1] + layout[row,col+2]].append((row,col))
            else:
                gates[layout[row,col+1] + layout[row,col+2]] = [(row,col)]
            layout[row,col+1] = "#"
                
    col = 36
    for row in range(37,93+1):
        if layout[row,col] == ".":
            layout[row,col] = layout[row,col+1] + layout[row,col+2]
            if layout[row,col+1] + layout[row,col+2] in gates:
                gates[layout[row,col+1] + layout[row,col+2]].append((row,col))
            else:
                gates[layout[row,col+1] + layout[row,col+2]] = [(row,col)]
            layout[row,col+1] = "#"
                
    start = gates["AA"][0]
    paths = [[start]]
    visited = {start}
    while True:
        new_paths = []
        for path in paths:
            latest = path[-1]
            options = [(latest[0]-1,latest[1]), (latest[0]+1, latest[1]), (latest[0], latest[1]-1), (latest[0], latest[1]+1)]
            if layout[latest] not in  {".", "AA"}:
                gate_locs = gates[layout[latest]]
                go_to = gate_locs[(gate_locs.index(latest) + 1)%2]
                options.append(go_to)
            
            for option in options:
                if option not in visited and layout[option] != "#":
                    new_paths.append(path + [option])
                    visited.add(option)
                    
        paths = new_paths
           
        if gates["ZZ"][0] in {x[-1] for x in paths}:
            break
    
    min_steps = len(paths[0])-1
    
    
    paths = [[(start, 0)]]
    visited = {(start, 0)}
    while True:
        new_paths = []
        for path in paths:
            latest = path[-1]
            
            options = [(latest[0][0]-1,latest[0][1]), (latest[0][0]+1, latest[0][1]), (latest[0][0], latest[0][1]-1), (latest[0][0], latest[0][1]+1)]
            for option in options:
                if (option, latest[1]) not in visited and layout[option] != "#":
                    new_paths.append(path + [(option, latest[1])])
                    visited.add((option, latest[1]))
                
            if layout[latest[0]] not in {".", "AA", "ZZ"}:
                gate_locs = gates[layout[latest[0]]]
                go_to = gate_locs[(gate_locs.index(latest[0]) + 1)%2]
                if (latest[0][0] in {2,126} or latest[0][1] in {2,128}) and (go_to, latest[1]-1) not in visited and latest[1] != 0:
                    new_paths.append(path + [(go_to, latest[1]-1)])
                    visited.add((go_to, latest[1]-1))
                elif (latest[0][0] in {36,92} or latest[0][1] in {36,94}) and (go_to, latest[1]+1) not in visited:
                    new_paths.append(path + [(go_to, latest[1]+1)])
                    visited.add((go_to, latest[1]+1))
        
        paths = new_paths
        
        if (gates["ZZ"][0], 0) in {x[-1] for x in paths}:
            break
    
    min_steps_recursive = len(paths[0])-1
    
    
    return min_steps, min_steps_recursive

#%%
# Day 21: Springdroid Adventure

@time_this_func
def day21():
    with open("input21.txt") as f:
        ints = tuple(int(x) for x in f.read().split(","))
    
    def next_op(i, ints_list, input_vals, relative_base):
        needs_input = False
        output = None
        inst = str(ints_list[i])
        opcode = int(inst[-2:])
        
        modes = [int(x) for x in inst[:-2][::-1]]      
        modes = modes + [0]*(3-len(modes)) 
        
        max_accessible_ind = max([(i+1) * (modes[0] == 1), \
                                  (i+2) * (modes[1] == 1), \
                                  (i+3) * (modes[2] == 1), \
                                  (ints_list[i+1]) * (modes[0] == 0), \
                                  (ints_list[i+2]) * (modes[1] == 0 and opcode in {1,2,5,6,7,8}), \
                                  (ints_list[i+3]) * (modes[2] == 0 and opcode in {1,2,7,8}), \
                                  (relative_base + ints_list[i+1]) * (modes[0] == 2), \
                                  (relative_base + ints_list[i+2]) * (modes[1] == 2), \
                                  (relative_base + ints_list[i+3]) * (modes[2] == 2)])
          
        if max_accessible_ind > len(ints_list)-1:
            ints_list += [0]*(max_accessible_ind-(len(ints_list)-1))      
        
        if opcode == 1: 
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            ints_list[write_to_ind] = input1 + input2
                
            i += 4
            
        if opcode == 2:
    
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
              
            ints_list[write_to_ind] = input1 * input2
                
            i += 4
        
        elif opcode == 3:
            
            if len(input_vals) == 0:
                needs_input = True
                return i, output, relative_base, needs_input
            
            if modes[0] == 0:
                ints_list[ints_list[i+1]] = input_vals.pop(0)
            elif modes[0] == 2:
                ints_list[ints_list[i+1]+relative_base] = input_vals.pop(0)
        
            i += 2
        
        elif opcode == 4:
            
            if modes[0] == 0:
                output = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                output = ints_list[i+1]
            elif modes[0] == 2:
                output = ints_list[ints_list[i+1]+relative_base]
                
            i += 2
            
        elif opcode == 5:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 != 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 6:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 == 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 7:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 < input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4
            
        elif opcode == 8:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 == input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4  
            
        elif opcode == 9:
    
            if modes[0] == 0:
                relative_base += ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                relative_base += ints_list[i+1]
            elif modes[0] == 2:
                relative_base += ints_list[relative_base + ints_list[i+1]]
            
            i += 2
            
        elif opcode == 99:
            i = -1
    
        return i, output, relative_base, needs_input
    
    def make_ascii(instructions_list):
        string = "\n".join(instructions_list)
        ascii_list = []
        for c in string:
            ascii_list.append(ord(c))
        ascii_list.append(ord("\n"))
        return ascii_list
            
    instructions = ["NOT A T", "NOT B J", "OR T J", "NOT C T", "OR T J", "AND D J", "WALK"]
    input_vals = make_ascii(instructions)
    
    i = 0
    relative_base = 0
    ints_list = list(ints)
    while True:
        i, output, relative_base, needs_input = next_op(i, ints_list, input_vals, relative_base)
            
        if i == -1:
            break
        
        if output != None:
            total_damage1 = output
            
    
    instructions = ["NOT B J", "NOT C T", "OR T J", "AND D J", "AND H J", "NOT A T", "OR T J", "RUN"]
    input_vals = make_ascii(instructions)
    
    i = 0
    relative_base = 0
    ints_list = list(ints)
    while True:
        i, output, relative_base, needs_input = next_op(i, ints_list, input_vals, relative_base)
            
        if i == -1:
            break
        
        if output != None:
            total_damage2 = output
            
            
    return total_damage1, total_damage2
#%%
# Day 22: Slam Shuffle

@time_this_func
def day22():
    operations = []
    with open("input22.txt") as f:
        for l in f:
            if "deal into new stack" in l:
                operations.append(("new",))
            elif "deal with increment" in l:
                operations.append(("increment",int(l.split()[-1])))
            elif "cut" in l:
                operations.append(("cut",int(l.split()[-1])))
                
    num_cards = 10007
    cards = list(range(num_cards))
                
    for op in operations:
        if op[0] == "new":
            cards = cards[::-1]
        elif op[0] == "increment":
            new_cards = cards.copy()
            for i,c in enumerate(cards):
                new_cards[(op[1]*i)%num_cards] = c
            cards = new_cards
        elif op[0] == "cut":
            cards = cards[op[1]:] + cards[:op[1]]
            
    pos_2019 = cards.index(2019)
    
    
    # Number theory. Got a lot of help.
    num_cards = 119315717514047
    card0 = 0
    card_diff = 1
    
    num_shuffles = 101741582076661
    
    for op in operations:
        if op[0] == "new":
            card_diff *= -1
            card0 += card_diff
        elif op[0] == "increment":
            card_diff *= pow(op[1], num_cards-2, num_cards)
        elif op[0] == "cut":
            card0 += card_diff * op[1]
        card0 %= num_cards
        card_diff %= num_cards
        
    card_diff_mult_per_shuffle = card_diff
    final_card_diff = pow(card_diff_mult_per_shuffle, num_shuffles, num_cards) % num_cards
    
    card0_add_per_shuffle = card0
    final_card0 = (card0_add_per_shuffle * (1 - pow(card_diff_mult_per_shuffle, num_shuffles, num_cards)) * pow(1-card_diff_mult_per_shuffle, num_cards-2, num_cards)) % num_cards
    
    card2020 = (final_card0 + (2020*final_card_diff))%num_cards
    
    
    return pos_2019, card2020

#%%
# Day 23: Category Six

@time_this_func
def day23():
    with open("input23.txt") as f:
        ints = tuple(int(x) for x in f.read().split(","))
    
    def next_op_network(address):
        i = all_is[address]
        relative_base = relative_bases[address]
        
        output = None
        inst = str(ints_lists[address][i])
        opcode = int(inst[-2:])
        
        modes = [int(x) for x in inst[:-2][::-1]]      
        modes = modes + [0]*(3-len(modes)) 
        
        max_accessible_ind = max([(i+1) * (modes[0] == 1), \
                                  (i+2) * (modes[1] == 1), \
                                  (i+3) * (modes[2] == 1), \
                                  (ints_lists[address][i+1]) * (modes[0] == 0), \
                                  (ints_lists[address][i+2]) * (modes[1] == 0 and opcode in {1,2,5,6,7,8}), \
                                  (ints_lists[address][i+3]) * (modes[2] == 0 and opcode in {1,2,7,8}), \
                                  (relative_base + ints_lists[address][i+1]) * (modes[0] == 2), \
                                  (relative_base + ints_lists[address][i+2]) * (modes[1] == 2), \
                                  (relative_base + ints_lists[address][i+3]) * (modes[2] == 2)])
          
        if max_accessible_ind > len(ints_lists[address])-1:
            ints_lists[address] += [0]*(max_accessible_ind-(len(ints_lists[address])-1))      
        
        if opcode == 1: 
            
            if modes[0] == 0:
                input1 = ints_lists[address][ints_lists[address][i+1]]
            elif modes[0] == 1:
                input1 = ints_lists[address][i+1]
            elif modes[0] == 2:
                input1 = ints_lists[address][relative_base + ints_lists[address][i+1]]
            if modes[1] == 0:
                input2 = ints_lists[address][ints_lists[address][i+2]]
            elif modes[1] == 1:
                input2 = ints_lists[address][i+2]
            elif modes[1] == 2:
                input2 = ints_lists[address][relative_base + ints_lists[address][i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_lists[address][i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_lists[address][i+3]
                
            ints_lists[address][write_to_ind] = input1 + input2
                
            i += 4
            
        if opcode == 2:
    
            if modes[0] == 0:
                input1 = ints_lists[address][ints_lists[address][i+1]]
            elif modes[0] == 1:
                input1 = ints_lists[address][i+1]
            elif modes[0] == 2:
                input1 = ints_lists[address][relative_base + ints_lists[address][i+1]]
            if modes[1] == 0:
                input2 = ints_lists[address][ints_lists[address][i+2]]
            elif modes[1] == 1:
                input2 = ints_lists[address][i+2]
            elif modes[1] == 2:
                input2 = ints_lists[address][relative_base + ints_lists[address][i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_lists[address][i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_lists[address][i+3]
              
            ints_lists[address][write_to_ind] = input1 * input2
                
            i += 4
        
        elif opcode == 3:
            
            if len(all_input_vals[address]) == 0:
                all_input_vals[address].append(-1)
                idling[address] = True
            else:
                idling[address] = False
            
            if modes[0] == 0:
                ints_lists[address][ints_lists[address][i+1]] = all_input_vals[address].pop(0)
            elif modes[0] == 2:
                ints_lists[address][ints_lists[address][i+1]+relative_base] = all_input_vals[address].pop(0)
        
            i += 2
        
        elif opcode == 4:
            
            if modes[0] == 0:
                output = ints_lists[address][ints_lists[address][i+1]]
            elif modes[0] == 1:
                output = ints_lists[address][i+1]
            elif modes[0] == 2:
                output = ints_lists[address][ints_lists[address][i+1]+relative_base]
                
            i += 2
            
        elif opcode == 5:
            
            if modes[0] == 0:
                input1 = ints_lists[address][ints_lists[address][i+1]]
            elif modes[0] == 1:
                input1 = ints_lists[address][i+1]
            elif modes[0] == 2:
                input1 = ints_lists[address][relative_base + ints_lists[address][i+1]]
            if modes[1] == 0:
                input2 = ints_lists[address][ints_lists[address][i+2]]
            elif modes[1] == 1:
                input2 = ints_lists[address][i+2]
            elif modes[1] == 2:
                input2 = ints_lists[address][relative_base + ints_lists[address][i+2]]
                
            if input1 != 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 6:
            
            if modes[0] == 0:
                input1 = ints_lists[address][ints_lists[address][i+1]]
            elif modes[0] == 1:
                input1 = ints_lists[address][i+1]
            elif modes[0] == 2:
                input1 = ints_lists[address][relative_base + ints_lists[address][i+1]]
            if modes[1] == 0:
                input2 = ints_lists[address][ints_lists[address][i+2]]
            elif modes[1] == 1:
                input2 = ints_lists[address][i+2]
            elif modes[1] == 2:
                input2 = ints_lists[address][relative_base + ints_lists[address][i+2]]
                
            if input1 == 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 7:
            
            if modes[0] == 0:
                input1 = ints_lists[address][ints_lists[address][i+1]]
            elif modes[0] == 1:
                input1 = ints_lists[address][i+1]
            elif modes[0] == 2:
                input1 = ints_lists[address][relative_base + ints_lists[address][i+1]]
            if modes[1] == 0:
                input2 = ints_lists[address][ints_lists[address][i+2]]
            elif modes[1] == 1:
                input2 = ints_lists[address][i+2]
            elif modes[1] == 2:
                input2 = ints_lists[address][relative_base + ints_lists[address][i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_lists[address][i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_lists[address][i+3]
                
            if input1 < input2:
                ints_lists[address][write_to_ind] = 1
            else:
                ints_lists[address][write_to_ind] = 0
            
            i += 4
            
        elif opcode == 8:
            
            if modes[0] == 0:
                input1 = ints_lists[address][ints_lists[address][i+1]]
            elif modes[0] == 1:
                input1 = ints_lists[address][i+1]
            elif modes[0] == 2:
                input1 = ints_lists[address][relative_base + ints_lists[address][i+1]]
            if modes[1] == 0:
                input2 = ints_lists[address][ints_lists[address][i+2]]
            elif modes[1] == 1:
                input2 = ints_lists[address][i+2]
            elif modes[1] == 2:
                input2 = ints_lists[address][relative_base + ints_lists[address][i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_lists[address][i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_lists[address][i+3]
                
            if input1 == input2:
                ints_lists[address][write_to_ind] = 1
            else:
                ints_lists[address][write_to_ind] = 0
            
            i += 4  
            
        elif opcode == 9:
    
            if modes[0] == 0:
                relative_base += ints_lists[address][ints_lists[address][i+1]]
            elif modes[0] == 1:
                relative_base += ints_lists[address][i+1]
            elif modes[0] == 2:
                relative_base += ints_lists[address][relative_base + ints_lists[address][i+1]]
            
            i += 2
            
        elif opcode == 99:
            i = -1
            
        all_is[address] = i
        relative_bases[address] = relative_base
    
        return output
    
    ints_lists = [list(ints) for _ in range(50)]
    all_input_vals = [[computer_address] for computer_address in range(50)]
    all_is = [0 for _ in range(50)]
    relative_bases = [0 for _ in range(50)]
    all_outputs = [[] for _ in range(50)]
    idling = [False for _ in range(50)]
    
    first_NAT_y_found = False
    
    last_NAT_y_delivered = None
    NAT_y_doubled = False
    
    NAT = None
    while True:
        for a in range(50):
    
            output = next_op_network(a)
            
            if output != None:
                all_outputs[a].append(output)
                
                if len(all_outputs[a]) == 3:
                    to_address = all_outputs[a].pop(0)
                    x = all_outputs[a].pop(0)
                    y = all_outputs[a].pop(0)
                    
                    if to_address == 255:
                        if not first_NAT_y_found:
                            first_NAT_y_found = True
                            first_y_for_255 = y
                            
                        NAT = (x,y)
                        
                    else:
                        all_input_vals[to_address].append(x)
                        all_input_vals[to_address].append(y)
    
            if sum(idling) == 50 and NAT != None:
                if NAT[1] == last_NAT_y_delivered:
                    NAT_y_doubled = True
                    first_doubled_NAT_y = NAT[1]
                    break
                
                all_input_vals[0].append(NAT[0])
                all_input_vals[0].append(NAT[1])
                
                last_NAT_y_delivered = NAT[1]
                NAT = None
    
        if NAT_y_doubled:
            break
        
    return first_y_for_255, first_doubled_NAT_y

#%%
# Day 24: Planet of Discord

@time_this_func
def day24():
    eris = []
    with open("input24.txt") as f:
        for l in f:
            eris.append([1 if x == "#" else 0 for x in l.strip()])
    original_eris = np.array(eris)
    
    eris = original_eris.copy()
    eris = np.pad(eris, 1, constant_values = 0)
    
    def hashifiable(numpy_array):
        hashified = ""
        for row in numpy_array:
            for x in row:
                hashified += str(x)
        return hashified
    
    seen = {hashifiable(eris)}
    while True:
        new_eris = eris.copy()
        
        for row in range(1, eris.shape[0]-1):
            for col in range(1, eris.shape[1]-1):
                if eris[row, col] == 1:
                    if eris[row-1, col] + eris[row+1, col] + eris[row, col-1] + eris[row, col+1] != 1:
                        new_eris[row, col] = 0
                
                if eris[row, col] == 0:
                    if  eris[row-1, col] + eris[row+1, col] + eris[row, col-1] + eris[row, col+1] in {1,2}:
                        new_eris[row, col] = 1
            
        eris = new_eris
            
        hashifiable_eris = hashifiable(eris)
        
        if hashifiable_eris not in seen:
            seen.add(hashifiable_eris)
        else:
            break
        
    biodiversity = 0
    tile_num = -1
    for row in range(1, eris.shape[0]-1):
        for col in range(1, eris.shape[1]-1):
            tile_num += 1
            
            if eris[row, col] == 1:
                biodiversity += pow(2,tile_num)
                
    
    eris_layers = [(original_eris.copy(), 0)]
    
    def get_layer_i(layer_i):
        ind = [x[1] for x in eris_layers].index(layer_i)
        return eris_layers[ind][0]
    
    for _ in range(200):
        layer_is = {x[1] for x in eris_layers}
        
        lowest_layer_i = min(layer_is)
        lowest_eris_layer = get_layer_i(lowest_layer_i)
        if np.sum(lowest_eris_layer[1:4,1:4]) > 0:
            eris_layers.append((np.zeros([5,5]), lowest_layer_i-1))
        
        highest_layer_i = max(layer_is)
        highest_eris_layer = get_layer_i(highest_layer_i)
        if np.sum(highest_eris_layer) - np.sum(lowest_eris_layer[1:4,1:4]) > 0:
            eris_layers.append((np.zeros([5,5]), highest_layer_i+1))
        
        new_eris_layers = []
        for eris, layer_i in eris_layers:
            new_eris = eris.copy()
            
            higher_layer_exists = layer_i + 1 in layer_is
            lower_layer_exists =  layer_i - 1 in layer_is
            
            for row in range(5):
                for col in range(5):
                    if row == 2 and col == 2:
                        continue
                    
                    if row == 0:
                        if higher_layer_exists:
                            N = get_layer_i(layer_i+1)[1,2]
                        else:
                            N = 0
                    elif row == 3 and col == 2:
                        if lower_layer_exists:
                            N = np.sum(get_layer_i(layer_i-1)[-1,:])
                        else:
                            N = 0
                    else:
                        N = eris[row-1, col]
                        
                    if row == 4:
                        if higher_layer_exists:
                            S = get_layer_i(layer_i+1)[3,2]
                        else:
                            S = 0
                    elif row == 1 and col == 2:
                        if lower_layer_exists:
                            S = np.sum(get_layer_i(layer_i-1)[0,:])
                        else:
                            S = 0
                    else:
                        S = eris[row+1, col]
                        
                    if col == 0:
                        if higher_layer_exists:
                            W = get_layer_i(layer_i+1)[2,1]
                        else:
                            W = 0
                    elif col == 3 and row == 2:
                        if lower_layer_exists:
                            W = np.sum(get_layer_i(layer_i-1)[:,-1])
                        else:
                            W = 0
                    else:
                        W = eris[row, col-1]
                        
                    if col == 4:
                        if higher_layer_exists:
                            E = get_layer_i(layer_i+1)[2,3]
                        else:
                            E = 0
                    elif col == 1 and row == 2:
                        if lower_layer_exists:
                            E = np.sum(get_layer_i(layer_i-1)[:,0])
                        else:
                            E = 0
                    else:
                        E = eris[row, col+1]
                        
                    if eris[row, col] == 1:
                        if N + S + W + E != 1:
                            new_eris[row, col] = 0
                    
                    if eris[row, col] == 0:
                        if  N + S + W + E in {1,2}:
                            new_eris[row, col] = 1
            
            new_eris_layers.append((new_eris, layer_i))
        
        eris_layers = new_eris_layers
        
    total_bugs = int(sum([np.sum(x[0]) for x in eris_layers]))
    
    
    return biodiversity, total_bugs

#%%
# Day 25: Cryostasis

@time_this_func
def day25():
    from itertools import combinations
    
    with open("input25.txt") as f:
        ints = tuple(int(x) for x in f.read().split(","))
    
    def next_op(i, ints_list, input_vals, relative_base):
        needs_input = False
        output = None
        inst = str(ints_list[i])
        opcode = int(inst[-2:])
        
        modes = [int(x) for x in inst[:-2][::-1]]      
        modes = modes + [0]*(3-len(modes)) 
        
        max_accessible_ind = max([(i+1) * (modes[0] == 1), \
                                  (i+2) * (modes[1] == 1), \
                                  (i+3) * (modes[2] == 1), \
                                  (ints_list[i+1]) * (modes[0] == 0), \
                                  (ints_list[i+2]) * (modes[1] == 0 and opcode in {1,2,5,6,7,8}), \
                                  (ints_list[i+3]) * (modes[2] == 0 and opcode in {1,2,7,8}), \
                                  (relative_base + ints_list[i+1]) * (modes[0] == 2), \
                                  (relative_base + ints_list[i+2]) * (modes[1] == 2), \
                                  (relative_base + ints_list[i+3]) * (modes[2] == 2)])
          
        if max_accessible_ind > len(ints_list)-1:
            ints_list += [0]*(max_accessible_ind-(len(ints_list)-1))      
        
        if opcode == 1: 
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            ints_list[write_to_ind] = input1 + input2
                
            i += 4
            
        if opcode == 2:
    
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
              
            ints_list[write_to_ind] = input1 * input2
                
            i += 4
        
        elif opcode == 3:
            
            if len(input_vals) == 0:
                needs_input = True
                return i, output, relative_base, needs_input
            
            if modes[0] == 0:
                ints_list[ints_list[i+1]] = input_vals.pop(0)
            elif modes[0] == 2:
                ints_list[ints_list[i+1]+relative_base] = input_vals.pop(0)
        
            i += 2
        
        elif opcode == 4:
            
            if modes[0] == 0:
                output = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                output = ints_list[i+1]
            elif modes[0] == 2:
                output = ints_list[ints_list[i+1]+relative_base]
                
            i += 2
            
        elif opcode == 5:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 != 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 6:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if input1 == 0:
                i = input2
            else:
                i += 3
                
        elif opcode == 7:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 < input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4
            
        elif opcode == 8:
            
            if modes[0] == 0:
                input1 = ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                input1 = ints_list[i+1]
            elif modes[0] == 2:
                input1 = ints_list[relative_base + ints_list[i+1]]
            if modes[1] == 0:
                input2 = ints_list[ints_list[i+2]]
            elif modes[1] == 1:
                input2 = ints_list[i+2]
            elif modes[1] == 2:
                input2 = ints_list[relative_base + ints_list[i+2]]
                
            if modes[2] == 0:
                write_to_ind = ints_list[i+3]
            elif modes[2] == 2:
                write_to_ind = relative_base + ints_list[i+3]
                
            if input1 == input2:
                ints_list[write_to_ind] = 1
            else:
                ints_list[write_to_ind] = 0
            
            i += 4  
            
        elif opcode == 9:
    
            if modes[0] == 0:
                relative_base += ints_list[ints_list[i+1]]
            elif modes[0] == 1:
                relative_base += ints_list[i+1]
            elif modes[0] == 2:
                relative_base += ints_list[relative_base + ints_list[i+1]]
            
            i += 2
            
        elif opcode == 99:
            i = -1
    
        return i, output, relative_base, needs_input
    
    def make_ascii(cmds: list):
        ascii_commands = []
        for cmd in cmds:
            ascii_command = []
            for c in cmd:
                ascii_command.append(ord(c))
            ascii_command.append(ord("\n"))
            ascii_commands.append(ascii_command)
        return ascii_commands
    
    def reverse_path(path):
        rev_path = []
        for move in path[::-1]:
            if move == "north":
                rev_path.append("south")
            elif move == "south":
                rev_path.append("north")
            elif move == "east":
                rev_path.append("west")
            elif move == "west":
                rev_path.append("east")
        return rev_path
    
    def check_path(directions):
        ints_list = list(ints)
        all_inputs = [x for x in make_ascii(directions)]
        i = 0
        relative_base = 0
        ints_list = list(ints)
        
        last_message = ""
        input_vals = []
        input_i = -1
        while True:
            i, output, relative_base, needs_input = next_op(i, ints_list, input_vals, relative_base)
            
            if needs_input:

                if "Command?" in last_message and "can't move" not in last_message:
                    
                    if "==" in last_message:
                        room_name = last_message[:last_message.index("=\n")][6:-2]
                    
                    if "Doors here lead:" in last_message:
                        doors = []
                        if "- north" in last_message:
                            doors.append("north")
                        if "- south" in last_message:
                            doors.append("south")
                        if "- west" in last_message:
                            doors.append("west")
                        if "- east" in last_message:
                            doors.append("east")
                    
                    if "Items here:" in last_message:
                        items = []
                        item = ""
                        for c in last_message[last_message.index("Items here:")+11:last_message.index("Command?")]:
                            if c == "-":
                                if item.strip() != "":
                                    items.append(item.strip())
                                    item = ""
                            else:
                                item  += c
                        items.append(item.strip())
                    elif "==" in last_message:
                        items = []
                
                else:
                    return (None,last_message)
                
                input_i += 1
                if input_i == len(all_inputs):
                    return room_name, items, doors
                input_vals = all_inputs[input_i]
                last_message = ""
                
            if i == -1:
                return (None,last_message)
            
            if output != None:
                last_message += chr(output)
                
    paths = [[]]
    guide = {}
    while True:
        new_paths = []
        for path in paths:
            result = check_path(path)
            
            if result[0] not in guide:
                guide[result[0]] = {"path":path, "items":result[1], "doors":result[2]}
                for door in result[2]:
                    new_paths.append(path + [door])
                    
        if len(new_paths) == 0:
            break
        
        paths = new_paths
        
    safe_items = {}
    for room in guide:
        if len(guide[room]["items"]) == 0:
            continue
        
        for item in guide[room]["items"]:
            if item == "infinite loop": #manually determined
                continue
            
            take_path = guide[room]["path"] + [f"take {item}"]
            result = check_path(take_path + [f"drop {item}", f"take {item}"])
            if result[0] != None:
                safe_items[item] = take_path
                
    take_and_return = {}
    for item in safe_items:
        take_and_return[item] = safe_items[item] + reverse_path(safe_items[item])
        
    all_combos = []
    for i in range(1,len(safe_items)+1):
        all_combos += list(combinations(safe_items,i))
    
    for combo in all_combos:
        directions = []
        for item in combo:
            directions += take_and_return[item]
        directions += guide["Pressure-Sensitive Floor"]["path"]
        
        result = check_path(directions)
        if result[0] == None:
            break
        
    password = ""
    for c in result[1]:
        if c.isnumeric():
            password += c
    password = int(password)
    
    return password