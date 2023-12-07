#%%
import numpy as np
import matplotlib.pyplot as plt
from hashlib import md5
from itertools import permutations

#%%
# Day 1: No Time for a Taxicab

def day1():
    with open("input1.txt") as f:
        directions = f.read().strip().split(", ")
        
    directions = [[x[0],int(x[1:])] for x in directions]
    
    faces = ["N", "E", "S", "W"]
    
    loc = [0,0]
    facing = 0
            
    visited = []
    for direc in directions:
        if direc[0] == "R":
            facing = (facing+1)%len(faces)
        elif direc[0] == "L":
            facing = (facing-1)%len(faces)
            
        if faces[facing] == "N":
            visited = visited + [(loc[0]-x,loc[1]) for x in range(direc[1])]
            loc[0] -= direc[1]
        elif faces[facing] == "E":
            visited = visited + [(loc[0],loc[1]+x) for x in range(direc[1])]
            loc[1] += direc[1]
        elif faces[facing] == "S":
            visited = visited + [(loc[0]+x,loc[1]) for x in range(direc[1])]
            loc[0] += direc[1]
        elif faces[facing] == "W":
            visited = visited + [(loc[0],loc[1]-x) for x in range(direc[1])]
            loc[1] -= direc[1]
            
    dist = sum([abs(x) for x in loc])
    
    
    visited_already = [visited[i] in visited[:i] for i in range(len(visited))]
    first_double = visited[visited_already.index(True)]
    dist_first_double = sum([abs(x) for x in first_double])
    
    
    return dist, dist_first_double

#%%
# Day 2: Bathroom Security

def day2():
    instructions = []
    with open("input2.txt") as f:
        for l in f:
            instructions.append([x for x in l.strip()])
            
    keypad = np.array([[1,2,3],[4,5,6],[7,8,9]])
    
    code = []
    loc = [1,1]
    for number in instructions:
        for move in number:
            if move == "U" and loc[0] != 0:
                loc[0] -= 1
            elif move == "R" and loc[1] != 2:
                loc[1] += 1
            elif move == "D" and loc[0] != 2:
                loc[0] += 1
            elif move == "L" and loc[1] != 0:
                loc[1] -= 1
        code.append(keypad[tuple(loc)])
    code = int("".join([str(x) for x in code]))
    
    
    keypad2 = np.zeros([7,7])
    keypad2[1,3] = 1
    keypad2[2,2:5] = np.array([2,3,4])
    keypad2[3,1:6] = np.array([5,6,7,8,9])
    keypad2[4,2:5] = np.array([-1,-2,-3])
    keypad2[5,3] = -4
    
    code2 = []
    loc = [3,1]
    for number in instructions:
        for move in number:
            if move == "U" and keypad2[loc[0]-1,loc[1]] != 0:
                loc[0] -= 1
            elif move == "R" and keypad2[loc[0],loc[1]+1] != 0:
                loc[1] += 1
            elif move == "D" and keypad2[loc[0]+1,loc[1]] != 0:
                loc[0] += 1
            elif move == "L" and keypad2[loc[0],loc[1]-1] != 0:
                loc[1] -= 1
        code2.append(keypad2[tuple(loc)])  
    code2 = "".join([str(int(x)) for x in code2])
    code2 = code2.replace("-1","A")
    code2 = code2.replace("-2","B")
    code2 = code2.replace("-3","C")
    code2 = code2.replace("-4","D")
    
    
    return code, code2

#%%
# Day 3: Squares With Three Sides

def day3():
    triangles1 = []
    with open("input3.txt") as f:
        for l in f:
            triangles1.append([int(x) for x in l.split()])
            
    def valid_triangle(lengths):
        sorted_lengths = sorted(lengths)
        if sum(sorted_lengths[:2]) <= sorted_lengths[2]:
            return False
        return True
    
    valid_triangles1 = 0
    for triangle in triangles1:
        if valid_triangle(triangle):
            valid_triangles1 += 1
            
            
    vert_triangles = [x[0] for x in triangles1] + [x[1] for x in triangles1] + [x[2] for x in triangles1]
    triangles2 = [vert_triangles[i:i+3] for i in range(0, len(vert_triangles), 3)]
    
    valid_triangles2 = 0
    for triangle in triangles2:
        if valid_triangle(triangle):
            valid_triangles2 += 1
            
    
    return valid_triangles1, valid_triangles2

#%%
# Day 4: Security Through Obscurity

def day4(return_room_names = False):
    rooms = []
    with open("input4.txt") as f:
        for l in f:
            new = l.strip()
            separator = len(new) - new[::-1].index("-")
            part1 = new[:separator]
            parts2and3 = new[separator:]
            separator = parts2and3.index("[")
            part2 = int(parts2and3[:separator])
            part3 = parts2and3[separator+1:-1]
            rooms.append([part1, part2, part3])
            
    total_correct_sectors = 0
    correct_rooms = []
    for room in rooms: 
        letters = "abcdefghijklmnopqrstuvwxyz"
        counts = [[x, room[0].count(x)] for x in letters]
        sorted_counts = sorted(sorted(counts), reverse = True, key = lambda x: x[1])
        checksum = "".join([x[0] for x in sorted_counts[:5]])
    
        if room[2] == checksum:
            correct_rooms.append(room)
            total_correct_sectors += room[1]
            
            
    replacements = dict(zip(letters,letters[1:]+"a"))
    replacements["-"] = " "
    replacements[" "] = " "
    
    room_names = []
    for room in correct_rooms:
        shift = room[1]%len(letters)
        room_name = room[0]
        for _ in range(shift):
            room_name = [replacements[x] for x in room_name]
        room_name = "".join(room_name)
        room_names.append(room_name)
        
        if "northpole object storage" in room_name:
            break
        
    north_pole_sector = room[1]
    
    if return_room_names:
        return total_correct_sectors, north_pole_sector, room_names
    else:
        return total_correct_sectors, north_pole_sector
    
#%%
# Day 5: How About a Nice Game of Chess?

def day5():
    door_ID = "uqwqemis"
    
    i = 0
    password1 = ""
    password2 = [None]*8
    while True:
        while True:
            i += 1
            to_hash = door_ID + str(i)
            hexed = md5(to_hash.encode()).hexdigest()
            if hexed[:5] == "00000":
                break
            
        if len(password1) < 8:
            password1 = password1 + hexed[5]
            
        if hexed[5] in "01234567":
            if password2[int(hexed[5])] == None:
                password2[int(hexed[5])] = hexed[6]
                
        if None not in password2:
            break
        
    password2 = "".join(password2)
    
    return password1, password2

#%%
# Day 6: Signals and Noise

def day6():
    messages = []
    with open("input6.txt") as f:
        for l in f:
            messages.append(l.strip())
    
    message_length = len(messages[0])
    
    letters = "abcdefghijklmnopqrstuvwxyz"
    decrypted_message1 = ""
    decrypted_message2 = ""
    for i in range(message_length):
        vertical = [x[i] for x in messages]
        counts = [[x,vertical.count(x)] for x in letters]
        sorted_counts = sorted(counts, reverse = True, key = lambda x: x[1])
        decrypted_message1 += sorted_counts[0][0]
        decrypted_message2 += sorted_counts[-1][0]
        
    return decrypted_message1, decrypted_message2

#%%
# Day 7: Internet Protocol Version 7

def day7():
    ips = []
    with open("input7.txt") as f:
        for l in f:
            ips.append(l.strip())
    
    num_tls = 0
    num_ssl = 0
    for ip in ips:
        abba_outside = False
        abba_inside = False
        
        abas = []
        babs = []
        
        three_ago = ip[0]
        two_ago = ip[1]
        one_ago = ip[2]
        if "[" in ip[:3] and "]" not in ip[:3]:
            bracketed = True
        else:
            bracketed = False
            
        if three_ago == one_ago and three_ago != two_ago and two_ago.isalpha():
            abas.append(three_ago + two_ago + one_ago)
            
        for i in range(3,len(ip)):
            curr = ip[i]
            one_ago = ip[i-1]
            two_ago = ip[i-2]
            three_ago = ip[i-3]
        
            if curr == "[":
                bracketed = True
            elif curr == "]":
                bracketed = False
                
            if two_ago == curr and two_ago != one_ago and one_ago.isalpha():
                if not bracketed:
                    abas.append(two_ago + one_ago + curr)
                if bracketed:
                    babs.append(two_ago + one_ago + curr)
                
            if curr == three_ago and one_ago == two_ago and curr != one_ago:
                if not bracketed:
                    abba_outside = True
                if bracketed:
                    abba_inside = True
        
        if abba_outside and not abba_inside:
            num_tls += 1
        
        aba_bab_match = False
        for aba in abas:
            wanted_bab = aba[1] + aba[0] + aba[1]
            if wanted_bab in babs:
                aba_bab_match = True
                break
            
        if aba_bab_match:
            num_ssl += 1
            
    return num_tls, num_ssl

#%%
# Day 8: Two-Factor Authentication

def day8(visualize = False):
    instructions = []
    with open("input8.txt") as f:
        for l in f:
            new = l.split()
            if "rotate" in new:
                instructions.append([new[1], int(new[2].split("=")[-1]), int(new[-1])])
            elif "rect" in new:
                instructions.append([new[0], [int(x) for x in new[1].split("x")]])
                
    screen = np.zeros([6,50])
    for instruct in instructions:
        if instruct[0] == "rect":
            rect_dim = instruct[1][::-1]
            screen[0:rect_dim[0], 0:rect_dim[1]] = 1
        else:
            if instruct[0] == "row":
                change_row = instruct[1]
                shift = instruct[2]
                screen[change_row,:] = np.concatenate([screen[change_row,-shift:],screen[change_row,:-shift]])
            elif instruct[0] == "column":
                change_col = instruct[1]
                shift = instruct[2]
                screen[:,change_col] = np.concatenate([screen[-shift:,change_col],screen[:-shift,change_col]])
                
    lit_pix = int(np.sum(screen))
    
    if visualize:
        plt.imshow(screen)
             
    return lit_pix

#%%
# Day 9: Explosives in Cyberspace

def day9():
    with open("input9.txt") as f:
        compressed = f.read().strip()
    
    def decompressed_length(to_decompress, unpack_all):
        i = 0
        repeats = []
        unrepeated = 0
        while i < len(to_decompress):
            if to_decompress[i] == "(":
                start = i
                while to_decompress[i] != ")":
                    i += 1
                repeat_info = [int(x) for x in to_decompress[start+1:i].split("x")]
                to_repeat = to_decompress[i+1:i+1+repeat_info[0]]
                repeats.append([repeat_info[1],to_repeat])
                i += repeat_info[0]
            else:
                unrepeated += 1
            i += 1
        
        if unpack_all:
            return sum([x[0]*decompressed_length(x[1], True) for x in repeats]) + unrepeated
        else:
            return sum([x[0]*len(x[1]) for x in repeats]) + unrepeated
    
    return decompressed_length(compressed, False), decompressed_length(compressed, True)

#%%
# Day 10: Balance Bots

def day10():
    handler = {}
    with open("input10.txt") as f:
        for l in f:
            new = l.split()
            if "value" in new:
                if " ".join(new[-2:]) not in handler.keys():
                    handler[" ".join(new[-2:])] = [[], [int(new[1])]]
                else:
                    handler[" ".join(new[-2:])][1].append(int(new[1]))
            elif "low" in new:
                if " ".join(new[:2]) not in handler.keys():
                    handler[" ".join(new[:2])] = [[" ".join(new[5:7]), " ".join(new[-2:])], []]
                else:
                    handler[" ".join(new[:2])][0] = [" ".join(new[5:7]), " ".join(new[-2:])]
                    
                if " ".join(new[5:7]) not in handler.keys():
                    handler[" ".join(new[5:7])] = [[],[]]
                if  " ".join(new[-2:]) not in handler.keys():
                    handler[ " ".join(new[-2:])] = [[], []]
                    
    while 2 in [len(x[1]) for x in handler.values()]:
        ind = [len(x[1]) for x in handler.values()].index(2)
        key = list(handler.keys())[ind]
                
        smaller = min(handler[key][1])
        larger = max(handler[key][1])
        smaller_to = handler[key][0][0]
        larger_to = handler[key][0][1]
        
        if smaller == 17 and larger == 61:
            handler_of_17_61 = int(key.split()[1])
        
        handler[smaller_to][1].append(smaller)
        handler[larger_to][1].append(larger)
        handler[key][1] = []
    
    output_mult = handler["output 0"][1][0] * handler["output 1"][1][0] * handler["output 2"][1][0]
            
    return handler_of_17_61, output_mult

#%%
# Day 11: Radioisotope Thermoelectric Generators

def day11():
    floor_info = []
    with open("input11.txt") as f:
        for l in f:
            floor_info.append(l.strip())
            
    gen_count = []
    chip_count = []
    for contents in floor_info:
        split_up = contents.split()
        
        gen_count.append(len([ind for ind in range(len(split_up)-1) if "generator" in split_up[ind+1]]))
        chip_count.append(len([x for x in split_up if "compatible" in x]))
        
        gen_count_orig = gen_count.copy()
        chip_count_orig = chip_count.copy()

    #Strategy worked out by hand.
    if chip_count[0] >= 2 and sum(gen_count[1:]) == 0:
        moves = 0
        moves += 3
        chip_count[0] -= 2
        moves += sum([2*(3-i)*x for i,x in enumerate(chip_count)])
        moves += sum([2*(3-i)*x for i,x in enumerate(gen_count)])
        min_moves1 = moves
        
        gen_count = gen_count_orig.copy()
        chip_count = chip_count_orig.copy()
        chip_count[0] += 2
        gen_count[0] += 2
        moves = 0
        moves += 3
        chip_count[0] -= 2
        moves += sum([2*(3-i)*x for i,x in enumerate(chip_count)])
        moves += sum([2*(3-i)*x for i,x in enumerate(gen_count)])
        min_moves2 = moves
        
        return min_moves1, min_moves2
    else:
        return "Invalid methodology for this input"
    
#%%
# Day 12: Leonardo's Monorail

def day12():
    operations = []
    with open("input12.txt") as f:
        for l in f:
            operations.append([int(x) if x.isnumeric() or "-" in x else x for x in l.split()])
    
    def get_registers(registers):
        i = 0
        while i in range(len(operations)):
            op = operations[i]
            if op[0] == "inc":
                registers[op[1]] += 1
            elif op[0] == "dec":
                registers[op[1]] -= 1
            elif op[0] == "cpy":
                if type(op[1]) == int:
                    registers[op[2]] = op[1]
                else:
                    registers[op[2]] = registers[op[1]]
            elif op[0] == "jnz":
                if type(op[1]) == int:
                    if op[1] == 0:
                        is_zero = True
                    else:
                        is_zero = False
                else:
                    if registers[op[1]] == 0:
                        is_zero = True
                    else:
                        is_zero = False
                if not is_zero:
                    if type(op[2]) == int:
                        i = i + op[2]
                    else:
                        i = i + registers[op[2]]
                    continue
            i = i + 1      
        return registers   
    
    a_final1 = get_registers(dict(zip("abcd",[0,0,0,0])))["a"]
    a_final2 = get_registers(dict(zip("abcd",[0,0,1,0])))["a"]   
    
    return a_final1, a_final2

#%%
# Day 13: A Maze of Twisty Little Cubicles

def day13():
    def is_open(coordinate, input_num = 1350):
        x = coordinate[0]
        y = coordinate[1]
        num = x*x + 3*x + 2*x*y + y + y*y + input_num
        bin_rep = bin(num)
        num_1s = bin_rep.count("1")
        if num_1s%2 == 0:
            return True
        else:
            return False
        
    def get_options(coordinate):
        x = coordinate[0]
        y = coordinate[1]
        options = [x for x in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)] if -1 not in x]
        return options
    
    def flatten(nested_lists):
        flattened = []
        for l in nested_lists:
            flattened = flattened + l
        return flattened
        
    paths = [[(1,1)]]
    dest = (31,39)
    steps = 0
    visited = [(1,1)]
    while True:
        steps += 1
        new_paths = []
        for path in paths:
            curr_loc = path[-1]
            options = get_options(curr_loc)
            for option in options:
                if option in flatten(paths) or not is_open(option):
                    continue
                else:
                    if steps <= 50:
                        visited.append(option)
                    new_paths.append(path.copy() + [option])
                    
        paths = new_paths
        if len(paths[0]) > 51:
            paths = sorted(paths, key = lambda x: abs(x[-1][0] - dest[0]) + abs(x[-1][1] - dest[1]))[:20]
        
        if dest in [x[-1] for x in paths]:
            break
    
    full_path = paths[[x[-1] for x in paths].index(dest)]
    num_steps = len(full_path) - 1
    
    visited_in_50 = len(set(visited))
    
    return num_steps, visited_in_50

#%%
# Day 14: One-Time Pad

def day14():
    print("Runtime: 80 s")
    
    salt = "yjdafjpo"
    
    def first_triple(string):
        for i in range(2, len(string)):
            two_ago = string[i-2]
            one_ago = string[i-1]
            curr = string[i]
            if two_ago == one_ago == curr:
                return curr*5
        return None
    
    def get_keys(num_md5_hashes):
        hexes = {}
        key_inds = []
        ind = 0
        while len(key_inds) < 64:
            if ind not in hexes.keys():
                to_hash = salt + str(ind)
                for _ in range(num_md5_hashes):
                    hexed = md5(to_hash.encode()).hexdigest()
                    to_hash = hexed
                hexes[ind] = hexed
            else:
                hexed = hexes[ind]
            
            triple_quint = first_triple(hexed)
            if triple_quint != None:
                is_key = False
                for i in range(ind+1,ind+1001):
                    if i not in hexes.keys():
                        to_hash = salt + str(i)
                        for _ in range(num_md5_hashes):
                            hexed = md5(to_hash.encode()).hexdigest()
                            to_hash = hexed
                        hexes[i] = hexed
                    else:
                        hexed = hexes[i]
                    if triple_quint in hexed:
                        is_key = True
                        break
                if is_key:
                    key_inds.append(ind)
                    
            ind += 1
                
        return key_inds[63]
    
    return get_keys(1), get_keys(1+2016)

#%%
# Day 15: Timing is Everything

def day15():
    def shift(disc, times = 1):
        shifted = disc[times:] + disc[:times]
        return shifted
    
    discs = []
    with open("input15.txt") as f:
        for l in f:
            info = l.split()
            disc = list(range(int(info[3])))
            disc = shift(disc, int(info[-1][:-1]))
            discs.append(disc)
    
    def first_drop_time(discs):
        drop_time = 0
        while True:
            disc_times = list(range(drop_time+1, drop_time + len(discs) + 1))
            disc_locs = [d[i%len(d)] for d,i in zip(discs,disc_times)]
            if sum(disc_locs) == 0:
                break
            drop_time += 1
        return drop_time
        
    first_drop1 = first_drop_time(discs)
    
    discs.append(list(range(11)))
    first_drop2 = first_drop_time(discs)
    
    return first_drop1, first_drop2

#%%
# Day 16: Dragon Checksum

def day16():
    state_og = "10010000000110000"
    
    def dragon_fold(previous):
        adding = "".join("0" if x == "1" else "1" for x in previous)
        return previous + "0" + adding[::-1]
    
    def get_checksum(code):
        pairs = []
        for i in range(0,len(code),2):
            pairs.append(code[i:i+2])
        
        possible_checksum = "".join(["0" if p.count("1") == 1 else "1" for p in pairs])
        
        if len(pairs)%2 == 0:
            return get_checksum(possible_checksum)
        else:
            return possible_checksum
    
    min_length = 272
    state = state_og
    while len(state) < min_length:
        state = dragon_fold(state)
    checksum1 = get_checksum(state[:min_length])
    
    
    min_length = 35651584
    state = state_og
    while len(state) < min_length:
        state = dragon_fold(state)
    checksum2 = get_checksum(state[:min_length])
    
    
    return int(checksum1), int(checksum2)

#%%
# Day 17: Two Steps Forward

def day17():
    salt = "qljzarfv"
    
    def which_open(hex_code):
        open_chars = "bcdef"
        open_doors = ""
        if hex_code[0] in open_chars:
            open_doors += "U"
        if hex_code[1] in open_chars:
            open_doors += "D"
        if hex_code[2] in open_chars:
            open_doors += "L"
        if hex_code[3] in open_chars:
            open_doors += "R"
        return open_doors
    
    def get_loc(salt_plus):
        if len(salt_plus) == len(salt):
            return [0,0]
        
        loc = [0,0]
        for move in salt_plus[8:]:
            if move == "U":
                loc[0] -= 1
            elif move == "D":
                loc[0] += 1
            elif move == "L":
                loc[1] -= 1
            elif move == "R":
                loc[1] += 1
            else:
                raise Exception("Invalid")
        return loc
        
    directions = [salt]
    paths_to_vault = []
    while len(directions) != 0:
        new_directions = []
        for direction in directions:
            
            hexed = md5(direction.encode()).hexdigest()
            open_doors = which_open(hexed)
            loc = get_loc(direction)
            if loc[0] == 0:
                open_doors = open_doors.replace("U","")
            if loc[0] == 3:
                open_doors = open_doors.replace("D","")
            if loc[1] == 0:
                open_doors = open_doors.replace("L","")
            if loc[1] == 3:
                open_doors = open_doors.replace("R","")
            for door in open_doors:
                if get_loc(direction + door) == [3,3]:
                    paths_to_vault.append(direction + door)
                else:
                    new_directions.append(direction + door)
                    
        directions = new_directions
        
    path_lengths_to_vault = [len(x)-8 for x in paths_to_vault]
    fastest_path = paths_to_vault[path_lengths_to_vault.index(min(path_lengths_to_vault))][8:]
    max_length = max(path_lengths_to_vault)
    
    return fastest_path, max_length

#%%
# Day 18: Like a Rogue

def day18():
    with open("input18.txt") as f:
        first_row = [x for x in f.read().strip()]
        first_row = [True if x == "." else False for x in first_row]
        
    row = np.array(first_row)
    
    def get_new_row(row):
        center = np.hstack([np.array(True), row, np.array(True)])
        left = np.hstack([np.array(True), center[:-1]])
        right = np.hstack([center[1:], np.array(True)])
        
        cond1 = (left == False) * (center == False) * right
        cond2 = left * (center == False) * (right == False)
        cond3 = (left == False) * center * right
        cond4 = left * center * (right == False)
        
        next_row = cond1 | cond2 | cond3 | cond4
        next_row = next_row == False
        next_row = next_row[1:-1]
        return next_row
    
    safe_tiles = np.sum(row)
    for r in range(2,400001):
        row = get_new_row(row)
        safe_tiles += np.sum(row)
        if r == 40:
            safe_tiles_40 = safe_tiles
    safe_tiles_400000 = safe_tiles
    
    return safe_tiles_40, safe_tiles_400000

#%%
# Day 19: An Elephant Named Joseph

def day19():
    num_elves = 3005290
    
    elf_nums = list(range(1,num_elves+1)) 
    while len(elf_nums) > 1:
        if len(elf_nums)%2 == 0:
            elf_nums = elf_nums[0::2]
        else:
            elf_nums = elf_nums[2::2]
      
    winner1 = elf_nums[0]
    
    # Used to find pattern. Too slow for actual num_elves.
    # def p2(num_elves):
    #     elf_nums = list(range(1,num_elves+1))
    #     i = 0
    #     while len(elf_nums) > 1:
    #         remove_ind = (i + (len(elf_nums)//2))%len(elf_nums)
    #         print(elf_nums[remove_ind])
    #         del elf_nums[remove_ind]
    #         if i > len(elf_nums)-1:
    #             i = 0
    #         elif remove_ind > i:
    #             i = (i+1)%len(elf_nums)
    #     return elf_nums[0]
    # pattern_finder = {x:p2(x) for x in range(1,101)} #manual examination
    
    lower_power_of_3 = 1
    while lower_power_of_3 * 3 < num_elves:
        lower_power_of_3 *= 3
        
    if num_elves < lower_power_of_3 * 2:
        winner2 = num_elves - lower_power_of_3
    else:
        winner2 = (lower_power_of_3/2) + 2*(num_elves - 2*lower_power_of_3)
        
    return winner1, winner2

#%%
# Day 20: Firewall Rules

def day20():
    blocked_ranges = []
    with open("input20.txt") as f:
        for l in f:
            blocked_ranges.append([int(x) for x in l.strip().split("-")])
    blocked_ranges.sort()
    
    def in_blocked_ranges(x):
        for ind, blocked in enumerate(blocked_ranges):
            if x >= blocked[0] and x <= blocked[1]:
                return True, ind
        return False, None
    
    if blocked_ranges[0][0] > 0:
        lowest = 0
    else:
        for blocked in blocked_ranges:
            first_outside = blocked[1]+1
            if not in_blocked_ranges(first_outside)[0] and first_outside <= 4294967295:
                lowest = first_outside
                break
        
        
    num_blocked = 0
    i = lowest #might as well start here
    while i <= 4294967295:
        is_blocked, ind = in_blocked_ranges(i)
        if is_blocked:
            i = blocked_ranges[ind][1]+1
        else:
            num_blocked += 1
            i += 1
            
            
    return lowest, num_blocked

#%%
# Day 21: Scrambled Letters and Hash

def day21():
    
    changes = []
    with open("input21.txt") as f:
        for l in f:
            new = l.split()
            if new[0] == "swap":
                if new[1] == "position":
                    changes.append([new[0], new[1], [int(new[2]), int(new[5])]])
                elif new[1] == "letter":
                    changes.append([new[0], new[1], [new[2], new[5]]])
            elif new[0] == "rotate":
                if new[1] == "left" or new[1] == "right":
                    changes.append([new[0], new[1], int(new[2])])
                elif new[1] == "based":
                    changes.append([new[0], new[1] + " " + new[2], new[6]])
            elif new[0] == "reverse":
                changes.append([new[0], [int(new[2]), int(new[4])]])
            elif new[0] == "move":
                changes.append([new[0], [int(new[2]), int(new[5])]])
    
    def swap_pos(password, swap):
        swap.sort()
        letter1 = password[swap[0]]
        letter2 = password[swap[1]]
        new_password = password[0:swap[0]] + letter2 + password[swap[0]+1:swap[1]] + letter1 + password[swap[1]+1:]
        return new_password
    
    def swap_letters(password, letters):
        swap = [password.index(letters[0]), password.index(letters[1])]
        return swap_pos(password, swap)
    
    def rotate(password, direction, steps_or_letter):
        if direction == "right":
            steps_or_letter = steps_or_letter%len(password)
            return password[-steps_or_letter:] + password[:-steps_or_letter]
        elif direction == "left":
            steps_or_letter = steps_or_letter%len(password)       
            return rotate(password, "right", len(password) - steps_or_letter)
        elif direction == "based on":
            ind = password.index(steps_or_letter)
            return rotate(password, "right", ind + 1 + (ind >= 4))
        
    def reverse(password, inds):
        inds.sort()
        return password[:inds[0]] + password[inds[0]:inds[1]+1][::-1] + password[inds[1]+1:]
    
    def move(password, inds):
        moving_letter = password[inds[0]]
        missing_letter_password = password[:inds[0]] + password[inds[0]+1:]
        new_password = [x for x in missing_letter_password]
        new_password.insert(inds[1], moving_letter)
        return "".join(new_password)
    
    def scramble(password):
        for change in changes:
            if change[0] == "swap":
                if change[1] == "position":
                    password = swap_pos(password, change[2])
                elif change[1] == "letter":
                    password = swap_letters(password, change[2])
            elif change[0] == "rotate":
                password = rotate(password, change[1], change[2])
            elif change[0] == "reverse":
                password = reverse(password, change[1])
            elif change[0] == "move":
                password = move(password, change[1])
        return password
    
    scrambled = scramble("abcdefgh")
    
    
    potentials = ["".join(x) for x in permutations("abcdefgh", len("abcdefgh"))]
    for potential in potentials:
        if scramble(potential) == "fbgdceah":
            break
    unscrambled = potential
    
    
    return scrambled, unscrambled

#%%
# Day 22: Grid Computing

def day22():
    nodes = {}
    with open("input22.txt") as f:
        for l in f:
            if "dev" not in l:
                continue
            
            node_data = l.split()
            loc = node_data[0].split("-")
            loc = (int(loc[1][1:]), int(loc[2][1:]))
            size = int(node_data[1][:-1])
            used = int(node_data[2][:-1])
            avail = int(node_data[3][:-1])
            use_p = int(node_data[4][:-1])
            nodes[loc] = {"size":size, "used":used, "avail":avail, "use%":use_p}
    
    viables = 0
    for a in nodes.keys():
        for b in nodes.keys():
            if nodes[a]["used"] == 0:
                continue
            
            if a == b:
                continue
            
            if nodes[a]["used"] <= nodes[b]["avail"]:
                viables += 1
                
    
    #Visual inspection of map to get strategy
    max_x = max([x[0] for x in nodes.keys()])
    
    all_used = [nodes[x]["used"] for x in nodes.keys()]
    zero_loc = list(nodes.keys())[all_used.index(0)]
    
    up = zero_loc[0]
    left = zero_loc[1]
    down = max_x
    back_up = (max_x-1)*5
    
    total = up + left + down + back_up
    
    
    return viables, total

#%%
# Day 23: Safe Cracking

def day23():
    from copy import deepcopy
    
    operations = []
    with open("input23.txt") as f:
        for l in f:
            operations.append([int(x) if x.isnumeric() or "-" in x else x for x in l.split()])
            
    operations_og = deepcopy(operations)
    
    def get_registers(registers):
        i = 0
        while i in range(len(operations)):
            op = operations[i]
            if op[0] == "tgl":
                if type(op[1]) == int:
                    to_toggle = op[1]
                else:
                    to_toggle = registers[op[1]]
                if i + to_toggle in range(len(operations)):
                    if operations[i+to_toggle][0] == "inc":
                        operations[i+to_toggle][0] = "dec"
                    elif operations[i+to_toggle][0] in ["dec", "tgl"]:
                        operations[i+to_toggle][0] = "inc"
                    elif operations[i+to_toggle][0] == "jnz":
                        operations[i+to_toggle][0] = "cpy"
                    elif operations[i+to_toggle][0] == "cpy":
                        operations[i+to_toggle][0] = "jnz"            
            elif op[0] == "inc":
                registers[op[1]] += 1
            elif op[0] == "dec":
                registers[op[1]] -= 1
            elif op[0] == "cpy":
                if type(op[1]) == int:
                    registers[op[2]] = op[1]
                else:
                    registers[op[2]] = registers[op[1]]
            elif op[0] == "jnz":
                if type(op[1]) == int:
                    if op[1] == 0:
                        is_zero = True
                    else:
                        is_zero = False
                else:
                    if registers[op[1]] == 0:
                        is_zero = True
                    else:
                        is_zero = False
                if not is_zero:
                    if type(op[2]) == int:
                        i = i + op[2]
                    else:
                        i = i + registers[op[2]]
                    continue
            i = i + 1
        return registers   
    
    a_final1 = get_registers(dict(zip("abcd",[7,0,0,0])))["a"]
    
    #Answer found by letting this run for a long time.
    operations = deepcopy(operations_og)
    a_final2 = get_registers(dict(zip("abcd",[12,0,0,0])))["a"]
    
    #Maybe come back to this one for an "efficient" answer. Answer from online.
    if "a_final2" not in locals():
        a_final2 = (79*77) + np.math.factorial(12)
        
    return a_final1, a_final2

#%%
# Day 24: Air Duct Spelunking

def day24():
    from itertools import combinations
    
    duct_map = []
    with open("input24.txt") as f:
        for l in f:
            new = l.strip()
            new = new.replace(".","8")
            new = new.replace("#","9")
            duct_map.append([int(x) for x in new])
    
    duct_map = np.array(duct_map)
    
    goal_locs = {}
    goal = 0
    while len(duct_map[duct_map == goal]) == 1:
        loc = [int(x) for x in np.where(duct_map == goal)]
        goal_locs[goal] = tuple(loc)
        goal += 1
        
    combinations = list(combinations(range(len(goal_locs)),2))
    
    duct_map[duct_map == 9] = -1
    duct_map[duct_map == 8] = 0
    
    def get_shortest_dist(goal1, goal2):
        def flatten(nested):
            flattened = []
            for l in nested:
                flattened = flattened + l
            return flattened
        
        start = goal_locs[goal1]
        finish = goal_locs[goal2]
        paths = [[start]]
        while True:
            new_paths = []
            flat_paths = flatten(paths)
            for path in paths:
                latest = path[-1]
                if duct_map[latest[0]-1, latest[1]] != -1 and tuple([latest[0]-1, latest[1]]) not in flat_paths:
                    new_paths.append(path + [tuple([latest[0]-1, latest[1]])])
                if duct_map[latest[0]+1, latest[1]] != -1 and tuple([latest[0]+1, latest[1]]) not in flat_paths:
                    new_paths.append(path + [tuple([latest[0]+1, latest[1]])])
                if duct_map[latest[0], latest[1]-1] != -1 and tuple([latest[0], latest[1]-1]) not in flat_paths:
                    new_paths.append(path + [tuple([latest[0], latest[1]-1])])
                if duct_map[latest[0], latest[1]+1] != -1 and tuple([latest[0], latest[1]+1]) not in flat_paths:
                    new_paths.append(path + [tuple([latest[0], latest[1]+1])])
                    
            if len(new_paths) == 0:
                raise Exception("Too pruned")
                    
            if finish in [x[-1] for x in new_paths]:
                break
            
            paths_to_add = []
            for path in new_paths:
                if path[-1] not in [x[-1] for x in paths_to_add]:
                    paths_to_add.append(path)
            
            sorted_paths = sorted(paths_to_add, key = lambda x: abs(x[-1][0]-finish[0]) + abs(x[-1][1]-finish[1]))
            paths = sorted_paths[:20]
            
        shortest = len(new_paths[0]) - 1
        return shortest
    
    path_lengths = {}
    for combo in combinations:
        path_lengths[tuple(sorted(combo))] = get_shortest_dist(*combo)
        
    possible_orders = list(permutations(range(1,len(goal_locs)),len(goal_locs)-1))
    
    shortest_journey_to = np.inf
    for order in possible_orders:
        
        pairs = [(0, order[0])]
        for i in range(len(order)-1):
            pairs.append((order[i], order[i+1]))
            
        journey_length = 0
        for pair in pairs:
            journey_length += path_lengths[tuple(sorted(pair))]
            if journey_length > shortest_journey_to:
                break
            
        shortest_journey_to = min(shortest_journey_to, journey_length)
    
    
    shortest_journey_to_and_back = np.inf
    for order in possible_orders:
        
        pairs = [(0, order[0])]
        for i in range(len(order)-1):
            pairs.append((order[i], order[i+1]))
        pairs.append((order[i+1], 0))
            
        journey_length = 0
        for pair in pairs:
            journey_length += path_lengths[tuple(sorted(pair))]
            if journey_length > shortest_journey_to_and_back:
                break
            
        shortest_journey_to_and_back = min(shortest_journey_to_and_back, journey_length)
    
    
    return shortest_journey_to, shortest_journey_to_and_back

#%%
# Day 25: Clock Signal

def day25():
    operations = []
    with open("input25.txt") as f:
        for l in f:
            operations.append([int(x) if x.isnumeric() or "-" in x else x for x in l.split()])
        
    def get_output(a, num_tics):
        registers = dict(zip("abcd",[a,0,0,0]))
        
        output = []
        i = 0
        while len(output) < num_tics:
            op = operations[i]
            if op[0] == "inc":
                registers[op[1]] += 1
            elif op[0] == "dec":
                registers[op[1]] -= 1
            elif op[0] == "cpy":
                if type(op[1]) == int:
                    registers[op[2]] = op[1]
                else:
                    registers[op[2]] = registers[op[1]]
            elif op[0] == "jnz":
                if type(op[1]) == int:
                    if op[1] == 0:
                        is_zero = True
                    else:
                        is_zero = False
                else:
                    if registers[op[1]] == 0:
                        is_zero = True
                    else:
                        is_zero = False
                if not is_zero:
                    if type(op[2]) == int:
                        i = i + op[2]
                    else:
                        i = i + registers[op[2]]
                    continue
            elif op[0] == "out":
                output.append(registers[op[1]])
                if len(output)%2 == 0:
                    correct = 1
                else:
                    correct = 0
                if output[-1] != correct:
                    return False
            i = i + 1    
            
        return True  
    
    num_tics = 10
    i = 0
    while True:
        if get_output(i, num_tics):
            break
        i += 1
    register_input = i
        
    return register_input