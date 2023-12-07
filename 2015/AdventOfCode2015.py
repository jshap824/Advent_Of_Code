#%%
import numpy as np
import re
import matplotlib.pyplot as plt
from itertools import combinations

#%%
# Day 1: Not Quite Lisp

def day1():
    with open("input1.txt") as f:
        instructions = [x for x in f.read()]
        
    final_floor = instructions.count("(") - instructions.count(")")
    
    
    floor = 0
    for i, el in enumerate(instructions):
        if el == "(":
            floor += 1
        elif el == ")":
            floor -= 1
        else:
            raise Exception("Invalid")
        if floor == -1:
            break
    first_basement = i + 1
    
    
    return final_floor, first_basement

#%%
# Day 2: I Was Told There Would Be No Math

def day2():
    gifts = []
    with open("input2.txt") as f:
        for l in f:
            new = l.strip().split("x")
            gifts.append([int(x) for x in new])
            
    def wrapping_paper(gift):
        extra = np.prod(sorted(gift)[:2])
        wrapping = 2*(gift[0]*gift[1] + gift[0]*gift[2] + gift[1]*gift[2])
        return wrapping + extra
    
    total_wrapping = sum([wrapping_paper(x) for x in gifts])
    
    
    def ribbon(gift):
        bow = np.prod(gift)
        around = 2*np.sum(sorted(gift)[:2])
        return bow + around

    total_ribbon = sum([ribbon(x) for x in gifts])
    
    
    return total_wrapping, total_ribbon

#%%
# Day 3: Perfectly Spherical Houses in a Vacuum

def day3():
    with open("input3.txt") as f:
        instructions = [x for x in f.read()]
    
    loc = [0,0]
    visited = {tuple(loc)}
    for instruction in instructions:
        if instruction == "^":
            loc[0] -= 1
        elif instruction == "v":
            loc[0] += 1
        elif instruction == ">":
            loc[1] += 1
        elif instruction == "<":
            loc[1] -= 1
        else:
            raise Exception("Invalid")
        visited.add(tuple(loc))
    
    visited_houses = len(visited)
    
    
    santa_loc = [0,0]
    bot_loc = [0,0]
    visited = {tuple(bot_loc)}
    for i, instruction in enumerate(instructions):
        if i%2 == 0:
            if instruction == "^":
                santa_loc[0] -= 1
            elif instruction == "v":
                santa_loc[0] += 1
            elif instruction == ">":
                santa_loc[1] += 1
            elif instruction == "<":
                santa_loc[1] -= 1
            else:
                raise Exception("Invalid")
            visited.add(tuple(santa_loc))
        else:
            if instruction == "^":
                bot_loc[0] -= 1
            elif instruction == "v":
                bot_loc[0] += 1
            elif instruction == ">":
                bot_loc[1] += 1
            elif instruction == "<":
                bot_loc[1] -= 1
            else:
                raise Exception("Invalid")
            visited.add(tuple(bot_loc))
    
    visited_houses_2 = len(visited)
    
    
    return visited_houses, visited_houses_2

#%%
# Day 4: The Ideal Stocking Stuffer

def day4():
    from hashlib import md5
    
    key = "iwrupvqb"
    
    hexed = "123456"
    five_not_found = True
    num = 0
    while hexed[:6] != "000000":
        num += 1
        full_key = key + str(num)
        hexed = md5(full_key.encode()).hexdigest()
        if hexed[:5] == "00000" and five_not_found:
            num_to_five = num
            five_not_found = False
    
    
    return num_to_five, num

#%%
# Day 5: Doesn't He Have Intern-Elves For This?

def day5():
    strings = []
    with open("input5.txt") as f:
        for l in f:
            strings.append(l.strip())
    
    nice = []
    for string in strings:
        if "ab" in string or "cd" in string or "pq" in string or "xy" in string:
            nice.append(False)
            continue
        
        cond1 = sum([string.count(x) for x in "aeiou"]) >= 3
        if not cond1:
            nice.append(False)
            continue
        
        cond2 = False
        s_last = string[0]
        for s in string[1:]:
            if s == s_last:
                cond2 = True
                break
            s_last = s
            
        if not cond2:
            nice.append(False)
        else:
            nice.append(True)
    
    nice_strings_1 = nice.count(True)
    
    
    nice = []
    for string in strings:
        all_pairs = []
        for i in range(len(string)-1):
            all_pairs.append(string[i:i+2])
            
        all_pairs = set(all_pairs)
        all_letters = set([x for x in string])
        
        cond1 = [bool(re.search(x + ".*" + x, string)) for x in all_pairs]
        
        if True not in cond1:
            nice.append(False)
            continue
        
        cond2 = [bool(re.search(x + "." + x, string)) for x in all_letters]
            
        if True not in cond2:
            nice.append(False)
        else:
            nice.append(True)
        
    nice_strings_2 = nice.count(True)
    
    
    return nice_strings_1, nice_strings_2

#%%
# Day 6: Probably a Fire Hazard

def day6(visualize1 = False, visualize2 = False):
    instructions = []
    with open("input6.txt") as f:
        for l in f:
            to_add = []
            new = l.strip().split()
            if "toggle" in new:
                to_add.append("toggle")
            elif "on" in new:
                to_add.append("on")
            elif "off" in new:
                to_add.append("off")
            for i in [-3,-1]:
                to_add.append([int(x) for x in new[i].split(",")])
            instructions.append(to_add)
        
    lights = np.ones([1000,1000]) != 1
    if visualize1:
        plt.imshow(lights)
    for instruction in instructions:
        row_1 = min(instruction[1][0], instruction[2][0])
        row_2 = max(instruction[1][0], instruction[2][0])
        col_1 = min(instruction[1][1], instruction[2][1])
        col_2 = max(instruction[1][1], instruction[2][1])
        if instruction[0] == "on":
            lights[row_1:row_2+1, col_1:col_2+1] = True
        elif instruction[0] == "off":
            lights[row_1:row_2+1, col_1:col_2+1] = False
        elif instruction[0] == "toggle":
            lights[row_1:row_2+1, col_1:col_2+1] = lights[row_1:row_2+1, col_1:col_2+1] == False
            
        if visualize1:
            plt.cla()
            plt.imshow(lights)
            plt.pause(0.01)
            
    lights_on = np.sum(lights)
    
    
    lights = np.zeros([1000,1000])
    if visualize2:
        plt.imshow(lights)
    for instruction in instructions:
        row_1 = min(instruction[1][0], instruction[2][0])
        row_2 = max(instruction[1][0], instruction[2][0])
        col_1 = min(instruction[1][1], instruction[2][1])
        col_2 = max(instruction[1][1], instruction[2][1])
        if instruction[0] == "on":
            lights[row_1:row_2+1, col_1:col_2+1] = lights[row_1:row_2+1, col_1:col_2+1] + 1
        elif instruction[0] == "off":
            lights[row_1:row_2+1, col_1:col_2+1] = lights[row_1:row_2+1, col_1:col_2+1] - 1
            lights[lights < 0] = 0
        elif instruction[0] == "toggle":
            lights[row_1:row_2+1, col_1:col_2+1] = lights[row_1:row_2+1, col_1:col_2+1] + 2
        
        if visualize2:
            plt.cla()
            plt.imshow(lights)
            plt.pause(0.01)
        
    light_sum = int(np.sum(lights))
    
    
    return lights_on, light_sum

#%%
# Day 7: Some Assembly Required

def day7():
    operations = []
    assigns = []
    with open("input7.txt") as f:
        for l in f:
            new = [x for x in l.strip().split()]
            if "AND" in new:
                op_on = []
                for i in [0,2]:
                    if new[i].isnumeric():
                        op_on.append(int(new[i]))
                    else:
                        op_on.append(new[i])
                operations.append(["AND", op_on, new[-1]])
                    
            elif "OR" in new:
                op_on = []
                for i in [0,2]:
                    if new[i].isnumeric():
                        op_on.append(int(new[i]))
                    else:
                        op_on.append(new[i])
                operations.append(["OR", op_on, new[-1]])
                
            elif "RSHIFT" in new:
                op_on = []
                for i in [0,2]:
                    if new[i].isnumeric():
                        op_on.append(int(new[i]))
                    else:
                        op_on.append(new[i])
                operations.append(["RSHIFT", op_on, new[-1]])
                
            elif "LSHIFT" in new:
                op_on = []
                for i in [0,2]:
                    if new[i].isnumeric():
                        op_on.append(int(new[i]))
                    else:
                        op_on.append(new[i])
                operations.append(["LSHIFT", op_on, new[-1]])
                
            elif "NOT" in new:
                if new[1].isnumeric():
                    op_on = int(new[1])
                else:
                    op_on = new[1]
                operations.append(["NOT", [op_on], new[-1]])
                
            else:
                if new[0].isnumeric():
                    assigns.append([int(new[0]), new[2]])
                else:
                    operations.append(["ASSIGN", [new[0]], new[2]])
      
    wires = {}
    for starter in assigns:
        wires[starter[1]] = np.array([starter[0]],dtype="uint16")
    
    found = False
    while True:
        for op in operations:
            if op[-1] in wires.keys():
                continue
            
            involved = op[1]
            ready = []
            inputs = []
            for inv in involved:
                if type(inv) == int:
                    inputs.append(inv)
                    ready.append(True)
                else:
                    if inv in wires.keys():
                        inputs.append(wires[inv])
                        ready.append(True)
                    else:
                        ready.append(False)
               
            if False in ready:
                continue     
            
            if op[0] == "AND":
                wires[op[-1]] = inputs[0] & inputs[1]
            elif op[0] == "OR":
                wires[op[-1]] = inputs[0] | inputs[1]
            elif op[0] =="RSHIFT":
                wires[op[-1]] = inputs[0] >> inputs[1]
            elif op[0] == "LSHIFT":
                wires[op[-1]] = inputs[0] << inputs[1]
            elif op[0] == "NOT":
                wires[op[-1]] = ~inputs[0]
            elif op[0] == "ASSIGN":
                wires[op[-1]] = inputs[0]
                
            if "a" in wires.keys():
                found = True
                break
            
        if found:
            break
    
    wire_a1 = wires["a"][0]
        
    
    wires = {}
    for starter in assigns:
        wires[starter[1]] = np.array([starter[0]],dtype="uint16")
    wires["b"] = wire_a1
    
    found = False
    while True:
        for op in operations:
            if op[-1] in wires.keys():
                continue
            
            involved = op[1]
            ready = []
            inputs = []
            for inv in involved:
                if type(inv) == int:
                    inputs.append(inv)
                    ready.append(True)
                else:
                    if inv in wires.keys():
                        inputs.append(wires[inv])
                        ready.append(True)
                    else:
                        ready.append(False)
               
            if False in ready:
                continue     
            
            if op[0] == "AND":
                wires[op[-1]] = inputs[0] & inputs[1]
            elif op[0] == "OR":
                wires[op[-1]] = inputs[0] | inputs[1]
            elif op[0] =="RSHIFT":
                wires[op[-1]] = inputs[0] >> inputs[1]
            elif op[0] == "LSHIFT":
                wires[op[-1]] = inputs[0] << inputs[1]
            elif op[0] == "NOT":
                wires[op[-1]] = ~inputs[0]
            elif op[0] == "ASSIGN":
                wires[op[-1]] = inputs[0]
                
            if "a" in wires.keys():
                found = True
                break
            
        if found:
            break   
    
    wire_a2 = wires["a"][0]
    
    
    return wire_a1, wire_a2

#%%
# Day 8: Matchsticks

def day8():
    chars = []
    with open("input8.txt") as f:
        while True:
            c = f.read(1)
            if not c:
                break
            chars.append(c)
    
    with open("input8.txt") as f:
        lines = len(f.readlines())
    
    escaped_quotes = 0
    escaped_slashes = 0
    escaped_chars = 0
    
    escaping = False
    for c in chars:
        if c == "\\":
            if escaping:
                escaped_slashes += 1
                escaping = False
            else:
                escaping = True  
                
        elif escaping and c == '"':
            escaped_quotes += 1
            escaping = False   
            
        elif escaping and c == "x":
            escaped_chars += 1
            escaping = False
            
    char_diff = lines*2 + escaped_quotes*1 + escaped_slashes*1 + escaped_chars*3
    
    
    chars_needed = lines*4 + escaped_quotes*2 + escaped_slashes*2 + escaped_chars
    
    
    return char_diff, chars_needed

#%%
# Day 9: All in a Single Night

def day9():
    distances = {}
    locs = set()
    with open("input9.txt") as f:
        for l in f:
            new = l.split()
            distances[(new[0], new[2])] = int(new[4])
            distances[(new[2], new[0])] = int(new[4])
            locs.add(new[0])
            locs.add(new[2])
            
    def path_length(path, distances = distances):
        if len(path) <= 1:
            return 0
        
        pairs = []
        for i in range(len(path)-1):
            pairs.append((path[i], path[i+1]))
            
        dist = 0
        for pair in pairs:
            dist += distances[pair]
        
        return dist   
        
    paths = []
    for loc in locs:
        paths.append([loc])
    
    for _ in range( len(locs)-1):
        new_paths = []
        for path in paths:
            for loc in locs:
                if loc in path:
                    continue
                new_paths.append(path.copy() + [loc])
                
        sorted_paths = sorted(new_paths, key = lambda x: path_length(x))
        paths = sorted_paths[:10]
        
    shortest_dist = path_length(paths[0])
    
    
    paths = []
    for loc in locs:
        paths.append([loc])
    
    for _ in range(len(locs)-1):
        new_paths = []
        for path in paths:
            for loc in locs:
                if loc in path:
                    continue
                new_paths.append(path.copy() + [loc])
                
        sorted_paths = sorted(new_paths, reverse = True, key = lambda x: path_length(x))
        paths = sorted_paths[:10]
        
    longest_dist = path_length(paths[0])
    
    
    return shortest_dist, longest_dist

#%%
# Day 10: Elves Look, Elves Say
    
def day10():
    number = "1113222113"
    
    for r in range(50):
        num = [number[0]]
        count = [1]
        for n in number[1:]:
            if n == num[-1]:
                count[-1] += 1
            else:
                num.append(n)
                count.append(1)
        
        number = "".join([str(count[i])+num[i] for i in range(len(num))])
        
        if r == 39:
            len_num_40 = len(number)
    
    len_num_50 = len(number)
    
    
    return len_num_40, len_num_50

#%%
# Day 11: Corporate Policy

def day11():
    password = "vzbxkghb"
    
    letters = "abcdefghijklmnopqrstuvwxyz"
    triple_runs = []
    for i in range(len(letters)-2):
        triple_runs.append(letters[i:i+3])
        
    def cond1(password, triple_runs = triple_runs):
        found = False
        for run in triple_runs:
            if run in password:
                return True
        return found
    
    bad_letters = "iol"
    def cond2(password, bad_letters = bad_letters):
        all_clear = True
        for letter in bad_letters:
            if letter in password:
                return False
        return all_clear

    doubles = [x+x for x in letters]    
    def cond3(password, doubles = doubles):
        count = 0
        for double in doubles:
            if double in password:
                count += 1
            if count >= 2:
                return True
        return False
    
    from_letter = letters
    to_letter = from_letter[1:] + from_letter[0]
    replacement = dict(zip(list(from_letter), list(to_letter)))
    def increment(password, replacement = replacement, bad_letters = bad_letters):
        password = list(password)
        if not cond2(password):
            bad_inds = []
            for letter in bad_letters:
                if letter in password:
                    bad_inds.append(password.index(letter))
            password[min(bad_inds)] = replacement[password[min(bad_inds)]]
            for i in range(min(bad_inds)+1,len(password)):
                password[i] = "a"
            return "".join(password)        
        
        for i in range(len(password))[::-1]:
            if password[i] != "z":
                password[i] = replacement[password[i]]
                break
            else:
                password[i] = "a"
        
        return "".join(password)
    
    while not (cond1(password) and cond2(password) and cond3(password)):
        password = increment(password) 
    password1 = password
    
    
    password = increment(password)
    while not (cond1(password) and cond2(password) and cond3(password)):
        password = increment(password)    
    password2 = password
    
    
    return password1, password2

#%%
# Day 12: JSAbacusFramework.io

def day12():
    with open("input12.txt") as f:
        json = f.read()
        
    numbers = re.findall(r"-{0,1}\d+", json)
    numbers = [int(x) for x in numbers]
    
    sum_num = sum(numbers)
    
    
    def inside(range1, range2):
        if range1[0] > range2[0] and range1[1] < range2[1]:
            return True
        else:
            return False
    
    json_six = []
    for i in range(len(json)-5):
        json_six.append(json[i:i+6])
        
    red_indices = [i for i, x in enumerate(json_six) if x == ':"red"']
    
    red_index_bounds = []
    for red_index in red_indices:
        opening_ind = red_index
        opening_needed = 1
        while True:
            opening_ind -= 1
            if json[opening_ind] == "{":
                opening_needed -= 1
            elif json[opening_ind] == "}":
                opening_needed += 1
            if opening_needed == 0:
                break
            
        closing_ind = red_index
        closing_needed = 1
        while True:
            closing_ind += 1
            if json[closing_ind] == "}":
                closing_needed -= 1
            elif json[closing_ind] == "{":
                closing_needed += 1
            if closing_needed == 0:
                break
    
        red_index_bounds.append([opening_ind,closing_ind+1])
        
    red_bounds = []
    for bound in red_index_bounds:
        contained = [inside(bound,x) for x in red_index_bounds]
        if True not in contained:
            red_bounds.append(bound)
            
    red_bounds = list(set([tuple(x) for x in red_bounds]))
    
    red_text = [json[x[0]:x[1]] for x in red_bounds]
    red_num = [re.findall(r"-{0,1}\d+", x) for x in red_text]
    red_num = [[int(x) for x in y] for y in red_num]
    red_sum = sum([sum(x) for x in red_num])
    
    sum_num_no_red = sum_num - red_sum
    
    
    return sum_num, sum_num_no_red

#%%
# Day 13: Knights of the Dinner Table

def day13():
    happy_pairs_orig = {}
    people = []
    with open("input13.txt") as f:
        for l in f:
            new = l.split()
            pair = tuple([new[0], new[-1][:-1]])
            people.append(new[0])
            if "gain" in new:
                happy_pairs_orig[pair] =  int(new[3])
            else:
                happy_pairs_orig[pair] =  -int(new[3]) 
                
    people = list(set(people))
    num_people = len(people)
                
    happy_pairs = {}
    for pair in happy_pairs_orig.keys():
        happy_pairs[pair] = happy_pairs_orig[pair] + happy_pairs_orig[pair[::-1]]
        
    def score_seating(order, happy_pairs = happy_pairs, num_people = num_people):
        if len(order) < 2:
            return 0
        
        pairs = []
        for i in range(len(order)-1):
            pairs.append((order[i],order[i+1]))
        if len(order) == num_people:
            pairs.append((order[-1],order[0]))
            
        score = 0
        for pair in pairs:
            score += happy_pairs[pair]
        return score
        
    orders = [[p] for p in people]
    for i in range(num_people-1):
        new_orders = []
        for order in orders:
            for person in people:
                if person in order:
                    continue
                new_orders.append(order.copy() + [person])
                
        orders_sorted = sorted(new_orders, reverse = True, key = lambda x: score_seating(x))
        orders = orders_sorted[:10]
        
    max_score = score_seating(orders[0])
    
    
    for person in people:
        happy_pairs[(person, "Me")] = 0
        happy_pairs[("Me", person)] = 0
    
    people.append("Me")
    num_people = len(people)
    
    orders = [[p] for p in people]
    for i in range(num_people-1):
        new_orders = []
        for order in orders:
            for person in people:
                if person in order:
                    continue
                new_orders.append(order.copy() + [person])
                
        orders_sorted = sorted(new_orders, reverse = True, key = lambda x: score_seating(x))
        orders = orders_sorted[:10]
        
    max_score_with_me = score_seating(orders[0])
    
    return max_score, max_score_with_me

#%%
# Day 14: Reindeer Olympics

def day14():
    reindeer = {}
    with open("input14.txt") as f:
        for l in f:
            new = l.split()
            speed = int(new[3])
            move_secs = int(new[6])
            rest_secs = int(new[13])
            reindeer[new[0]] = [[speed, move_secs, rest_secs], move_secs + rest_secs]
            
    end_time = 2503
    
    total = []
    for rd in reindeer.keys():
        cycles = end_time//reindeer[rd][1]
        extra = min(end_time%reindeer[rd][1], reindeer[rd][0][1])
        total.append(cycles*(reindeer[rd][0][0]*reindeer[rd][0][1]) + extra*reindeer[rd][0][0])
    
    max_dist = max(total)
    
    
    points = np.zeros(len(reindeer))
    dist = np.zeros(len(reindeer))
    names = list(reindeer.keys())
    
    for t in range(1,end_time+1):
        for rd in names:
            cycles = t//reindeer[rd][1]
            extra = min(t%reindeer[rd][1], reindeer[rd][0][1])
            dist[names.index(rd)] = cycles*(reindeer[rd][0][0]*reindeer[rd][0][1]) + extra*reindeer[rd][0][0]
            
        points[dist == max(dist)] += 1
        
    max_points = int(max(points))
    
    
    return max_dist, max_points

#%%
# Day 15: Science for Hungry People

def day15():
    ingredients = {}
    with open("input15.txt") as f:
        for l in f:
            new = l.split()
            name = new[0][:-1]
            cap = int(new[2][:-1])
            dur = int(new[4][:-1])
            flav = int(new[6][:-1])
            text = int(new[8][:-1])
            cal = int(new[10])
            ingredients[name] = {"cap":cap, "dur":dur, "flav":flav, "text":text, "cal":cal}
            
    def score(recipe_summary, ingredients = ingredients):
        names = list(ingredients.keys())
        cap = 0
        dur = 0
        flav = 0
        text = 0
        for name in names:
            tsp = recipe_summary[names.index(name)]
            cap += tsp*ingredients[name]["cap"]
            dur += tsp*ingredients[name]["dur"]
            flav += tsp*ingredients[name]["flav"]
            text += tsp*ingredients[name]["text"]
        if cap <= 0 or dur <= 0 or flav <= 0 or text <= 0:
            return 0
        else:
            return cap*dur*flav*text
        
    def calories(recipe_summary, ingredients = ingredients):
        names = list(ingredients.keys())
        cal = 0
        for name in names:
            tsp = recipe_summary[names.index(name)]
            cal += tsp*ingredients[name]["cal"]
        return cal
        
            
    names = list(ingredients.keys())
    recipes = [[0,0,0,0]]
    for _ in range(100):
        new_recipes = []
        for recipe in recipes:
            for ingredient in names:
                new = recipe.copy()
                new[names.index(ingredient)] += 1
                new_recipes.append(new)
        
        new_recipes = [list(x) for x in set([tuple(x) for x in new_recipes])]
        sorted_recipes = sorted(new_recipes, reverse = True, key = lambda x: score(x))
        recipes = sorted_recipes[:375]
        
    max_score = score(recipes[0])
    
    
    match_cal = [calories(x) == 500 for x in recipes]
    max_score_cals = score(recipes[match_cal.index(True)])
    

    return max_score, max_score_cals
        
#%%
# Day 16: Aunt Sue

def day16():
    aunts = []
    with open("input16.txt") as f:
        for l in f:
            new = l.split()
            to_add = {}
            if "children:" in new:
                to_add["children"] = int(new[new.index("children:")+1].replace(",",""))
            else:
                to_add["children"] = None
            
            if "cats:" in new:
                to_add["cats"] = int(new[new.index("cats:")+1].replace(",",""))
            else:
                to_add["cats"] = None
            
            if "samoyeds:" in new:
                to_add["samoyeds"] = int(new[new.index("samoyeds:")+1].replace(",",""))
            else:
                to_add["samoyeds"] = None
            
            if "pomeranians:" in new:
                to_add["pomeranians"] = int(new[new.index("pomeranians:")+1].replace(",",""))
            else:
                to_add["pomeranians"] = None
            
            if "akitas:" in new:
                to_add["akitas"] = int(new[new.index("akitas:")+1].replace(",",""))
            else:
                to_add["akitas"] = None
            
            if "vizslas:" in new:
                to_add["vizslas"] = int(new[new.index("vizslas:")+1].replace(",",""))
            else:
                to_add["vizslas"] = None
            
            if "goldfish:" in new:
                to_add["goldfish"] = int(new[new.index("goldfish:")+1].replace(",",""))
            else:
                to_add["goldfish"] = None
            
            if "trees:" in new:
                to_add["trees"] = int(new[new.index("trees:")+1].replace(",",""))
            else:
                to_add["trees"] = None
            
            if "cars:" in new:
                to_add["cars"] = int(new[new.index("cars:")+1].replace(",",""))
            else:
                to_add["cars"] = None
            
            if "perfumes:" in new:
                to_add["perfumes"] = int(new[new.index("perfumes:")+1].replace(",",""))
            else:
                to_add["perfumes"] = None
                
            aunts.append(to_add)
            
    answer = {"children": 3,
    "cats": 7,
    "samoyeds": 2,
    "pomeranians": 3,
    "akitas": 0,
    "vizslas": 0,
    "goldfish": 5,
    "trees": 3,
    "cars": 2,
    "perfumes": 1}
    
    for i, aunt in enumerate(aunts):
        found_aunt = True
        for clue in aunt.keys():
            if aunt[clue] == None:
                continue
            if aunt[clue] != answer[clue]:
                found_aunt = False
                break
        if found_aunt:
            break
            
    sue_number = i+1
    
    
    for i, aunt in enumerate(aunts):
        found_aunt = True
        for clue in aunt.keys():
            if aunt[clue] == None:
                continue
            if clue in ["cats", "trees"]:
                if aunt[clue] <= answer[clue]:
                    found_aunt = False
                    break
            elif clue in ["pomeranians", "goldfish"]:
                if aunt[clue] >= answer[clue]:
                    found_aunt = False
                    break
            else:
                if aunt[clue] != answer[clue]:
                    found_aunt = False
                    break
        if found_aunt:
            break
            
    sue_number_2 = i+1
    
    
    return sue_number, sue_number_2
            
#%%
# Day 17: No Such Thing as Too Much

def day17():    
    containers = []
    with open("input17.txt") as f:
        for l in f:
            containers.append(int(l.strip()))
            
    liters = 150
            
    containers = np.array(containers) 
    indices = list(range(len(containers)))
    
    exact_fits = 0
    smallest = False
    smallest_cutoff_off = True
    for i in range(1,len(containers)+1):
        container_nums = list(combinations(indices, i))
        for combo_container_nums in container_nums:
            combo_inds_log = [x in combo_container_nums for x in indices]
            if sum(containers[combo_inds_log]) == liters:
                smallest = True
                exact_fits += 1
        if smallest and smallest_cutoff_off:
            exact_fits_smallest = exact_fits
            smallest_cutoff_off = False
            
    exact_fits_all = exact_fits
    
    
    return exact_fits_all, exact_fits_smallest

#%%
# Day 18: Like a GIF For Your Yard

def day18(visualize = False):
    lights = []
    with open("input18.txt") as f:
        for l in f:
            new = l.strip().replace(".","0").replace("#","1")
            lights.append([int(x) for x in new])
    og_lights = np.array(lights)
    
    lights = np.copy(og_lights)
    
    
    def shifted_lights(lights):
        C = lights
        C = np.hstack([np.zeros([C.shape[0],1]),C,np.zeros([C.shape[0],1])])
        C = np.vstack([np.zeros([1,C.shape[1]]),C,np.zeros([1,C.shape[1]])])
        
        NW = C[1:,1:]
        NW = np.hstack([NW, np.zeros([NW.shape[0],1])])
        NW = np.vstack([NW, np.zeros([1,NW.shape[1]])])
        
        N = C[1:,:]
        N = np.vstack([N, np.zeros([1,N.shape[1]])])
        
        NE = C[1:,:-1]
        NE = np.hstack([np.zeros([NE.shape[0],1]), NE])
        NE = np.vstack([NE, np.zeros([1,NE.shape[1]])])
        
        W = C[:,1:]
        W = np.hstack([W, np.zeros([W.shape[0],1])])
        
        E = C[:,:-1]
        E = np.hstack([np.zeros([E.shape[0],1]), E])
        
        SW = C[:-1,1:]
        SW = np.vstack([np.zeros([1,SW.shape[1]]), SW])
        SW = np.hstack([SW, np.zeros([SW.shape[0],1])])
        
        S = C[:-1,:]
        S = np.vstack([np.zeros([1,S.shape[1]]), S])
        
        SE = C[:-1,:-1]
        SE = np.vstack([np.zeros([1,SE.shape[1]]), SE])
        SE = np.hstack([np.zeros([SE.shape[0],1]), SE])
                        
    
        return C, NW, N, NE, W, E, SW, S, SE
    
    if visualize:
        fig, axes = plt.subplots(1,2)
        axes = axes.ravel()
        axes[0].imshow(lights)
        axes[1].imshow(lights)
    for i in range(100):
        C, NW, N, NE, W, E, SW, S, SE = shifted_lights(lights)
        
        number_of_neighbors_on = NW[1:-1,1:-1] + N[1:-1,1:-1] + NE[1:-1,1:-1] + W[1:-1,1:-1] + E[1:-1,1:-1] + SW[1:-1,1:-1] + S[1:-1,1:-1] + SE[1:-1,1:-1]
        
        turn_off1 = lights == 1
        turn_off2 = number_of_neighbors_on != 2
        turn_off3 = number_of_neighbors_on != 3
        turn_off = turn_off1 * turn_off2 * turn_off3
        
        turn_on1 = lights == 0
        turn_on2 = number_of_neighbors_on == 3
        turn_on = turn_on1 * turn_on2
        
        lights[turn_off] = 0
        lights[turn_on] = 1
        
        if visualize:
            axes[0].cla()
            axes[0].imshow(lights)
            plt.pause(0.01)
    
    lights_on = np.sum(lights)
    
    
    lights = np.copy(og_lights)
    lights[0,0] = 1
    lights[-1,0] = 1
    lights[0,-1] = 1
    lights[-1,-1] = 1
    if visualize:
        axes[1].imshow(lights)
    for i in range(100):
        C, NW, N, NE, W, E, SW, S, SE = shifted_lights(lights)
        
        number_of_neighbors_on = NW[1:-1,1:-1] + N[1:-1,1:-1] + NE[1:-1,1:-1] + W[1:-1,1:-1] + E[1:-1,1:-1] + SW[1:-1,1:-1] + S[1:-1,1:-1] + SE[1:-1,1:-1]
        
        turn_off1 = lights == 1
        turn_off2 = number_of_neighbors_on != 2
        turn_off3 = number_of_neighbors_on != 3
        turn_off = turn_off1 * turn_off2 * turn_off3
        
        turn_on1 = lights == 0
        turn_on2 = number_of_neighbors_on == 3
        turn_on = turn_on1 * turn_on2
        
        lights[turn_off] = 0
        lights[turn_on] = 1
        
        lights[0,0] = 1
        lights[-1,0] = 1
        lights[0,-1] = 1
        lights[-1,-1] = 1
        
        if visualize:
            axes[1].cla()
            axes[1].imshow(lights)
            plt.pause(0.01)
    
    lights_on_2 = np.sum(lights)
    
    return lights_on, lights_on_2

#%%
# Day 19: Medicine for Rudolph

def day19():
    replacements = []
    with open("input19.txt") as f:
        for l in f:
            if "=>" in l:
                new = l.split()
                replacements.append([new[0],new[2]])
            else:
                molecule = l.strip()
    
    def get_new_molecules(molecule, replacements = replacements):
        new_molecules = []
        for replacement in replacements:
            rep_len = len(replacement[0])
            match_inds = []
            for i in range(len(molecule)-rep_len+1):
                if molecule[i:i+rep_len] == replacement[0]:
                    match_inds.append(i)
                    
            for i in match_inds:
                new_molecule = molecule[:i] + replacement[1] + molecule[i+rep_len:]
                new_molecules.append(new_molecule)
                
        new_molecules = list(set(new_molecules))
        return new_molecules
    
    num_new_molecules = len(get_new_molecules(molecule))
    
    
    made_molecules = [molecule]
    count = 0
    while "e" not in made_molecules:
        count += 1
        new_molecules = []
        for made_mol in made_molecules:
            new_molecules = new_molecules + get_new_molecules(made_mol, [[x[1],x[0]] for x in replacements])
        
        new_molecules = list(set(new_molecules))
            
        molecules_sorted = sorted(new_molecules, key = lambda x: len(x))
        made_molecules = molecules_sorted[:1]
        
        
    return num_new_molecules, count
            
#%%
# Day 20: Infinite Elves and Infinite Houses

def day20():    
    num_pres = 36000000
        
    def factors(x):
        i = 1
        facts = []
        while i <= np.sqrt(x):
            if x%i == 0:
                if x/i == i:
                    facts.append(i)
                else:
                    facts.append(i)
                    facts.append(int(x/i))
            i += 1
        return sorted(facts)
    
    n = 600
    while True:
        presents = 10*sum(factors(n))
        if presents >= num_pres:
            break
        n += 600
        
    lowest_house = n
    
    
    while True:
        facts = factors(n)
        facts_50 = [i for i in facts if i*50 >= facts[-1]]
        presents = 11*sum(facts_50)
        if presents >= num_pres:
            break
        n += 60
        
    lowest_house_2 = n
    
    
    return lowest_house, lowest_house_2

#%%
# Day 21: RPG Simulator 20XX

def day21():
    boss = {}
    with open("input21.txt") as f:
        for l in f:
            if "Hit Points" in l:
                boss["HP"] = int(l.split()[-1])
            elif "Damage" in l:
                boss["damage"] = int(l.split()[-1])
            elif "Armor" in l:
                boss["armor"] = int(l.split()[-1])
                
    weapons = {}
    weapons["Dagger"] = {"cost":8, "damage":4, "armor":0}
    weapons["Shortsword"] = {"cost":10, "damage":5, "armor":0}
    weapons["Warhammer"] = {"cost":25, "damage":6, "armor":0}
    weapons["Longsword"] = {"cost":40, "damage":7, "armor":0}
    weapons["Greataxe"] = {"cost":74, "damage":8, "armor":0}
    
    armor = {}
    armor["Leather"] = {"cost":13, "damage":0, "armor":1}
    armor["Chainmail"] = {"cost":31, "damage":0, "armor":2}
    armor["Splintmail"] = {"cost":53, "damage":0, "armor":3}
    armor["Bandedmail"] = {"cost":75, "damage":0, "armor":4}
    armor["Platemail"] = {"cost":102, "damage":0, "armor":5}
    
    rings = {}
    rings["Damage +1"] = {"cost":25, "damage":1, "armor":0}
    rings["Damage +2"] = {"cost":50, "damage":2, "armor":0}
    rings["Damage +3"] = {"cost":100, "damage":3, "armor":0}
    rings["Defense +1"] = {"cost":20, "damage":0, "armor":1}
    rings["Defense +2"] = {"cost":40, "damage":0, "armor":2}
    rings["Defense +3"] = {"cost":80, "damage":0, "armor":3}
    
    def play(player, boss = boss, verbose = False):
        player = player.copy()
        boss = boss.copy()
        while True:
            if verbose:
                print(f'Boss HP: {boss["HP"]}')
                print(f'Player HP: {player["HP"]}')
                
            boss["HP"] -= max(1, player["damage"]-boss["armor"])
            if boss["HP"] <= 0:
                return True
            player["HP"] -= max(1, boss["damage"]-player["armor"])
            if player["HP"] <= 0:
                return False
    
    ring_options = list(combinations(rings.keys(), 2)) + list(combinations(rings.keys(), 1)) + [tuple()]
    armor_options = list(armor.keys()) + [None]
    weapon_options = list(weapons.keys())
    
    rings[None] = {"cost":0, "damage":0, "armor":0}
    armor[None] = {"cost":0, "damage":0, "armor":0}
    
    cheapest_win_cost = np.inf
    most_expensive_loss_cost = 0
    for ring_choice in ring_options:
        ring_cost = sum([rings[x]["cost"] for x in ring_choice])        
        ring_damage = sum([rings[x]["damage"] for x in ring_choice])      
        ring_armor = sum([rings[x]["armor"] for x in ring_choice])                
        for armor_choice in armor_options:
            armor_cost = armor[armor_choice]["cost"]
            armor_armor = armor[armor_choice]["armor"]                
            for weapon_choice in weapon_options:
                weapon_cost = weapons[weapon_choice]["cost"]
                total_cost = ring_cost + armor_cost + weapon_cost
                
                if total_cost > cheapest_win_cost and total_cost < most_expensive_loss_cost:
                    continue
                
                weapon_damage = weapons[weapon_choice]["damage"]
                total_damage = ring_damage + weapon_damage
                
                total_armor = ring_armor + armor_armor
                
                player = {"HP":100, "damage":total_damage, "armor":total_armor}
                
                win = play(player)
                if win:
                    if total_cost < cheapest_win_cost:
                        cheapest_win_cost = total_cost
                else:
                    if total_cost > most_expensive_loss_cost:
                        most_expensive_loss_cost = total_cost                    
                
    return cheapest_win_cost, most_expensive_loss_cost
            
#%%
# Day 22: Wizard Simulator 20XX

def day22():
    boss_info = {}
    with open("input22.txt") as f:
        for l in f:
            if "Hit Points" in l:
                boss_info["HP"] = int(l.split()[-1])
            elif "Damage" in l:
                boss_info["damage"] = int(l.split()[-1])
    
    class boss():
        def __init__(self, hp = boss_info["HP"], damage = boss_info["damage"]):
            self.hp = hp
            self.damage = damage
        
        def __str__(self):
            return f"{self.hp}, {self.damage}"
        
        def __repr__(self):
            return self.__str__()
        
        def copy(self):
            return boss(hp = self.hp, damage = self.damage)
        
        def attack(self, player):
            player.hp -= max(1, self.damage - player.armor)
            
    class player():
        def __init__(self, hp = 50, armor = 0, mana = 500, mana_used = 0, shield_timer = 0, poison_timer = 0, recharge_timer = 0):
            self.hp = hp
            self.armor = armor
            self.mana = mana
            self.mana_used = mana_used
            self.shield_timer = shield_timer
            self.poison_timer = poison_timer
            self.recharge_timer = recharge_timer
            
        def __str__(self):
            return f"{self.hp}, {self.mana}, {self.shield_timer}, {self.poison_timer}, {self.recharge_timer}, {self.mana_used}"
        
        def __repr__(self):
            return self.__str__()
            
        def copy(self):
            return player(hp = self.hp,
                          armor = self.armor,
                          mana = self.mana,
                          mana_used = self.mana_used,
                          shield_timer = self.shield_timer,
                          poison_timer = self.poison_timer,
                          recharge_timer = self.recharge_timer)
        
        def can_magic_missile(self):
            if self.mana >= 53:
                return True
            else:
                return False
        
        def magic_missile(self, boss):
            self.mana -= 53
            self.mana_used += 53
            boss.hp -= 4
            
        def can_drain(self):
            if self.mana >= 73:
                return True
            else:
                return False
            
        def drain(self, boss):
            self.mana -= 73
            self.mana_used += 73
            self.hp += 2
            boss.hp -= 2
            
        def can_shield(self):
            if self.shield_timer == 0 and self.mana >= 113:
                return True
            else:
                return False
        
        def shield(self):
            self.mana -= 113
            self.mana_used += 113
            self.shield_timer = 6
            
        def can_poison(self):
            if self.poison_timer == 0 and self.mana >= 173:
                return True
            else:
                return False
        
        def poison(self):
            self.mana -= 173
            self.mana_used += 173
            self.poison_timer = 6
        
        def can_recharge(self):
            if self.recharge_timer == 0 and self.mana >= 229:
                return True
            else:
                return False
            
        def recharge(self):
            self.mana -= 229
            self.mana_used += 229
            self.recharge_timer = 5
            
        def effects_happen(self, boss):
            if self.shield_timer > 0:
                self.armor = 7
            else:
                self.armor = 0
                
            if self.poison_timer > 0:
                boss.hp -= 3
                
            if self.recharge_timer > 0:
                self.mana += 101
                
            self.shield_timer = max(0, self.shield_timer - 1)
            self.poison_timer = max(0, self.poison_timer - 1)
            self.recharge_timer = max(0, self.recharge_timer - 1)
            
    def winner(player_boss_list):
        player = player_boss_list[0]
        boss = player_boss_list[1]
        
        if player.hp <= 0 or player.mana < 53:
            return False
        elif boss.hp <= 0:
            return True
        else:
            return None
    
    def get_least_mana_win(mode):
        least_mana_used_win = np.inf
        states = [[player(), boss()]]
        while True:
            new_states = []
            for state in states:
                if state[0].mana_used > least_mana_used_win:
                    continue
                
                if mode == "hard":
                    state[0].hp -= 1
                    if winner(state) != None:
                        continue
                
                if state[0].can_magic_missile():
                    new_player = state[0].copy()
                    new_boss = state[1].copy()
                    new_player.magic_missile(new_boss)
                    new_player.effects_happen(new_boss)
                    if winner([new_player, new_boss]) != None:
                        if winner([new_player, new_boss]):
                            least_mana_used_win = min(least_mana_used_win, new_player.mana_used)                   
                    new_player.effects_happen(new_boss)
                    new_boss.attack(new_player)
                    if winner([new_player, new_boss]) == None:
                        new_states.append([new_player, new_boss])
                    
                if state[0].can_drain():
                    new_player = state[0].copy()
                    new_boss = state[1].copy()
                    new_player.drain(new_boss)
                    new_player.effects_happen(new_boss)
                    if winner([new_player, new_boss]) != None:
                        if winner([new_player, new_boss]):
                            least_mana_used_win = min(least_mana_used_win, new_player.mana_used)                   
                    new_player.effects_happen(new_boss)
                    new_boss.attack(new_player)
                    if winner([new_player, new_boss]) == None:
                        new_states.append([new_player, new_boss])
                    
                if state[0].can_shield():
                    new_player = state[0].copy()
                    new_boss = state[1].copy()
                    new_player.shield()
                    new_player.effects_happen(new_boss)
                    if winner([new_player, new_boss]) != None:
                        if winner([new_player, new_boss]):
                            least_mana_used_win = min(least_mana_used_win, new_player.mana_used)                   
                    new_player.effects_happen(new_boss)
                    new_boss.attack(new_player)
                    if winner([new_player, new_boss]) == None:
                        new_states.append([new_player, new_boss])
                        
                if state[0].can_poison():
                    new_player = state[0].copy()
                    new_boss = state[1].copy()
                    new_player.poison()
                    new_player.effects_happen(new_boss)
                    if winner([new_player, new_boss]) != None:
                        if winner([new_player, new_boss]):
                            least_mana_used_win = min(least_mana_used_win, new_player.mana_used)                   
                    new_player.effects_happen(new_boss)
                    new_boss.attack(new_player)
                    if winner([new_player, new_boss]) == None:
                        new_states.append([new_player, new_boss])
            
                if state[0].can_recharge():
                    new_player = state[0].copy()
                    new_boss = state[1].copy()
                    new_player.recharge()
                    new_player.effects_happen(new_boss)
                    if winner([new_player, new_boss]) != None:
                        if winner([new_player, new_boss]):
                            least_mana_used_win = min(least_mana_used_win, new_player.mana_used)                   
                    new_player.effects_happen(new_boss)
                    new_boss.attack(new_player)
                    if winner([new_player, new_boss]) == None:
                        new_states.append([new_player, new_boss])
                        
            if len(new_states) == 0:
                break
            sorted_states = sorted(new_states, key = lambda x: x[0].mana_used)
            if mode == "easy":
                cutoff = 30000
            elif mode == "hard":
                cutoff = 10000
            states = sorted_states[:cutoff]
            
        return least_mana_used_win
    
    
    return get_least_mana_win("easy"), get_least_mana_win("hard")

#%%
# Day 23: Opening the Turing Lock

def day23():
    commands = []
    with open("input23.txt") as f:
        for l in f:
            if "jie" in l or "jio" in l:
                new = l.split()
                commands.append([new[0],new[1][:-1],int(new[2])])
            else:
                new = l.split()
                if "jmp" in new:
                    commands.append([new[0],int(new[1])])
                else:
                    commands.append([new[0], new[1]])
                    
    def calculate_b(a_start, commands = commands):
        registers = {"a":a_start, "b":0}
        i= 0
        while i in range(len(commands)):
            command = commands[i]
            if command[0] == "hlf":
                registers[command[1]] = registers[command[1]]/2
            elif command[0] == "tpl":
                registers[command[1]] = registers[command[1]]*3
            elif command[0] == "inc":
                registers[command[1]] += 1
            elif command[0] == "jmp":
                i += command[1]
                continue
            elif command[0] == "jie":
                if registers[command[1]]%2 == 0:
                    i += command[2]
                    continue
            elif command[0] == "jio":
                if registers[command[1]] == 1:
                    i += command[2]
                    continue
            i += 1
            
        return registers["b"]
    
    
    return calculate_b(0), calculate_b(1)

#%%
# Day 24: It Hangs in the Balance

def day24():
    weights = []
    with open("input24.txt") as f:
        for l in f:
            weights.append(int(l))
    weights = np.array(sorted(weights, reverse = True))
            
    sum_weights = sum(weights)
    
    def best_front_quantum(sections):
        must_weigh = sum_weights/sections
        indices = list(range(len(weights)))
        for num in range(1,len(weights)+1):
            all_combos = list(combinations(indices, num))
            front_packs = []
            found = False
            for combo in all_combos:
                combo_log = np.array([x in combo for x in indices])
                front_weight = sum(weights*combo_log)
                if front_weight == must_weigh:
                    found = True
                    front_packs.append(list(weights[combo_log]))
            if found:
                break
            
        return min([np.prod(x, dtype = np.int64) for x in front_packs])
    
    return best_front_quantum(3), best_front_quantum(4)

#%%
# Day 25: Let It Snow

def day25():
    with open("input25.txt") as f:
        new = f.read().split()
        code_loc = (int(new[15][:-1]), int(new[17][:-1]))
        
    mult_by = 252533
    div_by_remainder = 33554393
    
    def next_loc(last):
        if last[0] == 1:
            new = (last[1]+1,1)
        else:
            new = (last[0]-1, last[1]+1)
        return new
    
    loc = (6,6)
    code = 27995004
    while loc != code_loc:
        loc = next_loc(loc)
        code = (code*mult_by)%div_by_remainder
        
    return code