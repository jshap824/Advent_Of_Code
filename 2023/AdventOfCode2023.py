# %%
from time import time
from math import prod, sqrt
import numpy as np
from itertools import product
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation

# %% Timer Decorator


def time_this_func(func):
    def timed_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        print(f"{time()-t1:0.3f} s runtime")
        return result
    return timed_func

# %% Day 1: Trebuchet?!


@time_this_func
def day1():
    # Input parsing
    with open("day1.txt") as f:
        raw = f.readlines()

    # Part 1
    numerics_1 = [[x for x in y if x.isnumeric()] for y in raw]
    coordinates_1 = [int(x[0]+x[-1]) for x in numerics_1]

    total_1 = sum(coordinates_1)

    # Part 2
    number_spellings = {"one": "o1e",
                        "two": "t2o",
                        "three": "t3e",
                        "four": "f4r",
                        "five": "f5e",
                        "six": "s6x",
                        "seven": "s7n",
                        "eight": "e8t",
                        "nine": "n9e"}

    decoded = []
    for entry in raw:
        decoded.append(entry)
        for number_spelling, number in number_spellings.items():
            decoded[-1] = decoded[-1].replace(number_spelling, number)

    numerics_2 = [[x for x in y if x.isnumeric()] for y in decoded]
    coordinates_2 = [int(x[0]+x[-1]) for x in numerics_2]

    total_2 = sum(coordinates_2)

    # Return results
    return total_1, total_2

# %%: Day 2: Cube Conundrum


@time_this_func
def day2():
    # Input parsing
    with open("day2.txt") as f:
        raw = f.readlines()

    games_info = {}
    for line in raw:
        info = line.split(":")
        game_num = int(info[0].split()[-1])
        game_drawings = []
        for drawing in info[1].split(";"):
            drawing_details = [x.split() for x in drawing.split(",")]
            drawing_details = {x[1]: int(x[0]) for x in drawing_details}
            game_drawings.append(drawing_details)
        games_info[game_num] = game_drawings

    # Part 1
    games_min_needed = {}
    for game_num, game_drawings in games_info.items():
        games_min_needed[game_num] = {"red": 0, "green": 0, "blue": 0}
        for game_drawing in game_drawings:
            for color in ["red", "green", "blue"]:
                if color in game_drawing and game_drawing[color] > games_min_needed[game_num][color]:
                    games_min_needed[game_num][color] = game_drawing[color]

    valid_game_nums = []
    for game_num, mins in games_min_needed.items():
        if mins["red"] <= 12 and mins["green"] <= 13 and mins["blue"] <= 14:
            valid_game_nums.append(game_num)

    valid_game_nums_sum = sum(valid_game_nums)

    # Part 2
    game_powers = [prod(x.values()) for x in games_min_needed.values()]

    game_powers_sum = sum(game_powers)

    # Return results
    return valid_game_nums_sum, game_powers_sum

# %% Day 3: Gear Ratios


@time_this_func
def day3():
    # Input parsing
    with open("day3.txt") as f:
        raw = f.readlines()

    schematic = np.array([[c for c in line[:-1]] for line in raw])

    # Part 1
    non_symbols = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "."}

    schem_rows = schematic.shape[0]
    schem_cols = schematic.shape[1]

    def is_part(coordinates):
        for coord in coordinates:
            if coord[0] != 0 and schematic[coord[0]-1, coord[1]] not in non_symbols:
                return True
            if coord[0] + 1 != schem_rows and schematic[coord[0]+1, coord[1]] not in non_symbols:
                return True
            if coord[1] != 0 and schematic[coord[0], coord[1]-1] not in non_symbols:
                return True
            if coord[1] + 1 != schem_cols and schematic[coord[0], coord[1]+1] not in non_symbols:
                return True

            if coord[0] != 0 and coord[1] != 0 and schematic[coord[0]-1, coord[1]-1] not in non_symbols:
                return True
            if coord[0] + 1 != schem_rows and coord[1] != 0 and schematic[coord[0]+1, coord[1]-1] not in non_symbols:
                return True
            if coord[0] + 1 != schem_rows and coord[1] + 1 != schem_cols and schematic[coord[0]+1, coord[1]+1] not in non_symbols:
                return True
            if coord[0] != 0 and coord[1] + 1 != schem_cols and schematic[coord[0]-1, coord[1]+1] not in non_symbols:
                return True

        return False

    possible_part_coordinates = [[]]
    for i in range(schem_rows):
        for j in range(schem_cols):
            if schematic[i, j].isnumeric():
                possible_part_coordinates[-1].append((i, j))
            elif possible_part_coordinates[-1] != []:
                possible_part_coordinates.append([])

        if possible_part_coordinates[-1] != []:
            possible_part_coordinates.append([])

    confirmed_parts = []
    confirmed_parts_coordinates = []
    for coordinates in possible_part_coordinates:
        if is_part(coordinates):
            confirmed_parts_coordinates.append(coordinates)
            confirmed_parts.append(int("".join([str(schematic[c0, c1]) for c0, c1 in coordinates])))

    confirmed_parts_sum = sum(confirmed_parts)

    # Part 2
    def part_connects_to_gear(coordinates):
        connects_to = set()
        for coord in coordinates:
            if coord[0] != 0 and schematic[coord[0]-1, coord[1]] == "*":
                connects_to.add((coord[0]-1, coord[1]))
            if coord[0] + 1 != schem_rows and schematic[coord[0]+1, coord[1]] == "*":
                connects_to.add((coord[0]+1, coord[1]))
            if coord[1] != 0 and schematic[coord[0], coord[1]-1] == "*":
                connects_to.add((coord[0], coord[1]-1))
            if coord[1] + 1 != schem_cols and schematic[coord[0], coord[1]+1] == "*":
                connects_to.add((coord[0], coord[1]+1))

            if coord[0] != 0 and coord[1] != 0 and schematic[coord[0]-1, coord[1]-1] == "*":
                connects_to.add((coord[0]-1, coord[1]-1))
            if coord[0] + 1 != schem_rows and coord[1] != 0 and schematic[coord[0]+1, coord[1]-1] == "*":
                connects_to.add((coord[0]+1, coord[1]-1))
            if coord[0] + 1 != schem_rows and coord[1] + 1 != schem_cols and schematic[coord[0]+1, coord[1]+1] == "*":
                connects_to.add((coord[0]+1, coord[1]+1))
            if coord[0] != 0 and coord[1] + 1 != schem_cols and schematic[coord[0]-1, coord[1]+1] == "*":
                connects_to.add((coord[0]-1, coord[1]+1))

        return connects_to

    part_coords_to_asterisks = {}
    for coordinates in confirmed_parts_coordinates:
        part_coords_to_asterisks[tuple(coordinates)] = part_connects_to_gear(coordinates)

    asterisks = {}
    for part_coords, asterisk_set in part_coords_to_asterisks.items():
        for asterisk in asterisk_set:
            if asterisk not in asterisks:
                asterisks[asterisk] = {int("".join([str(schematic[c0, c1])
                                                    for c0, c1 in part_coords]))}
            else:
                asterisks[asterisk].add(int("".join([str(schematic[c0, c1])
                                                     for c0, c1 in part_coords])))

    gear_ratios = [prod(x) for x in asterisks.values() if len(x) == 2]

    gear_ratios_sum = sum(gear_ratios)

    # Return results
    return confirmed_parts_sum, gear_ratios_sum

# %% Day 4: Scratchcards


@time_this_func
def day4():
    # Input parsing
    with open("day4.txt") as f:
        raw = f.read().split("\n")[:-1]

    cards = {}
    for line in raw:
        split_line = line.split(":")
        card_num = int(split_line[0].split()[1])
        numbers = split_line[1].split("|")
        winning_numbers = {int(x) for x in numbers[0].split()}
        my_numbers = {int(x) for x in numbers[1].split()}

        cards[card_num] = [winning_numbers, my_numbers]

    # Part 1
    winners = [len(winning_numbers & my_numbers) for winning_numbers, my_numbers in cards.values()]
    winnings = [2**(num_winners-1) if num_winners != 0 else 0 for num_winners in winners]
    total_winnings = sum(winnings)

    # Part 2
    num_cards = {x: 1 for x in cards}

    for c, to_copy in enumerate(winners):
        card = c + 1
        for copy_card in range(card + 1, card + 1 + to_copy):
            num_cards[copy_card] += num_cards[card]

    total_cards = sum(num_cards.values())

    # Return results
    return total_winnings, total_cards

# %% Day 5: If You Give A Seed A Fertilizer


@time_this_func
def day5():
    # Input parsing
    with open("day5.txt") as f:
        raw = f.read()

    raw_sections = [section.split("\n") for section in raw.strip().split("\n\n")]
    seed_journey = {"seed": [int(x) for x in raw_sections[0][0].split(":")[1].split()]}

    maps = {}
    for raw_section in raw_sections[1:]:
        maps[raw_section[0][:-5]] = [[int(x) for x in y.split()] for y in raw_section[1:]]

    # Part 1
    currently_on = "seed"
    while currently_on != "location":
        current_map_name = [x for x in maps if currently_on + "-to" in x][0]
        current_map = maps[current_map_name]
        next_on = current_map_name.split("-")[-1]

        seed_journey[next_on] = []
        for current in seed_journey[currently_on]:
            for map_entry in current_map:
                if map_entry[1] <= current <= map_entry[1] + map_entry[2]:
                    seed_journey[next_on].append(map_entry[0] + current - map_entry[1])
                    break
            else:
                seed_journey[next_on].append(current)

        currently_on = next_on

    closest_seed_location_1 = min(seed_journey["location"])

    # Part 2
    seed_ranges = []
    for i in range(0, len(seed_journey["seed"]), 2):
        from_seed = seed_journey["seed"][i]
        seed_range = seed_journey["seed"][i+1]
        seed_ranges.append((from_seed, from_seed + seed_range))

    def is_seed(s):
        for seed_range in seed_ranges:
            if seed_range[0] <= s <= seed_range[1]:
                return True
        return False

    def seed_from_location(loc):
        currently_on = "location"
        current = loc
        while currently_on != "seed":
            current_map_name = [x for x in maps if "to-" + currently_on in x][0]
            current_map = maps[current_map_name]
            next_on = current_map_name.split("-")[0]

            for map_entry in current_map:
                if map_entry[0] <= current <= map_entry[0] + map_entry[2]:
                    current = map_entry[1] + current - map_entry[0]
                    break

            currently_on = next_on

        return current

    location = 0
    while not is_seed(seed_from_location(location)):
        location += 1

    closest_seed_location_2 = location
    # Answer is 15290096, 400 s runtime

    # Return results
    return closest_seed_location_1, closest_seed_location_2

# %% Day 6: Wait For It


@time_this_func
def day6():
    # Input parsing
    with open("day6.txt") as f:
        raw = f.read()[:-1].split("\n")

    times = [int(x) for x in raw[0].split()[1:]]
    records = [int(x) for x in raw[1].split()[1:]]

    # Part 1
    def beats_record(hold):
        distance = hold*(race_time-hold)
        return distance > record

    ways_to_win = []
    for race_time, record in zip(times, records):
        wins = 0
        for hold in range(race_time+1):
            is_winner = beats_record(hold)
            wins += is_winner
            if wins != 0 and not is_winner:
                break
        ways_to_win.append(wins)

    wins_product = prod(ways_to_win)

    # Part 2
    race_time = int("".join([str(x) for x in times]))
    record = int("".join([str(x) for x in records]))

    bottom_bound = 0
    step = int("1" + "0"*(len(str(race_time))-1))
    while True:
        bottom_bound += step
        if beats_record(bottom_bound):
            if step == 1:
                break
            bottom_bound -= step
            step //= 10

    top_bound = race_time
    step = int("1" + "0"*(len(str(race_time))-1))
    while True:
        top_bound -= step
        if beats_record(top_bound):
            if step == 1:
                break
            top_bound += step
            step //= 10

    ways_to_win = top_bound - bottom_bound + 1

    # Return results
    return wins_product, ways_to_win

# %% Day 6: Wait For It (Alternate)


@time_this_func
def day6_alternate():
    # Input parsing
    with open("day6.txt") as f:
        raw = f.read()[:-1].split("\n")

    times = [int(x) for x in raw[0].split()[1:]]
    records = [int(x) for x in raw[1].split()[1:]]

    # Part 1
    def ways_to_win(race_time, record):
        a = 1
        b = -race_time
        c = record

        top_bound = (-b + sqrt(b**2 - 4*a*c))/(2*a)
        bottom_bound = (-b - sqrt(b**2 - 4*a*c))/(2*a)
        return int(top_bound) - int(bottom_bound+1) + 1

    part_1 = prod([ways_to_win(race_time, record) for race_time, record in zip(times, records)])

    # Part 2
    race_time = int("".join([str(x) for x in times]))
    record = int("".join([str(x) for x in records]))

    part_2 = ways_to_win(race_time, record)

    # Return results
    return part_1, part_2

# %% Day 7: Camel Cards


@time_this_func
def day7():
    # Input parsing
    with open("day7.txt") as f:
        raw = f.read()[:-1].split("\n")

    split_raw = [x.split() for x in raw]
    hands = [x[0] for x in split_raw]
    bids = [int(x[1]) for x in split_raw]

    # Part 1
    card_strength = {"A": 13, "K": 12, "Q": 11, "J": 10, "T": 9,
                     "9": 8, "8": 7, "7": 6, "6": 5, "5": 4, "4": 3, "3": 2, "2": 1}

    def hand_strength(hand):
        hand_count = [hand.count(x) for x in set(hand)]
        if hand_count == [5]:
            return 7
        if max(hand_count) == 4:
            return 6
        if sorted(hand_count) == [2, 3]:
            return 5
        if 3 in hand_count:
            return 4
        if hand_count.count(2) == 2:
            return 3
        if 2 in hand_count:
            return 2
        else:
            return 1

    for i in range(4, -1, -1):
        hands, bids = zip(
            *sorted(zip(hands, bids), key=lambda x: card_strength[x[0][i]], reverse=True))

    hands, bids = zip(*sorted(zip(hands, bids), key=lambda x: hand_strength(x[0]), reverse=True))

    total_winnings = sum([(i+1)*b for i, b in enumerate(bids[::-1])])

    # Part 2
    card_strength["J"] = 0

    # Old Version (try every hand)
    # def better_hand_than(hand_1, compare_to_hand):
    #     comparison = [hand_1, compare_to_hand]
    #     for i in range(4, -1, -1):
    #         comparison = sorted(comparison, key=lambda x: card_strength[x[i]], reverse=True)

    #     comparison = sorted(comparison, key=lambda x: hand_strength(x), reverse=True)

    #     return comparison[0] == hand_1

    def best_hand(hand):
        # Old Version (try every hand)
        # j_inds = [i for i, c in enumerate(hand) if c == "J"]
        # split_hand = [x for x in hand]

        # j_options = product("AKQT98765432", repeat=len(j_inds))
        # best = hand.replace("J", "2")

        # for j_option in j_options:
        #     for j_sub, j_ind in zip(j_option, j_inds):
        #         split_hand[j_ind] = j_sub
        #     if better_hand_than("".join(split_hand), best):
        #         best = "".join(split_hand)

        # return best

        hand_counts = {c: hand.count(c) for c in set(hand) if c != "J"}
        joker_to_become = sorted(hand_counts, key=lambda x: card_strength[x], reverse=True)
        joker_to_become = sorted(joker_to_become, key=lambda x: hand_counts[x], reverse=True)
        if len(joker_to_become) == 0:
            joker_to_become.append("A")
        return hand.replace("J", joker_to_become[0])

    original_hands = hands
    wilded_hands = [best_hand(hand) for hand in hands]

    for i in range(4, -1, -1):
        original_hands, wilded_hands, bids = zip(
            *sorted(zip(original_hands, wilded_hands, bids), key=lambda x: card_strength[x[0][i]], reverse=True))

    original_hands, wilded_hands, bids = zip(
        *sorted(zip(original_hands, wilded_hands, bids), key=lambda x: hand_strength(x[1]), reverse=True))

    total_winnings_wild = sum([(i+1)*b for i, b in enumerate(bids[::-1])])

    # Return results
    return total_winnings, total_winnings_wild

# %% Day 8: Haunted Wasteland


@time_this_func
def day8():
    # Input parsing
    with open("day8.txt") as f:
        raw = f.read()

    turns, links_raw = raw.split("\n\n")

    links = {}
    for link_line in links_raw[:-1].split("\n"):
        link_from, link_to = link_line.split(" = (")
        link_left, link_right = link_to[:-1].split(", ")
        links[link_from] = {"L": link_left, "R": link_right}

    # Part 1
    def instructions():
        while True:
            for x in turns:
                yield x

    turn = instructions()

    curr = "AAA"
    steps = 0
    while curr != "ZZZ":
        steps += 1
        curr = links[curr][next(turn)]

    AAA_ZZZ_steps = steps

    # Part 2
    currs = [x for x in links if x[-1] == "A"]

    steps = []
    for curr in currs:
        steps.append(0)
        turn = instructions()
        while curr[-1] != "Z":
            steps[-1] += 1
            curr = links[curr][next(turn)]

    def is_prime(num):
        if num <= 1:
            return False
        if num % 2 == 0 and num != 2:
            return False
        if num % 3 == 0 and num != 3:
            return False
        if num % 5 == 0 and num != 5:
            return False
        for i in range(2, int(sqrt(num)) + 1):
            if num % i == 0:
                return False
        else:
            return True

    def get_factor_pair(num):
        for i in range(2, num):
            if num % i == 0:
                return [i, num//i]

    def get_prime_factors(num):
        primes = set()
        not_primes = set()
        prime_factors = [num]
        while False in [x in primes for x in prime_factors]:
            factors = []
            for n in prime_factors:
                if n in not_primes or not is_prime(n):
                    not_primes.add(num)
                    factors += get_factor_pair(n)
                else:
                    primes.add(n)
                    factors += [n]
            prime_factors = factors

        return {x: prime_factors.count(x) for x in set(prime_factors)}

    prime_factors_union = {}
    for step in steps:
        prime_factors = get_prime_factors(step)
        for factor, n in prime_factors.items():
            if factor not in prime_factors_union or prime_factors_union[factor] < n:
                prime_factors_union[factor] = n

    steps_LCM = prod([p**n for p, n in prime_factors_union.items()])

    # Return results
    return AAA_ZZZ_steps, steps_LCM

# %% Day 9: Mirage Maintenance


@time_this_func
def day9():
    # Input parsing
    with open("day9.txt") as f:
        raw = f.read()[:-1]

    raw_sequences = [[int(x) for x in y.split()] for y in raw.split("\n")]

    # Part 1
    def get_progression(sequence):
        progression = [sequence]
        while progression[-1].count(0) != len(progression[-1]):
            progression.append([])
            for i in range(1, len(progression[-2])):
                progression[-1].append(progression[-2][i] - progression[-2][i-1])

        return progression

    progressions = []
    for sequence in raw_sequences:
        progressions.append(get_progression(sequence))
    
    for j in range(len(progressions)):
        progressions[j][-1].append(0)
        for i in range(len(progressions[j])-2, -1, -1):
            progressions[j][i].append(progressions[j][i][-1] + progressions[j][i+1][-1])

    next_total = sum([p[0][-1] for p in progressions])

    # Part 2
    for j in range(len(progressions)):
        progressions[j][-1] = [0] + progressions[j][-1]
        for i in range(len(progressions[j])-2, -1, -1):
            progressions[j][i] = [progressions[j][i][0] -
                                  progressions[j][i+1][0]] + progressions[j][i]

    previous_total = sum([p[0][0] for p in progressions])

    # Return results
    return next_total, previous_total

# %% Day 10: Pipe Maze


@time_this_func
def day10():
    # Input parsing
    with open("day10.txt") as f:
        raw = f.read()[:-1]
        
    pipe_map = np.array([[x for x in y] for y in raw.split("\n")])
    
    # Part 1
    rows = pipe_map.shape[0]
    cols = pipe_map.shape[1]
    
    start = tuple([int(x[0]) for x in np.where(pipe_map == "S")])
    
    def get_first_step(loc):
        if loc[0] != 0 and pipe_map[loc[0]-1, loc[1]] in {"|", "7", "F"}:
            return (loc[0]-1, loc[1])
        if loc[0] < rows and pipe_map[loc[0]+1, loc[1]] in {"|", "J", "L"}:
            return (loc[0]+1, loc[1])
        if loc[1] != 0 and pipe_map[loc[0], loc[1]-1] in {"-", "F", "L"}:
            return (loc[0], loc[1]-1)
        if loc[1] < cols and pipe_map[loc[0], loc[1]+1] in {"-", "7", "J"}:
            return (loc[0], loc[1]+1)
    
    def next_loc(loc, last_loc):
        if pipe_map[loc] == "-":
            return (loc[0], loc[1] + loc[1] - last_loc[1])
        if pipe_map[loc] == "|":
            return (loc[0] + loc[0] - last_loc[0], loc[1])
        if pipe_map[loc] == "L":
            if last_loc[0] == loc[0]:
                return (loc[0]-1, loc[1])
            else:
                return (loc[0], loc[1]+1)
        if pipe_map[loc] == "J":
            if last_loc[0] == loc[0]:
                return (loc[0]-1, loc[1])
            else:
                return (loc[0], loc[1]-1)
        if pipe_map[loc] == "7":
            if last_loc[0] == loc[0]:
                return (loc[0]+1, loc[1])
            else:
                return (loc[0], loc[1]-1)
        if pipe_map[loc] == "F":
            if last_loc[0] == loc[0]:
                return (loc[0]+1, loc[1])
            else:
                return (loc[0], loc[1]+1)
    
    pipe_loop = [start, get_first_step(start)]
    
    while True:
        loc = next_loc(pipe_loop[-1], pipe_loop[-2])
        
        if loc == start:
            break
        
        pipe_loop.append(loc)
        
    farthest_steps = len(pipe_loop)//2
    
    # Part 2
    start_neighbors = {pipe_loop[1], pipe_loop[-1]}
    
    if start_neighbors == {(start[0],start[1]-1), (start[0],start[1]+1)}:
        pipe_map[start] = "-"
    elif start_neighbors == {(start[0]-1,start[1]), (start[0]+1,start[1])}:
        pipe_map[start] = "|"
    elif start_neighbors == {(start[0]-1,start[1]), (start[0],start[1]+1)}:
        pipe_map[start] = "L"
    elif start_neighbors == {(start[0]-1,start[1]), (start[0],start[1]-1)}:
        pipe_map[start] = "J"
    elif start_neighbors == {(start[0]+1,start[1]), (start[0],start[1]-1)}:
        pipe_map[start] = "7"
    elif start_neighbors == {(start[0]+1,start[1]), (start[0],start[1]+1)}:
        pipe_map[start] = "F"
    
    pipe_loop_set = set(pipe_loop)
    
    enclosed = 0
    for i in range(rows):
        inside = False
        for j in range(cols):
            if (i,j) in pipe_loop_set:
                if pipe_map[i,j] in {"|", "L", "J"}:
                    inside = not inside
            
            elif (i,j) not in pipe_loop_set and inside:
                enclosed += 1
          
    # Return results
    return farthest_steps, enclosed
        