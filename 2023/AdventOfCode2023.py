# %%
from time import time
from math import prod, sqrt
import numpy as np
from functools import cache
import heapq
from itertools import permutations, combinations
import z3
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
        if num % 1 != 0:
            return False
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

    if start_neighbors == {(start[0], start[1]-1), (start[0], start[1]+1)}:
        pipe_map[start] = "-"
    elif start_neighbors == {(start[0]-1, start[1]), (start[0]+1, start[1])}:
        pipe_map[start] = "|"
    elif start_neighbors == {(start[0]-1, start[1]), (start[0], start[1]+1)}:
        pipe_map[start] = "L"
    elif start_neighbors == {(start[0]-1, start[1]), (start[0], start[1]-1)}:
        pipe_map[start] = "J"
    elif start_neighbors == {(start[0]+1, start[1]), (start[0], start[1]-1)}:
        pipe_map[start] = "7"
    elif start_neighbors == {(start[0]+1, start[1]), (start[0], start[1]+1)}:
        pipe_map[start] = "F"

    pipe_loop_set = set(pipe_loop)

    enclosed = 0
    for i in range(rows):
        inside = False
        for j in range(cols):
            if (i, j) in pipe_loop_set:
                if pipe_map[i, j] in {"|", "L", "J"}:
                    inside = not inside

            elif (i, j) not in pipe_loop_set and inside:
                enclosed += 1

    # Return results
    return farthest_steps, enclosed

# %% Day 11: Cosmic Expansion


@time_this_func
def day11():
    # Input parsing
    with open("day11.txt") as f:
        raw = f.read()[:-1].split("\n")

    space = np.array([[x for x in row] for row in raw])

    # Part 1
    empty_rows = set()
    for i in range(space.shape[0]-1, -1, -1):
        if "#" not in space[i, :]:
            empty_rows.add(i)

    empty_cols = set()
    for j in range(space.shape[1]-1, -1, -1):
        if "#" not in space[:, j]:
            empty_cols.add(j)

    galaxy_locs = []
    n_rows = space.shape[0]
    n_cols = space.shape[1]
    for i in range(n_rows):
        for j in range(n_cols):
            if space[i, j] == "#":
                galaxy_locs.append((i, j))

    def manhattan_plus(loc1, loc2):
        empties = 0
        for empty_row in empty_rows:
            if min(loc1[0], loc2[0]) < empty_row < max(loc1[0], loc2[0]):
                empties += 1

        for empty_col in empty_cols:
            if min(loc1[1], loc2[1]) < empty_col < max(loc1[1], loc2[1]):
                empties += 1

        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1]), empties

    galaxy_pairs = []
    for i, galaxy1 in enumerate(galaxy_locs[:-1]):
        for galaxy2 in galaxy_locs[i+1:]:
            galaxy_pairs.append((galaxy1, galaxy2))

    measures = [manhattan_plus(galaxy1, galaxy2) for galaxy1, galaxy2 in galaxy_pairs]
    min_dists = [x[0] for x in measures]
    empties = [x[1] for x in measures]

    total_min_distances_1 = sum(min_dists) + sum(empties)

    # Part 2
    total_min_distances_2 = sum(min_dists) + (1000000-1)*sum(empties)

    # Return results
    return total_min_distances_1, total_min_distances_2

# %% Day 12: Hot Springs


@time_this_func
def day12():
    # Input parsing
    with open("day12.txt") as f:
        raw = f.read()[:-1].split("\n")

    rows = [row.split()[0] for row in raw]
    groups = [tuple([int(x) for x in row.split()[1].split(",")]) for row in raw]

    # Part 1
    @cache
    def arrangements(remaining, curr_group, remaining_groups):
        if remaining == "":
            if len(remaining_groups) == 0 and curr_group == 0:
                return 1
            elif len(remaining_groups) == 1 and curr_group == remaining_groups[0]:
                return 1
            else:
                return 0

        count = 0
        if remaining[0] == ".":
            if curr_group != 0 and (len(remaining_groups) == 0 or curr_group != remaining_groups[0]):
                return 0
            elif len(remaining_groups) != 0 and curr_group == remaining_groups[0]:
                count += arrangements(remaining[1:], 0, remaining_groups[1:])
            else:
                count += arrangements(remaining[1:], 0, remaining_groups)

        elif remaining[0] == "#":
            count += arrangements(remaining[1:], curr_group + 1, remaining_groups)

        else:
            count += arrangements("." + remaining[1:], curr_group, remaining_groups)
            count += arrangements("#" + remaining[1:], curr_group, remaining_groups)

        return count

    arrangements1 = sum([arrangements(x, 0, y) for x, y in zip(rows, groups)])

    # Part 2
    rows5 = ["?".join(5*[x]) for x in rows]
    groups5 = [5*x for x in groups]

    arrangements5 = sum([arrangements(x, 0, y) for x, y in zip(rows5, groups5)])

    # Return results
    return arrangements1, arrangements5

# %% Day 13: Point of Incidence


@time_this_func
def day13():
    # Input parsing
    with open("day13.txt") as f:
        raw = f.read()[:-1].split("\n\n")

    patterns = []
    for raw_pattern in raw:
        pattern = [[0 if x == "." else 1 for x in row] for row in raw_pattern.split("\n")]
        patterns.append(np.array(pattern))

    pattern = patterns[0]

    # Part 1
    def check_horizontal(pattern):
        rows = pattern.shape[0]

        for i in range(rows-1):
            top = pattern[:i+1, :]
            bottom = np.flipud(pattern[i+1:, :])

            reflection_rows = min(top.shape[0], bottom.shape[0])

            top = top[-reflection_rows:, :]
            bottom = bottom[-reflection_rows:, :]

            if np.array_equal(top, bottom):
                return True, i+1

        return False, None

    def check_vertical(pattern):
        cols = pattern.shape[1]

        for j in range(cols-1):
            left = pattern[:, :j+1]
            right = np.fliplr(pattern[:, j+1:])

            reflection_cols = min(left.shape[1], right.shape[1])

            left = left[:, -reflection_cols:]
            right = right[:, -reflection_cols:]

            if np.array_equal(left, right):
                return True, j+1

        return False, None

    total = 0
    for pattern in patterns:
        result, rows_above = check_horizontal(pattern)

        if result:
            total += 100*rows_above
        else:
            total += check_vertical(pattern)[1]

    # Part 2
    def check_horizontal_smudge(pattern):
        rows = pattern.shape[0]

        for i in range(rows-1):
            top = pattern[:i+1, :]
            bottom = np.flipud(pattern[i+1:, :])

            reflection_rows = min(top.shape[0], bottom.shape[0])

            top = top[-reflection_rows:, :]
            bottom = bottom[-reflection_rows:, :]

            if np.sum(top != bottom) == 1:
                return True, i+1

        return False, None

    def check_vertical_smudge(pattern):
        cols = pattern.shape[1]

        for j in range(cols-1):
            left = pattern[:, :j+1]
            right = np.fliplr(pattern[:, j+1:])

            reflection_cols = min(left.shape[1], right.shape[1])

            left = left[:, -reflection_cols:]
            right = right[:, -reflection_cols:]

            if np.sum(left != right) == 1:
                return True, j+1

        return False, None

    total_smudge = 0
    for pattern in patterns:
        result, rows_above = check_horizontal_smudge(pattern)

        if result:
            total_smudge += 100*rows_above
        else:
            total_smudge += check_vertical_smudge(pattern)[1]

    # Return results
    return total, total_smudge

# %% Day 14: Parabolic Reflector Dish


@time_this_func
def day14():
    # Parsing input
    with open("day14.txt") as f:
        raw = f.read()[:-1]

    platform = np.array([[x for x in row] for row in raw.split("\n")])

    # Part 1
    def roll_north(platform):
        while True:
            roll_to = platform[:-1, :]
            roll_from = platform[1:, :]
            rollable_mask = (roll_to == ".") * (roll_from == "O")

            if np.sum(rollable_mask) == 0:
                return platform

            false_row = np.array([False]*platform.shape[1])

            platform[np.vstack([false_row, rollable_mask])] = "."
            platform[np.vstack([rollable_mask, false_row])] = "O"

    platform = roll_north(platform)

    total_load = 0
    for i, row in enumerate(platform):
        total_load += (platform.shape[0]-i)*np.sum(row == "O")

    # Part 2
    def roll_west(platform):
        while True:
            roll_to = platform[:, :-1]
            roll_from = platform[:, 1:]
            rollable_mask = (roll_to == ".") * (roll_from == "O")

            if np.sum(rollable_mask) == 0:
                return platform

            false_col = np.array([[False]]*platform.shape[0])

            platform[np.hstack([false_col, rollable_mask])] = "."
            platform[np.hstack([rollable_mask, false_col])] = "O"

    def roll_south(platform):
        while True:
            roll_to = platform[1:, :]
            roll_from = platform[:-1, :]
            rollable_mask = (roll_to == ".") * (roll_from == "O")

            if np.sum(rollable_mask) == 0:
                return platform

            false_row = np.array([False]*platform.shape[1])

            platform[np.vstack([rollable_mask, false_row])] = "."
            platform[np.vstack([false_row, rollable_mask])] = "O"

    def roll_east(platform):
        while True:
            roll_to = platform[:, 1:]
            roll_from = platform[:, :-1]
            rollable_mask = (roll_to == ".") * (roll_from == "O")

            if np.sum(rollable_mask) == 0:
                return platform

            false_col = np.array([[False]]*platform.shape[0])

            platform[np.hstack([rollable_mask, false_col])] = "."
            platform[np.hstack([false_col, rollable_mask])] = "O"

    def spin(platform):
        platform = roll_north(platform)
        platform = roll_west(platform)
        platform = roll_south(platform)
        platform = roll_east(platform)
        return platform

    platform = np.array([[x for x in row] for row in raw.split("\n")])

    def hashify(np_platform):
        tuple_platform = tuple([tuple([x for x in row]) for row in np_platform])
        return hash(tuple_platform)

    spins = 0
    history = [hashify(platform)]
    while True:
        spins += 1
        platform = spin(platform)
        hashified = hashify(platform)
        if hashified in history:
            first_instance = history.index(hashified)
            next_instance = spins
            break

        history.append(hashified)

    remainder_cycles = (1000000000-next_instance) % (next_instance-first_instance)
    for i in range(remainder_cycles):
        platform = spin(platform)

    total_load_cycle = 0
    for i, row in enumerate(platform):
        total_load_cycle += (platform.shape[0]-i)*np.sum(row == "O")

    # Return results
    return total_load, total_load_cycle

# %% Day 15: Lens Library


@time_this_func
def day15():
    # Input parsing
    with open("day15.txt") as f:
        raw = f.read()[:-1]

    steps = raw.split(",")

    # Part 1
    def lava_hash(step):
        curr = 0
        for c in step:
            curr += ord(c)
            curr *= 17
            curr %= 256
        return curr

    lava_hash_sum = sum([lava_hash(step) for step in steps])

    # Part 2
    box_steps = []
    for step in steps:
        if "-" in step:
            new = [step[:-1], step[-1]]
        else:
            new = step.split("=")
            new.insert(1, "=")
            new[2] = int(new[2])
        box_steps.append(new)

    boxes = {i: [] for i in range(256)}

    for step in box_steps:
        box_num = lava_hash(step[0])
        box_inds = {info[0]: i for i, info in enumerate(boxes[box_num])}
        if step[1] == "-" and step[0] in box_inds:
            del boxes[box_num][box_inds[step[0]]]

        elif step[1] == "=":
            if step[0] in box_inds:
                boxes[box_num][box_inds[step[0]]] = step[::2]
            else:
                boxes[box_num].append(step[::2])

    power = 0
    for b, box in boxes.items():
        for s, info in enumerate(box):
            power += (b+1) * (s+1) * info[1]

    # Return results
    return lava_hash_sum, power

# %% Day 16: The Floor Will Be Lava


@time_this_func
def day16():
    # Parsing input
    with open("day16.txt") as f:
        raw = f.read()[:-1].split("\n")

    tiles = np.array([[x for x in row] for row in raw])

    # Part 1
    seen = set()

    def beam(loc, direction):
        while 0 <= loc[0] < tiles.shape[0] and 0 <= loc[1] < tiles.shape[1] and (loc, direction) not in seen:
            seen.add((loc, direction))

            if tiles[loc] == ".":
                if direction == "N":
                    loc = (loc[0] - 1, loc[1])
                elif direction == "E":
                    loc = (loc[0], loc[1] + 1)
                elif direction == "S":
                    loc = (loc[0] + 1, loc[1])
                elif direction == "W":
                    loc = (loc[0], loc[1] - 1)

            elif tiles[loc] == "\\":
                if direction == "N":
                    loc = (loc[0], loc[1] - 1)
                    direction = "W"
                elif direction == "E":
                    loc = (loc[0] + 1, loc[1])
                    direction = "S"
                elif direction == "S":
                    loc = (loc[0], loc[1] + 1)
                    direction = "E"
                elif direction == "W":
                    loc = (loc[0] - 1, loc[1])
                    direction = "N"

            elif tiles[loc] == "/":
                if direction == "N":
                    loc = (loc[0], loc[1] + 1)
                    direction = "E"
                elif direction == "E":
                    loc = (loc[0] - 1, loc[1])
                    direction = "N"
                elif direction == "S":
                    loc = (loc[0], loc[1] - 1)
                    direction = "W"
                elif direction == "W":
                    loc = (loc[0] + 1, loc[1])
                    direction = "S"

            elif tiles[loc] == "|":
                if direction == "N":
                    loc = (loc[0] - 1, loc[1])
                elif direction == "S":
                    loc = (loc[0] + 1, loc[1])
                else:
                    beam((loc[0] - 1, loc[1]), "N")
                    beam((loc[0] + 1, loc[1]), "S")

            elif tiles[loc] == "-":
                if direction == "E":
                    loc = (loc[0], loc[1] + 1)
                elif direction == "W":
                    loc = (loc[0], loc[1] - 1)
                else:
                    beam((loc[0], loc[1] + 1), "E")
                    beam((loc[0], loc[1] - 1), "W")

    beam((0, 0), "E")

    energized = len({x[0] for x in seen})

    # Part 2
    most_energized = 0
    direction = "S"
    for i in [0, tiles.shape[0]-1]:
        for j in range(tiles.shape[1]):
            seen = set()
            beam((i, j), direction)
            energized_option = len({x[0] for x in seen})
            if energized_option > most_energized:
                most_energized = energized_option
        direction = "N"

    direction = "E"
    for j in [0, tiles.shape[1]-1]:
        for i in range(tiles.shape[0]):
            seen = set()
            beam((i, j), direction)
            energized_option = len({x[0] for x in seen})
            if energized_option > most_energized:
                most_energized = energized_option
        direction = "W"

    # Return results
    return energized, most_energized

# %% Day 17: Clumsy Crucible


@time_this_func
def day17():
    # Input parsing
    with open("day17.txt") as f:
        raw = f.read()[:-1].split("\n")

    blocks = np.array([[int(x) for x in row] for row in raw])

    # Part 1
    rows = blocks.shape[0]
    cols = blocks.shape[1]

    # Node structure: (energy loss, location, movement into node, streak of <- movements)
    # Neighbor structure: (location, movement into node, streak of <- movements)

    start = ((0, 0), 'E', 0)
    energy_losses = {start: 0}
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            for n in range(1, min(4, i + 1)):
                energy_losses[((i, j), "S", n)] = np.inf

            for n in range(1, min(4, cols - j)):
                energy_losses[((i, j), "W", n)] = np.inf

            for n in range(1, min(4, rows - i)):
                energy_losses[((i, j), "N", n)] = np.inf

            for n in range(1, min(4, j + 1)):
                energy_losses[((i, j), "E", n)] = np.inf

    def get_neighbors(from_node):
        loc = from_node[0]
        direction = from_node[1]
        count = from_node[2]
        if direction == "N":
            neighbors = [((loc[0]-1, loc[1]), "N", count+1),
                         ((loc[0], loc[1]+1), "E", 1),
                         ((loc[0], loc[1]-1), "W", 1)]
        elif direction == "E":
            neighbors = [((loc[0]-1, loc[1]), "N", 1),
                         ((loc[0], loc[1]+1), "E", count+1),
                         ((loc[0]+1, loc[1]), "S", 1)]
        elif direction == "S":
            neighbors = [((loc[0], loc[1]+1), "E", 1),
                         ((loc[0]+1, loc[1]), "S", count+1),
                         ((loc[0], loc[1]-1), "W", 1)]
        elif direction == "W":
            neighbors = [((loc[0]-1, loc[1]), "N", 1),
                         ((loc[0]+1, loc[1]), "S", 1),
                         ((loc[0], loc[1]-1), "W", count+1)]

        return [n for n in neighbors if 0 <= n[0][0] < rows and 0 <= n[0][1] < cols and n[2] < 4]

    to_visit = [(0, start[0], start[1], start[2])]
    while len(to_visit) != 0:
        current_node = heapq.heappop(to_visit)
        neighbors = get_neighbors(current_node[1:])
        for neighbor in neighbors:
            if energy_losses[current_node[1:]] + blocks[neighbor[0]] < energy_losses[neighbor]:
                energy_losses[neighbor] = energy_losses[current_node[1:]] + blocks[neighbor[0]]
                new_node = (energy_losses[current_node[1:]] +
                            blocks[neighbor[0]], neighbor[0], neighbor[1], neighbor[2])
                heapq.heappush(to_visit, new_node)

    min_energy_loss = min(
        [energy_loss for k, energy_loss in energy_losses.items() if k[0] == (rows-1, cols-1)])

    # Part 2
    starts = [((0, 0), 'E', 0), ((0, 0), 'S', 0)]
    energy_losses = {s: 0 for s in starts}
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            for n in range(1, min(11, i + 1)):
                energy_losses[((i, j), "S", n)] = np.inf

            for n in range(1, min(11, cols - j)):
                energy_losses[((i, j), "W", n)] = np.inf

            for n in range(1, min(11, rows - i)):
                energy_losses[((i, j), "N", n)] = np.inf

            for n in range(1, min(11, j + 1)):
                energy_losses[((i, j), "E", n)] = np.inf

    def get_ultra_neighbors(from_node):
        loc = from_node[0]
        direction = from_node[1]
        count = from_node[2]

        if direction == "N":
            if count < 4:
                neighbors = [((loc[0]-1, loc[1]), "N", count+1)]
            else:
                neighbors = [((loc[0]-1, loc[1]), "N", count+1),
                             ((loc[0], loc[1]+1), "E", 1),
                             ((loc[0], loc[1]-1), "W", 1)]
        elif direction == "E":
            if count < 4:
                neighbors = [((loc[0], loc[1]+1), "E", count+1)]
            else:
                neighbors = [((loc[0]-1, loc[1]), "N", 1),
                             ((loc[0], loc[1]+1), "E", count+1),
                             ((loc[0]+1, loc[1]), "S", 1)]
        elif direction == "S":
            if count < 4:
                neighbors = [((loc[0]+1, loc[1]), "S", count+1)]
            else:
                neighbors = [((loc[0], loc[1]+1), "E", 1),
                             ((loc[0]+1, loc[1]), "S", count+1),
                             ((loc[0], loc[1]-1), "W", 1)]
        elif direction == "W":
            if count < 4:
                neighbors = [((loc[0], loc[1]-1), "W", count+1)]
            else:
                neighbors = [((loc[0]-1, loc[1]), "N", 1),
                             ((loc[0]+1, loc[1]), "S", 1),
                             ((loc[0], loc[1]-1), "W", count+1)]

        return [n for n in neighbors if 0 <= n[0][0] < rows and 0 <= n[0][1] < cols and n[2] < 11]

    to_visit = []
    for s in starts:
        heapq.heappush(to_visit, (0, s[0], s[1], s[2]))

    while len(to_visit) != 0:
        current_node = heapq.heappop(to_visit)
        neighbors = get_ultra_neighbors(current_node[1:])
        for neighbor in neighbors:
            if energy_losses[current_node[1:]] + blocks[neighbor[0]] < energy_losses[neighbor]:
                energy_losses[neighbor] = energy_losses[current_node[1:]] + blocks[neighbor[0]]
                new_node = (energy_losses[current_node[1:]] +
                            blocks[neighbor[0]], neighbor[0], neighbor[1], neighbor[2])
                heapq.heappush(to_visit, new_node)

    min_ultra_energy_loss = min(
        [energy_loss for k, energy_loss in energy_losses.items() if k[0] == (rows-1, cols-1) and k[2] >= 4])

    # Return results
    return min_energy_loss, min_ultra_energy_loss

# %% Day 18: Lavaduct Lagoon


@time_this_func
def day18():
    # Input parsing
    with open("day18.txt") as f:
        raw = f.read()[:-1].split("\n")

    digs = []
    for row in raw:
        direction, n, color = row.split()
        digs.append((direction, int(n), color[1:-1]))

    # Part 1
    vertices = []
    loc = (0, 0)
    boundary_points = 0
    for direction, count, _ in digs:
        if direction == "U":
            loc = (loc[0] - count, loc[1])
        elif direction == "R":
            loc = (loc[0], loc[1] + count)
        elif direction == "D":
            loc = (loc[0] + count, loc[1])
        elif direction == "L":
            loc = (loc[0], loc[1] - count)

        boundary_points += count
        vertices.append(loc)

    def shoelace_theorem(vertices):
        rs = [x[0] for x in vertices]
        cs = [x[1] for x in vertices]

        part1 = sum([r*c for r, c in zip(rs, cs[1:]+[cs[0]])])
        part2 = sum([r*c for r, c in zip(rs[1:]+[rs[0]], cs)])

        area = abs(part1-part2)/2
        return area

    def picks_theorem(area, boundary_points):
        return area - (boundary_points/2) + 1

    area = shoelace_theorem(vertices)
    interior_points = picks_theorem(area, boundary_points)
    dig_area = int(boundary_points + interior_points)

    # A fun, but outdated "fill" methodology. Too slow for Part 2.

    # delta = {"U": [-1, 0], "R": [0, 1], "D": [1, 0], "L": [0, -1]}

    # loc = (0, 0)
    # trench = {loc}
    # for direction, count, _ in digs:
    #     d = delta[direction]
    #     for _ in range(count):
    #         loc = (loc[0] + d[0], loc[1] + d[1])
    #         trench.add(loc)

    # rows = {x[0] for x in trench}
    # min_row = min(rows)
    # max_row = max(rows)

    # cols = {x[1] for x in trench}
    # min_col = min(cols)
    # max_col = max(cols)

    # hole = set()
    # outside = {(min_row - 1, c) for c in range(min_col-1, max_col + 2)} | \
    #           {(max_row + 1, c) for c in range(min_col-1, max_col + 2)} | \
    #           {(r, min_col - 1) for r in range(min_row-1, max_row + 2)} | \
    #           {(r, max_col + 1) for r in range(min_row-1, max_row + 2)}

    # for i in range(min_row, max_row + 1):
    #     for j in range(min_col, max_col + 1):
    #         loc = (i, j)
    #         if loc in trench or loc in hole or loc in outside:
    #             continue

    #         paths = [[loc]]
    #         new_paths = "initializing"
    #         visited = {loc}
    #         is_outside = False
    #         while len(new_paths) > 0:
    #             new_paths = []
    #             for path in paths:
    #                 latest = path[-1]
    #                 for d in delta.values():
    #                     next_loc = (latest[0] + d[0], latest[1] + d[1])
    #                     if next_loc in trench or next_loc in visited:
    #                         continue

    #                     elif next_loc in outside:
    #                         is_outside = True
    #                         break

    #                     visited.add(next_loc)
    #                     new_paths.append(path + [next_loc])

    #                 if is_outside:
    #                     break

    #             if is_outside:
    #                 break

    #             paths = new_paths

    #         else:
    #             hole |= visited

    #         if is_outside:
    #             outside |= visited

    # digout = trench | hole

    # dig_size = len(digout)

    # Part 2
    hex_to_direction = {"3": "U", "0": "R", "1": "D", "2": "L"}

    vertices = []
    loc = (0, 0)
    boundary_points = 0
    for _, _, hex_info in digs:
        count = int(hex_info[1:-1], 16)
        direction = hex_to_direction[hex_info[-1]]
        if direction == "U":
            loc = (loc[0] - count, loc[1])
        elif direction == "R":
            loc = (loc[0], loc[1] + count)
        elif direction == "D":
            loc = (loc[0] + count, loc[1])
        elif direction == "L":
            loc = (loc[0], loc[1] - count)

        boundary_points += count
        vertices.append(loc)

    area = shoelace_theorem(vertices)
    interior_points = picks_theorem(area, boundary_points)
    dig_area_big = int(boundary_points + interior_points)

    # Return results
    return dig_area, dig_area_big

# %% Day 19: Aplenty


@time_this_func
def day19():
    # Input parsing
    with open("day19.txt") as f:
        raw_workflows, raw_parts = f.read()[:-1].split("\n\n")

    workflows = {}
    for w in raw_workflows.split("\n"):
        i = w.index("{")
        name = w[:i]
        details = w[i+1:-1].split(",")

        steps = []
        for d in details[:-1]:
            i = d.index(":")
            steps.append((d[0], d[1], d[2:i], d[i+1:]))
        steps.append(details[-1])

        workflows[name] = steps

    parts = [eval(raw_part.replace("=", ":"),
                  {"x": "x", "m": "m", "a": "a", "s": "s"}) for raw_part in raw_parts.split("\n")]

    # Part 1
    def check_workflow(part, workflow_name):
        workflow = workflows[workflow_name]
        for check in workflow[:-1]:
            if eval("x" + check[1] + check[2], {"x": part[check[0]]}):
                return check[3]
        return workflow[-1]

    def check_part(part):
        result = check_workflow(part, "in")
        while result not in {"A", "R"}:
            result = check_workflow(part, result)
        return result

    accepted = []
    for part in parts:
        if check_part(part) == "A":
            accepted.append(part)

    accepted_sum = sum([sum(x.values()) for x in accepted])

    # Part 2
    def check_workflow_with_range(part_range, workflow_name):
        workflow = workflows[workflow_name]
        decided_ranges = []
        undecided_range = part_range
        for stat, check, num, to in workflow[:-1]:
            num = int(num)
            if check == "<":
                if undecided_range[stat][1] < num:
                    decided_ranges.append((undecided_range, to))
                    return decided_ranges
                elif undecided_range[stat][0] >= num:
                    continue
                else:
                    decided = undecided_range.copy()
                    decided[stat] = (decided[stat][0], num - 1)
                    decided_ranges.append((decided, to))

                    undecided_range[stat] = (num, undecided_range[stat][1])

            elif check == ">":
                if undecided_range[stat][0] > num:
                    decided_ranges.append((undecided_range, to))
                    return decided_ranges
                elif undecided_range[stat][1] <= num:
                    continue
                else:
                    decided = undecided_range.copy()
                    decided[stat] = (num + 1, decided[stat][1])
                    decided_ranges.append((decided, to))

                    undecided_range[stat] = (undecided_range[stat][0], num)

        decided_ranges.append((undecided_range, workflow[-1]))
        return decided_ranges

    def split_and_decide(decided):
        acceptable = 0
        if decided[1] == "A":
            acceptable += prod([x[1] - x[0] + 1 for x in decided[0].values()])
        elif decided[1] == "R":
            pass
        else:
            new_decideds = check_workflow_with_range(decided[0], decided[1])
            for new_decided in new_decideds:
                acceptable += split_and_decide(new_decided)

        return acceptable

    decided_start = ({"x": (1, 4000), "m": (1, 4000), "a": (1, 4000), "s": (1, 4000)}, "in")
    acceptable = split_and_decide(decided_start)

    # Return results
    return accepted_sum, acceptable

# %% Day 20: Pulse Propagation


@time_this_func
def day20():
    # Input parsing
    with open("day20.txt") as f:
        raw = f.read()[:-1].split("\n")

    # Part 1
    nodes = {}
    for row in raw:
        identity, links_to = row.split(" -> ")
        if identity == "broadcaster":
            nodes[identity] = {"type": "broadcaster", "to": links_to.split(", ")}
        else:
            node_type = identity[0]
            node_name = identity[1:]
            links_to = links_to.split(", ")

            if node_type == "%":
                nodes[node_name] = {"type": node_type, "to": links_to, "state": "off"}
            elif node_type == "&":
                nodes[node_name] = {"type": node_type, "to": links_to, "memory": {}}

    for node in nodes:
        for link_to in nodes[node]["to"]:
            if link_to in nodes and nodes[link_to]["type"] == "&":
                nodes[link_to]["memory"][node] = "low"

    pulse_counts = {"low": 0, "high": 0}

    for _ in range(1000):
        pulse_log = [("low", "broadcaster", "button")]
        pulse_counts["low"] += 1

        while len(pulse_log) > 0:
            type_sent, sent_to, sent_from = pulse_log.pop(0)

            if sent_to not in nodes:
                continue

            elif nodes[sent_to]["type"] == "broadcaster":
                for to in nodes[sent_to]["to"]:
                    pulse_log.append((type_sent, to, sent_to))
                    pulse_counts[type_sent] += 1

            elif nodes[sent_to]["type"] == "%" and type_sent == "low":
                if nodes[sent_to]["state"] == "off":
                    nodes[sent_to]["state"] = "on"
                    for to in nodes[sent_to]["to"]:
                        pulse_log.append(("high", to, sent_to))
                        pulse_counts["high"] += 1
                else:
                    nodes[sent_to]["state"] = "off"
                    for to in nodes[sent_to]["to"]:
                        pulse_log.append(("low", to, sent_to))
                        pulse_counts["low"] += 1

            elif nodes[sent_to]["type"] == "&":
                nodes[sent_to]["memory"][sent_from] = type_sent
                if all([x == "high" for x in nodes[sent_to]["memory"].values()]):
                    for to in nodes[sent_to]["to"]:
                        pulse_log.append(("low", to, sent_to))
                        pulse_counts["low"] += 1
                else:
                    for to in nodes[sent_to]["to"]:
                        pulse_log.append(("high", to, sent_to))
                        pulse_counts["high"] += 1

    pulse_counts_prod = prod(pulse_counts.values())

    # Part 2
    nodes = {}
    for row in raw:
        identity, links_to = row.split(" -> ")
        if identity == "broadcaster":
            nodes[identity] = {"type": "broadcaster", "to": links_to.split(", ")}
        else:
            node_type = identity[0]
            node_name = identity[1:]
            links_to = links_to.split(", ")

            if node_type == "%":
                nodes[node_name] = {"type": node_type, "to": links_to, "state": "off"}
            elif node_type == "&":
                nodes[node_name] = {"type": node_type, "to": links_to, "memory": {}}

    for node in nodes:
        for link_to in nodes[node]["to"]:
            if link_to in nodes and nodes[link_to]["type"] == "&":
                nodes[link_to]["memory"][node] = "low"

    # Manually determined: rx receives a low pulse if mg remembers high pulses from all jg, rh, jm, hf
    def rx_low(deciders={"jg", "rh", "jm", "hf"}):
        high_pulse_from_deciders = {}
        button_count = 0
        while True:
            button_count += 1
            pulse_log = [("low", "broadcaster", "button")]

            while len(pulse_log) > 0:
                type_sent, sent_to, sent_from = pulse_log.pop(0)

                if sent_to not in nodes:  # This is rx
                    continue

                elif nodes[sent_to]["type"] == "broadcaster":
                    for to in nodes[sent_to]["to"]:
                        pulse_log.append((type_sent, to, sent_to))

                elif nodes[sent_to]["type"] == "%" and type_sent == "low":
                    if nodes[sent_to]["state"] == "off":
                        nodes[sent_to]["state"] = "on"
                        for to in nodes[sent_to]["to"]:
                            pulse_log.append(("high", to, sent_to))
                    else:
                        nodes[sent_to]["state"] = "off"
                        for to in nodes[sent_to]["to"]:
                            pulse_log.append(("low", to, sent_to))

                elif nodes[sent_to]["type"] == "&":
                    nodes[sent_to]["memory"][sent_from] = type_sent
                    if all([x == "high" for x in nodes[sent_to]["memory"].values()]):
                        for to in nodes[sent_to]["to"]:
                            pulse_log.append(("low", to, sent_to))
                    else:
                        if sent_to in deciders and sent_to not in high_pulse_from_deciders:
                            high_pulse_from_deciders[sent_to] = button_count
                            if len(high_pulse_from_deciders) == len(deciders):
                                return high_pulse_from_deciders
                        for to in nodes[sent_to]["to"]:
                            pulse_log.append(("high", to, sent_to))

    cycles = rx_low()

    def is_prime(num):
        if num % 1 != 0:
            return False
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

    def get_lcm(numbers):
        prime_factors_union = {}
        for num in numbers.values():
            prime_factors = get_prime_factors(num)
            for factor, n in prime_factors.items():
                if factor not in prime_factors_union or prime_factors_union[factor] < n:
                    prime_factors_union[factor] = n

        return prod([p**n for p, n in prime_factors_union.items()])

    first_rx_low = get_lcm(cycles)

    # Return results
    return pulse_counts_prod, first_rx_low

# %% Day 21: Step Counter


@time_this_func
def day21():
    # Input parsing
    with open("day21.txt") as f:
        raw = f.read()[:-1].split("\n")

    plots = np.array([[x for x in row] for row in raw])

    # Part 1
    rows = plots.shape[0]
    cols = plots.shape[1]

    def get_neighbors1(loc):
        neighbors = []
        if loc[0] != 0:
            neighbors.append((loc[0] - 1, loc[1]))
        if loc[1] != cols - 1:
            neighbors.append((loc[0], loc[1] + 1))
        if loc[0] != rows - 1:
            neighbors.append((loc[0] + 1, loc[1]))
        if loc[1] != 0:
            neighbors.append((loc[0], loc[1] - 1))
        return neighbors

    start_loc = tuple([int(x[0]) for x in np.where(plots == "S")])

    locs = {start_loc}
    steps = 0
    while steps != 64:
        new_locs = set()
        for loc in locs:
            for neighbor in get_neighbors1(loc):
                if plots[neighbor] != "#":
                    new_locs.add(neighbor)
        steps += 1
        locs = new_locs

    reach_in_64 = len(locs)

    # Part 2
    def get_neighbors2(loc):
        neighbors = [(loc[0] - 1, loc[1]),
                     (loc[0], loc[1] + 1),
                     (loc[0] + 1, loc[1]),
                     (loc[0], loc[1] - 1)]
        return neighbors

    locs = {start_loc}
    steps = 0
    step_options = {}
    # steps = 65+(k*131). Manually determined that quadratic applies to every k steps.
    while steps != 65+(2*131):
        new_locs = set()
        for loc in locs:
            for neighbor in get_neighbors2(loc):
                if plots[(neighbor[0] % rows, neighbor[1] % cols)] != "#":
                    new_locs.add(neighbor)
        steps += 1
        locs = new_locs

        step_options[steps] = len(locs)

    def gaussian_elimination(matrix):
        matrix = matrix.astype(np.float64)
        if matrix.shape[1] != matrix.shape[0] + 1:
            raise Exception("This function implementation requires an n-by-n+1 matrix")

        eliminatable = False
        for order in permutations(range(matrix.shape[0]), matrix.shape[0]):
            for i_must_not_be_0, row_ind in enumerate(order):
                if matrix[row_ind, i_must_not_be_0] == 0:
                    break
            else:
                eliminatable = True
                break

        if not eliminatable:
            raise Exception("Not all coefficients are represented in this matrix")
        else:
            matrix = np.vstack([matrix[i, :] for i in order])

        for i in range(matrix.shape[0]):
            if matrix[i, i] == 0:
                raise Exception("Matrix includes rows that are linear combinations of each other")
            matrix[i, :] /= matrix[i, i]
            for j in range(matrix.shape[0]):
                if j == i:
                    continue
                matrix[j, :] = matrix[j, :] - (matrix[j, i] * matrix[i, :])
        return matrix

    matrix = np.array([[k**2, k**1, k**0, step_options[65+(131*k)]] for k in range(3)])

    eliminated = gaussian_elimination(matrix)
    a = eliminated[0, -1]
    b = eliminated[1, -1]
    c = eliminated[2, -1]

    k = (26501365 - 65)/131
    reach_in_big = int(a*(k**2) + b*k + c)

    # Return results
    return reach_in_64, reach_in_big

# %% Day 22: Sand Slabs


@time_this_func
def day22():
    # Input parsing
    with open("day22.txt") as f:
        raw = f.read()[:-1].split("\n")

    bricks = []
    for row in raw:
        end1, end2 = row.split("~")
        end1 = [int(x) for x in end1.split(",")]
        end2 = [int(x) for x in end2.split(",")]
        axes_equal = [end1[i] == end2[i] for i in range(3)]
        if False not in axes_equal:
            brick = [tuple(end1)]
        else:
            long_axis = axes_equal.index(False)
            brick = []
            if long_axis == 0:
                for x in range(end1[0], end2[0] + 1):
                    brick.append((x, end1[1], end1[2]))
            elif long_axis == 1:
                for y in range(end1[1], end2[1] + 1):
                    brick.append((end1[0], y, end1[2]))
            else:
                for z in range(end1[2], end2[2] + 1):
                    brick.append((end1[0], end1[1], z))
        bricks.append(brick)

    # Part 1
    all_block_locs = set(sum(bricks, []))

    while True:
        new_bricks = []
        for brick in bricks:
            for block in brick:
                all_block_locs.remove(block)
                
            while True:
                drop_brick = [(block[0], block[1], block[2]-1) for block in brick]

                can_fall = all([block not in all_block_locs and block[2]
                               > 0 for block in drop_brick])
                if can_fall:
                    brick = drop_brick
                else:
                    break
                
            for block in brick:
                all_block_locs.add(block)            

            new_bricks.append(brick)

        if bricks == new_bricks:
            break

        bricks = new_bricks

    supports = {tuple(brick): set() for brick in bricks}
    for brick in supports:
        up_brick = [(block[0], block[1], block[2]+1) for block in brick]
        for possible_on_top in supports:
            if possible_on_top == brick:
                continue
            if any([block in possible_on_top for block in up_brick]):
                supports[brick].add(possible_on_top)

    def num_supported_by(brick):
        return sum([brick in supporteds for supporteds in supports.values()])

    can_remove = 0
    for support_brick, supported_bricks in supports.items():
        if len(supported_bricks) == 0:
            can_remove += 1
        else:
            if 1 not in {num_supported_by(supported) for supported in supported_bricks}:
                can_remove += 1

    # Part 2
    supported_by = {tuple(brick): set() for brick in bricks}
    for brick in supported_by:
        down_brick = [(block[0], block[1], block[2]-1) for block in brick]
        for possible_supporter in supported_by:
            if possible_supporter == brick:
                continue
            if any([block in possible_supporter for block in down_brick]):
                supported_by[brick].add(possible_supporter)

    total_falls = 0
    for brick in supported_by:
        fallen = {brick}
        while True:
            another_falls = False
            for possible_fall_brick in supported_by:
                if supported_by[possible_fall_brick].issubset(fallen) and \
                   len(supported_by[possible_fall_brick]) != 0 and \
                   possible_fall_brick not in fallen:
                    another_falls = True
                    fallen.add(possible_fall_brick)

            if not another_falls:
                total_falls += len(fallen) - 1
                break

    # Return results
    return can_remove, total_falls

# %% Day 23: A Long Walk


@time_this_func
def day23():
    with open("day23.txt") as f:
        raw = f.read()[:-1].split("\n")
        
    trail_map = np.array([[x for x in row] for row in raw])
    
    # Part 1
    rows = trail_map.shape[0]
    cols = trail_map.shape[1]
    
    start = (0, np.where(trail_map[0,:] == ".")[0][0])
    finish = (rows-1, np.where(trail_map[rows-1,:] == ".")[0][0])
    
    def get_neighbors1(loc):
        if trail_map[loc] == "^":
            return [(loc[0] - 1, loc[1])]
        elif trail_map[loc] == ">":
            return [(loc[0], loc[1] + 1)]
        elif trail_map[loc] == "v":
            return [(loc[0] + 1, loc[1])]
        elif trail_map[loc] == "<":
            return [(loc[0], loc[1] - 1)]
        
        neighbors = []
        if loc[0] != 0:
            neighbors.append((loc[0] - 1, loc[1]))
        if loc[1] != cols - 1:
            neighbors.append((loc[0], loc[1] + 1))
        if loc[0] != rows - 1:
            neighbors.append((loc[0] + 1, loc[1]))
        if loc[1] != 0:
            neighbors.append((loc[0], loc[1] - 1))
        return neighbors
    
    complete_trails = []
    
    paths = [[(start)]]
    set_paths = [{start}]
    while True:
        new_paths = []
        new_set_paths = []
        for path, set_path in zip(paths, set_paths):
            latest = path[-1]
            if latest == finish:
                complete_trails.append(path)
                continue
            
            for neighbor in get_neighbors1(latest):
                if trail_map[neighbor] != "#" and neighbor not in set_path:
                    new_paths.append(path + [neighbor])
                    new_set_paths.append(set_path | {neighbor})
                    
        if len(new_paths) == 0:
            break
        
        paths = new_paths
        set_paths = new_set_paths
    
    longest_hike = max([len(trail) for trail in complete_trails]) - 1
    
    # Part 2
    def get_neighbors2(loc):
        neighbors = []
        if loc[0] != 0:
            neighbors.append((loc[0] - 1, loc[1]))
        if loc[1] != cols - 1:
            neighbors.append((loc[0], loc[1] + 1))
        if loc[0] != rows - 1:
            neighbors.append((loc[0] + 1, loc[1]))
        if loc[1] != 0:
            neighbors.append((loc[0], loc[1] - 1))
        return neighbors
    
    def is_node(loc):
        if trail_map[loc] == "#":
            return False
        
        neighbors = get_neighbors2(loc)
        
        return sum([trail_map[l] != "#" for l in neighbors]) > 2
    
    nodes = set()
    for i in range(rows):
        for j in range(cols):
            if is_node((i,j)):
                nodes.add((i,j))
    
    nodes |= {start, finish}
    
    def from_node(node):
        node_to_node = {}
        
        paths = [[(node)]]
        set_paths = [{node}]
        while True:
            new_paths = []
            new_set_paths = []
            for path, set_path in zip(paths, set_paths):
                latest = path[-1]
                if latest in nodes and latest != node:
                    node_to_node[latest] = len(path) - 1
                    continue
                
                for neighbor in get_neighbors2(latest):
                    if trail_map[neighbor] != "#" and neighbor not in set_path:
                        new_paths.append(path + [neighbor])
                        new_set_paths.append(set_path | {neighbor})
                        
            if len(new_paths) == 0:
                break
            
            paths = new_paths
            set_paths = new_set_paths
    
        return node_to_node
    
    nodes_to_nodes = {}
    for node in nodes:
        nodes_to_nodes[node] = from_node(node)
    
    farthest = [0]
    def dfs(path, path_set, dist):
        latest = path[-1]
        if latest == finish:
            farthest[0] = max(farthest[0], dist)
        
        for neighbor, neighbor_dist in nodes_to_nodes[latest].items():
            if neighbor not in path_set:
                dfs(path + [neighbor], path_set | {neighbor}, dist + neighbor_dist)
                
    dfs([start], {start}, 0)
    longest_hike_no_slips = farthest[0]
    
    # Return results
    return longest_hike, longest_hike_no_slips

# %% Day 24: Never Tell Me The Odds


@time_this_func
def day24():
    # Input parsing
    with open("day24.txt") as f:
        raw = f.read()[:-1].split("\n")
        
    hails = []
    for row in raw:
        pos, vel = row.split(" @ ")
        p = [int(x) for x in pos.split(",")]
        v = [int(x) for x in vel.split(",")]
        hails.append((tuple(p), tuple(v)))
        
    test_area = tuple([(200000000000000, 400000000000000)]*2)
    
    # Part 1
    def get_slope_intercept(pos, vel):
        y0 = pos[1] - (vel[1]*pos[0])/vel[0]
        return vel[1]/vel[0], y0
    
    def get_intercept(mb1, mb2):
        m1, b1 = mb1
        m2, b2 = mb2
        
        if m1 != m2:
            x_int = (b2 - b1)/(m1-m2)
            y_int = (m1*x_int) + b1
        else:
            if b1 == b2:
                print("UH OH. Speed and direction dependent. Hope this isn't a case..")
            else:
                x_int = np.inf
                y_int = np.inf
        
        return x_int, y_int
    
    def in_front(pos, vel, target):
        return np.sign(target[0]-pos[0]) == np.sign(vel[0]) and np.sign(target[1]-pos[1]) == np.sign(vel[1])
    
    contacts = 0
    for i, (pos1, vel1) in enumerate(hails):
        for pos2, vel2 in hails[i+1:]:
            mb1 = get_slope_intercept(pos1, vel1)
            mb2 = get_slope_intercept(pos2, vel2)
            
            x_int, y_int = get_intercept(mb1, mb2)
            
            if not (test_area[0][0] <= x_int <= test_area[0][1] and test_area[1][0] <= y_int <= test_area[1][1]):
                continue
            
            if in_front(pos1, vel1, (x_int, y_int)) and in_front(pos2, vel2, (x_int, y_int)):
                contacts += 1
                
    # Part 2
    model = z3.Solver()
    
    (ph1x, ph1y, ph1z), (vh1x, vh1y, vh1z) = hails[0]
    (ph2x, ph2y, ph2z), (vh2x, vh2y, vh2z) = hails[1]
    (ph3x, ph3y, ph3z), (vh3x, vh3y, vh3z) = hails[2]
    
    t1 = z3.Int("t1")
    t2 = z3.Int("t2")
    t3 = z3.Int("t3")
    
    prx = z3.Int("prx")
    pry = z3.Int("pry")
    prz = z3.Int("prz")
    vrx = z3.Int("vrx")
    vry = z3.Int("vry")
    vrz = z3.Int("vrz")
    
    eqs = [prx + (vrx * t1) == ph1x + (vh1x * t1),
           pry + (vry * t1) == ph1y + (vh1y * t1),
           prz + (vrz * t1) == ph1z + (vh1z * t1),
           prx + (vrx * t2) == ph2x + (vh2x * t2),
           pry + (vry * t2) == ph2y + (vh2y * t2),
           prz + (vrz * t2) == ph2z + (vh2z * t2),
           prx + (vrx * t3) == ph3x + (vh3x * t3),
           pry + (vry * t3) == ph3y + (vh3y * t3),
           prz + (vrz * t3) == ph3z + (vh3z * t3)]
    
    model.add(*eqs)
    model.check()
    
    rock_pos0_sum = model.model().eval(prx+pry+prz)
    
    # Return results
    return contacts, rock_pos0_sum

# %% Day 25: Snowverload


@time_this_func
def day25():
    # Input parsing
    with open("day25.txt") as f:
        raw = f.read()[:-1].split("\n")
    
    # Part 1
    links = {}
    all_nodes = set()
    for row in raw:
        origin, destinations = row.split(": ")
        all_nodes.add(origin)
        if origin not in links:
            links[origin] = set()
        for destination in destinations.split():
            all_nodes.add(destination)
            links[origin].add(destination)
            if destination not in links:
                links[destination] = {origin}
            else:
                links[destination].add(origin)
                
    main_group = all_nodes.copy()
    other_group = {main_group.pop()}   
    
    external_relations = {k:len(v & other_group) for k,v in links.items() if k in main_group}   
    while sum(external_relations.values()) != 3:
        to_move = max(external_relations, key = lambda x: external_relations[x])
        main_group.remove(to_move)
        other_group.add(to_move)
        external_relations = {k:len(v & other_group) for k,v in links.items() if k in main_group}
        
    group_sizes_prod = len(main_group)*len(other_group)
    
    # A cool brute force method that is way to slow for the actual problem, but was fun to code:
    def brute_force():
        from copy import deepcopy
        
        links = {}
        all_nodes = set()
        for row in raw:
            origin, destinations = row.split(": ")
            all_nodes.add(origin)
            if origin not in links:
                links[origin] = set()
            for destination in destinations.split():
                all_nodes.add(destination)
                links[origin].add(destination)
                if destination not in links:
                    links[destination] = {origin}
                else:
                    links[destination].add(origin)
        
        connections = set()
        for row in raw:
            origin, destinations = row.split(": ")
            for destination in destinations.split():
                connections.add(tuple(sorted((origin, destination))))
        connections = list(connections)
                 
        def explore(curr, explore_links):
            visited = {curr}
            on = [curr]
            while len(on) > 0:
                curr = on.pop(0)
                for to in explore_links[curr]:
                    if to in visited:
                        continue
                    else:
                        visited.add(to)
                        on.append(to)
                
            return visited
        
        def get_groups(explore_links):
            ungrouped = all_nodes.copy()
            groups = []
            while len(ungrouped) > 0:
                groups.append(explore(ungrouped.pop(), explore_links))
                ungrouped -= groups[-1]
            return groups
        
        all_possible_cuts = combinations(range(len(connections)), 3)
        for possible_cuts in all_possible_cuts:
            possible_links = deepcopy(links)
            for link in [connections[i] for i in possible_cuts]:
                possible_links[link[0]].remove(link[1])
                possible_links[link[1]].remove(link[0])
            
            new_groups = get_groups(possible_links)
            if len(new_groups) == 2:
                break
            
        return prod([len(x) for x in new_groups])
    
    return group_sizes_prod
