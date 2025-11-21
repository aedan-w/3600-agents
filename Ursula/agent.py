from collections.abc import Callable, Set
from time import sleep
from collections import deque
from typing import List, Set, Tuple

import numpy as np
from game import board
from game.enums import Direction, MoveType

"""
    ye we finna make it fr - aedan
"""

class PlayerAgent:
    """
    You may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        """
        Initialize the agent.
        """
        self.board_size = board.game_map.MAP_SIZE
        
        # self.white_trapdoor_probs: Probability of trapdoor on an EVEN square (i+j is even)
        self.white_trapdoor_probs = np.zeros((self.board_size, self.board_size))
        
        # self.black_trapdoor_probs: Probability of trapdoor on an ODD square (i+j is odd)
        self.black_trapdoor_probs = np.zeros((self.board_size, self.board_size))
        
        # Track previous locations to deduce when a trapdoor is hit
        self.my_last_loc = None
        self.opponent_last_loc = None

        # --- NEW: Add memory for exploration ---
        # Track which squares we have already visited
        self.visited_squares = np.zeros((self.board_size, self.board_size))
        # --- End of New ---

        self._initialize_probabilities()



    def _initialize_probabilities(self):
        """
        Sets the initial weighted probabilities for trapdoor locations
        based on the rules in trapdoor_manager.py and assignment.pdf.
        """
        weights = np.zeros((self.board_size, self.board_size))
        
        # Weight 0: Rows/Cols 0, 1, 6, 7
        # Weight 1: Rows/Cols 2, 5 (indices 2:6)
        weights[2:6, 2:6] = 1.0
        # Weight 2: Rows/Cols 3, 4 (indices 3:5)
        weights[3:5, 3:5] = 2.0

        total_white_weight = 0
        total_black_weight = 0

        # Assign weights to the correct parity map
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r + c) % 2 == 0:  # White square
                    self.white_trapdoor_probs[r, c] = weights[r, c]
                    total_white_weight += weights[r, c]
                else:  # Black square
                    self.black_trapdoor_probs[r, c] = weights[r, c]
                    total_black_weight += weights[r, c]

        # Normalize to create a probability distribution
        if total_white_weight > 0:
            self.white_trapdoor_probs /= total_white_weight
            
        if total_black_weight > 0:
            self.black_trapdoor_probs /= total_black_weight

    def _get_hear_prob(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> float:
        """
        Calculates the probability of HEARING a trapdoor at loc2 from loc1.
        Logic is copied from game_map.py.
        """
        delta_x = abs(loc1[0] - loc2[0])
        delta_y = abs(loc1[1] - loc2[1])

        if delta_x > 2 or delta_y > 2:
            return 0.0
        if delta_x == 2 and delta_y == 2:
            return 0.0
        if delta_x == 2 or delta_y == 2:
            return 0.1
        if delta_x == 1 and delta_y == 1:
            return 0.25
        if delta_x == 1 or delta_y == 1: # This covers (1,0) and (0,1)
            return 0.5
        return 0.0 # This covers (0,0)

    def _get_feel_prob(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> float:
        """
        Calculates the probability of FEELING a trapdoor at loc2 from loc1.
        Logic is copied from game_map.py.
        """
        delta_x = abs(loc1[0] - loc2[0])
        delta_y = abs(loc1[1] - loc2[1])

        if delta_x > 1 or delta_y > 1:
            return 0.0
        if delta_x == 1 and delta_y == 1:
            return 0.15
        if delta_x == 1 or delta_y == 1: # This covers (1,0) and (0,1)
            return 0.3
        return 0.0 # This covers (0,0)

    def _set_trapdoor_found(self, loc: Tuple[int, int]):
        """
        A trapdoor has been definitively found at 'loc'.
        Update the corresponding probability map to be 1.0 at that location
        and 0.0 everywhere else.
        """
        # print(f"Trapdoor found at {loc}!")
        # for i in range(30):
        #     print(f"Trapdoor found at {loc}!")

        r, c = loc
        if (r + c) % 2 == 0: # It was a white-square trapdoor
            if self.white_trapdoor_probs[r, c] < 1.0: # Only print if it's new info
                # print(f"** Confirmed WHITE trapdoor at {loc} **")
                self.white_trapdoor_probs = np.zeros((self.board_size, self.board_size))
                self.white_trapdoor_probs[r, c] = 1.0
        else: # It was a black-square trapdoor
            if self.black_trapdoor_probs[r, c] < 1.0:
                # print(f"** Confirmed BLACK trapdoor at {loc} **")
                self.black_trapdoor_probs = np.zeros((self.board_size, self.board_size))
                self.black_trapdoor_probs[r, c] = 1.0

    def _get_potential_egg_squares(
        self,
        board: board.Board,
        start_loc: Tuple[int, int],
        player_parity: int,
        all_obstacles: Set[Tuple[int, int]]
    ) -> int:
        """
        Uses BFS to find the count of all reachable, empty, valid egg-laying
        squares for a player.
        'all_obstacles' is a set of all impassable squares for *this* player.
        (FIXED based on "can move through own turds/eggs")
        """
        q = deque([start_loc])
        visited = {start_loc}
        potential_egg_count = 0

        # Check the starting square itself
        start_r, start_c = start_loc[1], start_loc[0]
        if (start_r + start_c) % 2 == player_parity and \
           start_loc not in all_obstacles:
            potential_egg_count += 1
            
        while q:
            current_loc = q.popleft()

            for direction in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
                
                next_loc = self._get_next_loc_stateless(current_loc, direction)

                if board.is_valid_cell(next_loc):
                    # --- THIS IS THE FIX ---
                    # Only check against the consolidated 'all_obstacles' set
                    if next_loc not in visited and \
                       next_loc not in all_obstacles:
                        
                        visited.add(next_loc)
                        q.append(next_loc)
                        
                        r, c = next_loc[1], next_loc[0]
                        
                        if (r + c) % 2 == player_parity:
                            potential_egg_count += 1
                    # --- END FIX ---
                            
        return potential_egg_count
    
    def _find_nearest_potential_egg_square(
        self,
        board: board.Board,
        start_loc: Tuple[int, int],
        player_parity: int,
        all_obstacles: Set[Tuple[int, int]]
    ) -> int:
        """
        Uses BFS to find the distance to the *nearest* reachable, empty, 
        valid egg-laying square.
        (FIXED based on "can move through own turds/eggs")
        """
        
        # Check the starting square itself
        start_r, start_c = start_loc[1], start_loc[0]
        if (start_r + start_c) % 2 == player_parity and \
           start_loc not in all_obstacles:
            return 0 # We are on a valid square

        q = deque([(start_loc, 0)]) # (loc, distance)
        visited = {start_loc}

        while q:
            current_loc, distance = q.popleft()

            for direction in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
                
                next_loc = self._get_next_loc_stateless(current_loc, direction)

                if board.is_valid_cell(next_loc):
                    # --- THIS IS THE FIX ---
                    # Only check against the consolidated 'all_obstacles' set
                    if next_loc not in visited and next_loc not in all_obstacles:
                        
                        visited.add(next_loc)
                        
                        r, c = next_loc[1], next_loc[0]
                        
                        if (r + c) % 2 == player_parity:
                            return distance + 1
                        
                        q.append((next_loc, distance + 1))
                    # --- END FIX ---
                            
        return -1 # No potential egg squares are reachable

    def _get_next_loc_stateless(self, loc: Tuple[int, int], direction: Direction) -> Tuple[int, int]:
        """
        A stateless helper to calculate the next (x, y) location from a given loc.
        """
        x, y = loc
        if direction == Direction.UP:
            return (x, y - 1)
        elif direction == Direction.DOWN:
            return (x, y + 1)
        elif direction == Direction.LEFT:
            return (x - 1, y)
        elif direction == Direction.RIGHT:
            return (x + 1, y)
        return loc # Should not happen
    
    def _get_manhattan_distance(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> int:
        """Calculates Manhattan distance between two (x, y) locations."""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        my_current_loc = board.chicken_player.get_location()
        my_spawn = board.chicken_player.get_spawn()
        opponent_current_loc = board.chicken_enemy.get_location()
        opponent_spawn = board.chicken_enemy.get_spawn()

        # --- 1. Mark Current Location as Visited ---
        my_r, my_c = my_current_loc[1], my_current_loc[0]
        self.visited_squares[my_r, my_c] = 1

        # --- 2. Check for Found Trapdoors (Deduction) ---
        
        if (self.my_last_loc is not None and 
            my_current_loc == my_spawn and 
            self.my_last_loc != my_spawn):
            self._set_trapdoor_found(self.my_last_loc)

        if (self.opponent_last_loc is not None and 
            opponent_current_loc == opponent_spawn and 
            self.opponent_last_loc != opponent_spawn):
            self._set_trapdoor_found(self.opponent_last_loc)


        # --- 3. Bayesian Update (Inference) ---
        (hear_white, feel_white) = sensor_data[0]
        (hear_black, feel_black) = sensor_data[1]

        white_likelihood_map = np.zeros((self.board_size, self.board_size))
        black_likelihood_map = np.zeros((self.board_size, self.board_size))

        for r in range(self.board_size):
            for c in range(self.board_size):
                hypothetical_loc = (c, r) # (x, y)
                
                p_hear = self._get_hear_prob(my_current_loc, hypothetical_loc)
                p_feel = self._get_feel_prob(my_current_loc, hypothetical_loc)
                
                p_hear_white = p_hear if hear_white else (1.0 - p_hear)
                p_feel_white = p_feel if feel_white else (1.0 - p_feel)
                white_likelihood_map[r, c] = p_hear_white * p_feel_white
                
                p_hear_black = p_hear if hear_black else (1.0 - p_hear)
                p_feel_black = p_feel if feel_black else (1.0 - p_feel)
                black_likelihood_map[r, c] = p_hear_black * p_feel_black

        self.white_trapdoor_probs = self.white_trapdoor_probs * white_likelihood_map
        self.black_trapdoor_probs = self.black_trapdoor_probs * black_likelihood_map

        sum_white = np.sum(self.white_trapdoor_probs)
        sum_black = np.sum(self.black_trapdoor_probs)

        if sum_white > 0:
            self.white_trapdoor_probs /= sum_white
        
        if sum_black > 0:
            self.black_trapdoor_probs /= sum_black
        
        # --- 4. Choose a Move (Corrected Territorial Heuristic) ---
        
        TRAPDOOR_THRESHOLD = 0.25
        EGG_BONUS = 1 
        EXPLORATION_BONUS = .5
        
        # --- Constants to FIX TURD-SPAM ---
        TURD_STRATEGIC_BONUS = 2.0  # Increased bonus for *good* turds
        STRATEGIC_TURD_THRESHOLD = 4 # Min squares a turd must block
        USELESS_TURD_PENALTY = 1.0   # Penalty for laying a useless turd
        DISTANCE_TO_OPPONENT_PENALTY = 0.05 # Small penalty per-square
        
        valid_moves = board.get_valid_moves()

        if not valid_moves:
            return (Direction.UP, MoveType.PLAIN)
        
        safe_scored_moves = []
        risky_moves = [] 
        
        # --- Get data needed for BFS ---
        my_turds_on_board = board.turds_player
        opp_turds_on_board = board.turds_enemy
        all_my_eggs = board.eggs_player
        all_opp_eggs = board.eggs_enemy
        my_parity = (board.chicken_player.get_spawn()[0] + board.chicken_player.get_spawn()[1]) % 2
        opp_parity = (board.chicken_enemy.get_spawn()[0] + board.chicken_enemy.get_spawn()[1]) % 2

        # --- CORRECTED OBSTACLE LOGIC ---
        # Obstacles for me: Opponent's turds and eggs
        my_obstacles = opp_turds_on_board.union(all_opp_eggs)
        
        # Obstacles for opponent: My turds and eggs
        opp_obstacles_base = my_turds_on_board.union(all_my_eggs)
        # --- END NEW LOGIC ---

        # --- FIX FOR FILL_MODE LOOP ---
        # For finding a new *potential* spot, we can't lay on
        # our own eggs either.
        obstacles_for_laying = my_obstacles.union(all_my_eggs)
        # --- END FIX ---

        # Check opponent's potential from their CURRENT location
        base_opp_potential = self._get_potential_egg_squares(
            board, opponent_current_loc, opp_parity,
            opp_obstacles_base # Opponent is blocked by my turds/eggs
        )

        base_my_potential = self._get_potential_egg_squares(
            board, my_current_loc, my_parity,
            obstacles_for_laying
        )
        
        FILL_MODE = (base_opp_potential == 0 and base_my_potential > 0)

        for move in valid_moves:
            direction, move_type = move
            next_loc = board.chicken_player.get_next_loc(direction)
            
            x, y = next_loc
            r, c = y, x # Convert (x, y) to (row, col)
            
            trap_prob = 0.0
            if (r + c) % 2 == 0: # White square
                trap_prob = self.white_trapdoor_probs[r, c]
            else: # Black square
                trap_prob = self.black_trapdoor_probs[r, c]
            
            if trap_prob < TRAPDOOR_THRESHOLD:
                # This is a SAFE move. Score it.
                score = 0.0

                if FILL_MODE:
                    # --- FILL MODE SCORING ---
                    # Goal: Get to the nearest egg square and lay an egg.
                    
                    # 1. Find distance to nearest potential egg square from `next_loc`
                    dist = self._find_nearest_potential_egg_square(
                        board, next_loc, my_parity,
                        obstacles_for_laying # <-- USE THE CORRECT SET
                    )
                    
                    # 2. Score is inversely proportional to distance.
                    if dist == 0: # We are ON an egg square
                        score = 2000.0 
                    elif dist > 0: # We are near an egg square
                        score = 1000.0 / dist
                    else: # No path to an egg square
                        score = 0.0
                    
                    # 3. Add HUGE bonus for laying an egg *now*
                    if move_type == MoveType.EGG:
                        score += 5000.0 # High bonus to prioritize laying
                
                else:
                    # --- TERRITORY MODE SCORING ---
                    
                    # My potential (doesn't change with my turds)
                    my_potential_eggs = self._get_potential_egg_squares(
                        board, next_loc, my_parity,
                        obstacles_for_laying # Pass my correct obstacles
                    )
                    
                    # Opponent's potential (changes if I lay a turd)
                    simulated_opp_obstacles = opp_obstacles_base.copy()
                    if move_type == MoveType.TURD:
                        simulated_opp_obstacles.add(next_loc)
                    
                    opp_potential_eggs = self._get_potential_egg_squares(
                        board, opponent_current_loc, opp_parity,
                        simulated_opp_obstacles # Pass opponent's correct obstacles
                    )

                    score = my_potential_eggs - opp_potential_eggs
                    
                    if move_type == MoveType.EGG:
                        score += EGG_BONUS
                    
                    if move_type == MoveType.TURD:
                        reduction = base_opp_potential - opp_potential_eggs
                        if reduction >= STRATEGIC_TURD_THRESHOLD:
                            score += TURD_STRATEGIC_BONUS
                        else:
                            # This penalizes useless turds
                            score -= USELESS_TURD_PENALTY
                            

                if self.visited_squares[r, c] == 0:
                    score += EXPLORATION_BONUS
                
                safe_scored_moves.append((score, move))
            
            else:
                risky_moves.append((trap_prob, move))

        # --- Select the best move ---
        
        best_move = None
        
        if safe_scored_moves:
            safe_scored_moves.sort(key=lambda item: item[0], reverse=True)
            best_move = safe_scored_moves[0][1]
            
        elif risky_moves:
            risky_moves.sort(key=lambda item: item[0])
            best_move = risky_moves[0][1]
            
        else:
             best_move = valid_moves[0]

        
        # --- 5. Store state for next turn's deductions ---
        direction, _ = best_move
        
        self.my_last_loc = board.chicken_player.get_next_loc(direction)
        self.opponent_last_loc = opponent_current_loc

        return best_move