from collections.abc import Callable
from time import sleep
from typing import List, Set, Tuple

import numpy as np
from game import board
from game.enums import Direction, MoveType

"""
    hey hillegass.
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
        
        self.white_trapdoor_probs = np.zeros((self.board_size, self.board_size))
        self.black_trapdoor_probs = np.zeros((self.board_size, self.board_size))
        
        # Track previous locations to deduce when a trapdoor is hit
        self.my_last_loc = None
        self.opponent_last_loc = None

        # --- NEW: Add memory for exploration ---
        # Track which squares we have already visited
        self.visited_squares = np.zeros((self.board_size, self.board_size))
        # --- End of New ---

        self._initialize_probabilities()
        self.parity = (board.chicken_player.get_spawn()[0] + board.chicken_player.get_spawn()[1]) % 2




    def _initialize_probabilities(self):
        """
        Sets the initial weighted probabilities for trapdoor locations
        based on the rules in trapdoor_manager.py and assignment.pdf.
        """
        weights = np.zeros((self.board_size, self.board_size))
        
        weights[2:6, 2:6] = 1.0
        weights[3:5, 3:5] = 2.0

        total_white_weight = 0
        total_black_weight = 0

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

    def get_canonical_state(self, board_obj):
        """
        Normalizes the board so the Agent ALWAYS plays as if it is:
        1. On the Left Edge (x=0)
        2. In the Top Half (y < 4)
        
        The Opponent will always appear on the Right Edge (x=7), Top Half (y < 4).
        """
        # Get raw locations
        my_loc = board_obj.chicken_player.get_location()
        opp_loc = board_obj.chicken_enemy.get_location()
        
        spawn = board_obj.chicken_player.get_spawn()
        
        # --- STEP 1: DETECT TRANSFORMS ---
        # 1. Transpose (Swap X/Y) if we are on Top/Bottom edges
        #    (This moves us to Left/Right edges)
        self.needs_transpose = (spawn[1] == 0 or spawn[1] == self.board_size - 1)
        
        # Apply Transpose temporarily to check for Flips
        temp_x = spawn[1] if self.needs_transpose else spawn[0]
        temp_y = spawn[0] if self.needs_transpose else spawn[1]
        
        # 2. Horizontal Flip (if we are on the Right side)
        self.needs_h_flip = (temp_x >= self.board_size // 2)
        
        # 3. Vertical Flip (if we are on the Bottom half)
        self.needs_v_flip = (temp_y >= self.board_size // 2)

        # --- HELPER FUNCTION ---
        def transform(loc):
            x, y = loc
            # 1. Transpose
            if self.needs_transpose:
                x, y = y, x
            # 2. Horizontal Flip (x -> 7-x)
            if self.needs_h_flip:
                x = (self.board_size - 1) - x
            # 3. Vertical Flip (y -> 7-y)
            if self.needs_v_flip:
                y = (self.board_size - 1) - y
            return (x, y)

        # --- CREATE CANONICAL STATE ---
        canonical_state = {
            "my_loc": transform(my_loc),
            "opp_loc": transform(opp_loc),
            "my_eggs": {transform(l) for l in board_obj.eggs_player},
            "opp_eggs": {transform(l) for l in board_obj.eggs_enemy},
            "my_turds": {transform(l) for l in board_obj.turds_player},
            "opp_turds": {transform(l) for l in board_obj.turds_enemy}
        }
        
        return canonical_state

    def get_real_move(self, canonical_move):
        """
        Converts the 'Canonical Move' (made in the normalized Top-Left world)
        back into the 'Real World Move' by reversing the transforms.
        """
        direction, move_type = canonical_move
        
        # Reverse order: V-Flip -> H-Flip -> Transpose
        
        # 1. Reverse Vertical Flip (UP <-> DOWN)
        if self.needs_v_flip:
            if direction == Direction.UP: direction = Direction.DOWN
            elif direction == Direction.DOWN: direction = Direction.UP
            
        # 2. Reverse Horizontal Flip (LEFT <-> RIGHT)
        if self.needs_h_flip:
            if direction == Direction.LEFT: direction = Direction.RIGHT
            elif direction == Direction.RIGHT: direction = Direction.LEFT
            
        # 3. Reverse Transpose (Swap Axes)
        # Map: UP->LEFT, DOWN->RIGHT, LEFT->UP, RIGHT->DOWN
        if self.needs_transpose:
            if direction == Direction.UP: direction = Direction.LEFT
            elif direction == Direction.DOWN: direction = Direction.RIGHT
            elif direction == Direction.LEFT: direction = Direction.UP
            elif direction == Direction.RIGHT: direction = Direction.DOWN
            
        return (direction, move_type)

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
        # (We do this first, so we don't try to "explore" the square we're on)
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
        
        # --- 4. Choose a Move (Territorial Heuristic) ---
        
        TRAPDOOR_THRESHOLD = 0.25 # Our risk tolerance
        EXPLORATION_BONUS = 100  # A large bonus for visiting a new square
        
        valid_moves = board.get_valid_moves()

        if not valid_moves:
            # print("No valid moves available!")
            return (Direction.UP, MoveType.PLAIN)
        
        ## AEDAN
        # sets me to the left, opp on the right.
        # cstate = canonical state
        cstate = self.get_canonical_state(board)
        if board.turns_left_player > 38:
            if board.chicken_player.can_lay_egg():
                return self.get_real_move((Direction.RIGHT, MoveType.EGG))
            # elif cstate # I'M DOING THIS RIGHT NOW
            return self.get_real_move((Direction.RIGHT, MoveType.PLAIN))

        # Store moves as (score, move) tuples for sorting
        safe_egg_moves = []
        safe_turd_moves = []
        safe_plain_moves = []
        risky_moves = [] 

        for move in valid_moves:
            direction, move_type = move
            next_loc = board.chicken_player.get_next_loc(direction)
            
            x, y = next_loc
            r, c = y, x # Convert (x, y) to (row, col)
            
            # Get trapdoor probability
            trap_prob = 0.0
            if (r + c) % 2 == 0: # White square
                trap_prob = self.white_trapdoor_probs[r, c]
            else: # Black square
                trap_prob = self.black_trapdoor_probs[r, c]
            
            # --- Categorize the move by safety ---
            if trap_prob < TRAPDOOR_THRESHOLD:
                # This is a SAFE move. Score it.
                score = 0
                
                # 1. Exploration Bonus (Maximize places you've been)
                if self.visited_squares[r, c] == 0:
                    score += EXPLORATION_BONUS
                
                # 2. Strategic Bonus (based on opponent)
                dist_to_opp = abs(next_loc[0] - opponent_current_loc[0]) + abs(next_loc[1] - opponent_current_loc[1])

                if move_type == MoveType.EGG:
                    # Prioritize safe eggs, far from opponent
                    score += dist_to_opp 
                    safe_egg_moves.append((score, move))
                
                elif move_type == MoveType.TURD:
                    # Prioritize tactical turds, close to opponent
                    # We use (self.board_size * 2 - dist_to_opp) to reward closeness
                    score += (self.board_size * 2 - dist_to_opp)
                    safe_turd_moves.append((score, move))
                
                else: # MoveType.PLAIN
                    # Prioritize moving *towards* opponent to claim space
                    score += (self.board_size * 2 - dist_to_opp)
                    safe_plain_moves.append((score, move))
            
            else:
                # This is a RISKY move.
                # Score is just its probability (lower is better)
                risky_moves.append((trap_prob, move))

        # --- Select the best move based on priority and score ---
        
        best_move = None
        
        if safe_egg_moves:
            # Priority 1: Lay the BEST safe egg
            safe_egg_moves.sort(key=lambda item: item[0], reverse=True) # Sort by score, descending
            best_move = safe_egg_moves[0][1]
            # print(f"Playing best EGG move: {best_move} (Score: {safe_egg_moves[0][0]})")
            
        elif safe_turd_moves:
            # Priority 2: Lay the BEST safe turd
            safe_turd_moves.sort(key=lambda item: item[0], reverse=True)
            best_move = safe_turd_moves[0][1]
            # print(f"Playing best TURD move: {best_move} (Score: {safe_turd_moves[0][0]})")
            
        elif safe_plain_moves:
            # Priority 3: Make the BEST safe plain move
            safe_plain_moves.sort(key=lambda item: item[0], reverse=True)
            best_move = safe_plain_moves[0][1]
            # print(f"Playing best PLAIN move: {best_move} (Score: {safe_plain_moves[0][0]})")
            
        else:
            # Fallback: No safe moves. Pick the *least* risky move.
            risky_moves.sort(key=lambda item: item[0]) # Sort by prob, ascending
            best_move = risky_moves[0][1]
            # print(f"No safe moves. Playing LEAST RISKY move: {best_move} (Prob: {risky_moves[0][0]:.3f})")

        
        # --- 5. Store state for next turn's deductions ---
        direction, _ = best_move
        
        self.my_last_loc = board.chicken_player.get_next_loc(direction)
        self.opponent_last_loc = opponent_current_loc

        return best_move