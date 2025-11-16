from collections.abc import Callable
from time import sleep
from typing import List, Set, Tuple

import numpy as np
from game import board
from game.enums import Direction, MoveType
from . import decision_maker

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

        self._initialize_probabilities()

        
        # print("initialized.")
        # print("Initial White Trapdoor Probabilities:")
        # print(np.round(self.white_trapdoor_probs, 3))
        # print("Initial Black Trapdoor Probabilities:")
        # print(np.round(self.black_trapdoor_probs, 3))


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

        # print(f"\n--- Turn {board.turn_count} ---")
        # print(f"My Location: {my_current_loc}. Sensor Data: {sensor_data}")

        # --- 1. Check for Found Trapdoors (Deduction) ---
        
        # Check if WE were reset (we are at spawn, but our last *intended* location was not spawn)
        if (self.my_last_loc is not None and 
            my_current_loc == my_spawn and 
            self.my_last_loc != my_spawn):
            # print(f"I was reset! Trapdoor must be at {self.my_last_loc}")
            self._set_trapdoor_found(self.my_last_loc)

        # Check if OPPONENT was reset
        if (self.opponent_last_loc is not None and 
            opponent_current_loc == opponent_spawn and 
            self.opponent_last_loc != opponent_spawn):
            # print(f"Opponent was reset! Trapdoor must be at {self.opponent_last_loc}")
            self._set_trapdoor_found(self.opponent_last_loc)


        # --- 2. Bayesian Update (Inference) ---
        (hear_white, feel_white) = sensor_data[0]
        (hear_black, feel_black) = sensor_data[1]

        # Create likelihood maps: P(Observation | Trapdoor at [r, c])
        white_likelihood_map = np.zeros((self.board_size, self.board_size))
        black_likelihood_map = np.zeros((self.board_size, self.board_size))

        for r in range(self.board_size):
            for c in range(self.board_size):
                hypothetical_loc = (c, r) # (x, y)
                
                p_hear = self._get_hear_prob(my_current_loc, hypothetical_loc)
                p_feel = self._get_feel_prob(my_current_loc, hypothetical_loc)

                # P(Observation) = P(Hear) * P(Feel) since they are independent
                
                # Likelihood for White Trapdoor
                p_hear_white = p_hear if hear_white else (1.0 - p_hear)
                p_feel_white = p_feel if feel_white else (1.0 - p_feel)
                white_likelihood_map[r, c] = p_hear_white * p_feel_white
                
                # Likelihood for Black Trapdoor
                p_hear_black = p_hear if hear_black else (1.0 - p_hear)
                p_feel_black = p_feel if feel_black else (1.0 - p_feel)
                black_likelihood_map[r, c] = p_hear_black * p_feel_black

        # P(Trap | Obs) = P(Obs | Trap) * P(Trap)
        self.white_trapdoor_probs = self.white_trapdoor_probs * white_likelihood_map
        self.black_trapdoor_probs = self.black_trapdoor_probs * black_likelihood_map

        # Re-normalize
        sum_white = np.sum(self.white_trapdoor_probs)
        sum_black = np.sum(self.black_trapdoor_probs)

        if sum_white > 0:
            self.white_trapdoor_probs /= sum_white
        
        if sum_black > 0:
            self.black_trapdoor_probs /= sum_black
        
        # --- 3. (DEBUG) Print Updated Probabilities ---
        # print("Updated White Trapdoor Probabilities:")
        # print(np.round(self.white_trapdoor_probs, 2))
        # print("Updated Black Trapdoor Probabilities:")
        # print(np.round(self.black_trapdoor_probs, 2))
        
        # --- 4. Choose a Move (Yolanda's random logic) ---
        # TODO: Replace this with a smart move, e.g., Minimax
        # Your heuristic should use:
        #   - (my_eggs - opp_eggs)
        #   - Probabilities of stepping on a trapdoor (from self.white_trapdoor_probs, etc.)
        
        return decision_maker.choose_move(
            board,
            board.get_valid_moves(),
            board.chicken_enemy.get_location()
        )

        # valid_moves = board.get_valid_moves()
        # possible_directions = np.unique([move[0] for move in valid_moves if move[1] == MoveType.PLAIN])
        # # print(possible_directions)

        # # print(result)
        # if not valid_moves:
        #     # No valid moves, this is a losing state.
        #     # We must return *something*, but it won't be used.
        #     print("No valid moves available!")
        #     # ye lowk we're cooked
        #     return (Direction.UP, MoveType.PLAIN)
        
        # ## CLAUDE'S

        # # # For now, just pick a random valid move
        # # result = result[np.random.randint(len(result))]
        # # Calculate Manhattan distance to opponent for each move
        # min_distance = float('inf')
        # best_direction = None

        # for direction in possible_directions:
        #     next_loc = board.chicken_player.get_next_loc(direction)
        #     distance_to_opponent = abs(next_loc[0] - opponent_current_loc[0]) + abs(next_loc[1] - opponent_current_loc[1])

        #     if distance_to_opponent < min_distance:
        #         min_distance = distance_to_opponent
        #         best_direction = direction

        # # Map np.int64 values to Direction enum
        # direction_map = {
        #     0: Direction.UP,
        #     1: Direction.RIGHT,
        #     2: Direction.DOWN,
        #     3: Direction.LEFT
        # }

        # # Convert best_direction to the corresponding Direction enum
        # if best_direction in direction_map:
        #     best_direction = direction_map[best_direction]

        # # If within two blocks of the opponent, lay a turd
        # if min_distance <= 2 and board.chicken_player.has_turds_left() < 5 and (best_direction, MoveType.TURD) in valid_moves:
        #     best_move = (best_direction, MoveType.TURD)
        # elif (best_direction, MoveType.EGG) in valid_moves:
        #     best_move = (best_direction, MoveType.EGG)
        # else:
        #     best_move = (best_direction, MoveType.PLAIN)

        # result = best_move



        ## CLAUDE'S END



        
        # print(f"Playing move: {result}")

        # --- 5. Store state for next turn's deductions ---
        # direction, _ = result
        # print(result)
        # # We store our *intended* next location
        # self.my_last_loc = board.chicken_player.get_next_loc(direction)
        # # We store the opponent's *current* location
        # self.opponent_last_loc = opponent_current_loc

        # return result