import numpy as np
from game import board
from game.enums import Direction, MoveType
from typing import List, Tuple

def choose_move(board: board.Board, valid_moves: List[Tuple[Direction, MoveType]], opponent_loc: Tuple[int, int]) -> Tuple[Direction, MoveType]:
    """
    Analyzes the board state and returns the best move.
    """

    # 1. Find the best direction to move (minimizing distance to opponent)
    min_distance = float('inf')
    best_direction = None

    # Get all unique, possible directions we can *at least* walk in
    possible_directions = np.unique([move[0] for move in valid_moves if move[1] == MoveType.PLAIN])
    
    if possible_directions.size == 0:
         # We are trapped and can only lay an egg or turd where we stand.
         # Just pick the first valid move (which will be EGG or TURD)
         best_move = valid_moves[0]
    else:
        for direction in possible_directions:
            next_loc = board.chicken_player.get_next_loc(direction)
            
            # Simple heuristic: move towards opponent
            distance_to_opponent = abs(next_loc[0] - opponent_loc[0]) + abs(next_loc[1] - opponent_loc[1])

            if distance_to_opponent < min_distance:
                min_distance = distance_to_opponent
                best_direction = direction

        # If no direction is valid (e.g., all plain moves are blocked),
        # just pick the direction from the first available move.
        if best_direction is None:
            best_direction = valid_moves[0][0] # Get direction from first valid move

        # 2. Decide on the move type (Turd, Egg, or Plain)
        # We will try to use our best_direction with the "best" move type.
        # We check in order of priority: TURD, EGG, PLAIN.
        
        turd_move = (best_direction, MoveType.TURD)
        egg_move = (best_direction, MoveType.EGG)
        plain_move = (best_direction, MoveType.PLAIN)

        # Check for TURD:
        if turd_move in valid_moves:
            # This is a valid move. The 'valid_moves' list already confirmed
            # we are not adjacent to the enemy, have turds left, etc.
            best_move = turd_move
        
        # Check for EGG:
        elif egg_move in valid_moves:
            # This is a valid move. The 'valid_moves' list already confirmed
            # our parity is correct and the square is empty.
            best_move = egg_move
        
        # Fallback to PLAIN:
        else:
            best_move = plain_move

    return best_move