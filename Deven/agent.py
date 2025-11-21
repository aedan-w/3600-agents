import os
import numpy as np
from collections.abc import Callable
from typing import List, Tuple

from game import board as game_board
from game.enums import Direction, MoveType

# --- 1. Setup PyTorch ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- 2. Define the Network ---
if TORCH_AVAILABLE:
    class SimpleNet(nn.Module):
        # CHANGED: spatial_channels default increased from 7 to 8
        def __init__(self, spatial_channels=8, non_spatial_features=4):
            super(SimpleNet, self).__init__()
            self.conv1 = nn.Conv2d(spatial_channels, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            conv_out_size = 32 * 8 * 8
            self.fc1 = nn.Linear(conv_out_size + non_spatial_features, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)

        def forward(self, x_spatial, x_non_spatial):
            x = F.relu(self.conv1(x_spatial))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1) 
            combined = torch.cat((x, x_non_spatial), dim=1)
            combined = F.relu(self.fc1(combined))
            combined = F.relu(self.fc2(combined))
            score = torch.tanh(self.fc3(combined))
            return score

class PlayerAgent:
    def __init__(self, board: game_board.Board, time_left: Callable):
        self.net = None
        self.device = torch.device("cpu")
        self.board_size = 8
        
        # --- TRUMAN LOGIC: Init Probabilities ---
        self.white_trapdoor_probs = np.zeros((8, 8))
        self.black_trapdoor_probs = np.zeros((8, 8))
        self.my_last_loc = None
        self.opponent_last_loc = None
        self._initialize_probabilities()
        # ----------------------------------------

        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            
            # Initialize Net with 8 channels
            self.net = SimpleNet(spatial_channels=8).to(self.device)
            
            script_location = os.path.dirname(os.path.realpath(__file__))
            model_path = os.path.join(script_location, "agent_model.pth")
            
            if os.path.exists(model_path):
                try:
                    # Note: This might fail if you load an OLD 7-channel model.
                    # You will need to RETRAIN immediately after this.
                    self.net.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                    self.net.eval()
                except Exception as e:
                    print(f"Castle Warning: Failed to load model (Arch Mismatch?): {e}")
                    self.net = None 
            else:
                self.net = None

    # --- TRUMAN HELPER METHODS ---
    def _initialize_probabilities(self):
        weights = np.zeros((8, 8))
        weights[2:6, 2:6] = 1.0
        weights[3:5, 3:5] = 2.0
        total_white = 0; total_black = 0
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 0:
                    self.white_trapdoor_probs[r, c] = weights[r, c]
                    total_white += weights[r, c]
                else:
                    self.black_trapdoor_probs[r, c] = weights[r, c]
                    total_black += weights[r, c]
        if total_white > 0: self.white_trapdoor_probs /= total_white
        if total_black > 0: self.black_trapdoor_probs /= total_black

    def _get_hear_prob(self, loc1, loc2):
        dx = abs(loc1[0] - loc2[0]); dy = abs(loc1[1] - loc2[1])
        if dx > 2 or dy > 2: return 0.0
        if dx == 2 and dy == 2: return 0.0
        if dx == 2 or dy == 2: return 0.1
        if dx == 1 and dy == 1: return 0.25
        if dx == 1 or dy == 1: return 0.5
        return 0.0

    def _get_feel_prob(self, loc1, loc2):
        dx = abs(loc1[0] - loc2[0]); dy = abs(loc1[1] - loc2[1])
        if dx > 1 or dy > 1: return 0.0
        if dx == 1 and dy == 1: return 0.15
        if dx == 1 or dy == 1: return 0.3
        return 0.0
    
    def _set_trapdoor_found(self, loc):
        r, c = loc
        if (r + c) % 2 == 0:
            self.white_trapdoor_probs = np.zeros((8, 8))
            self.white_trapdoor_probs[r, c] = 1.0
        else:
            self.black_trapdoor_probs = np.zeros((8, 8))
            self.black_trapdoor_probs[r, c] = 1.0
    # -----------------------------

    def _featurize(self, board_obj):
        # Existing 7 Channels
        my_loc_map = torch.zeros((8, 8), dtype=torch.float32)
        opp_loc_map = torch.zeros((8, 8), dtype=torch.float32)
        my_eggs_map = torch.zeros((8, 8), dtype=torch.float32)
        opp_eggs_map = torch.zeros((8, 8), dtype=torch.float32)
        my_turds_map = torch.zeros((8, 8), dtype=torch.float32)
        opp_turds_map = torch.zeros((8, 8), dtype=torch.float32)
        opp_turd_zone_map = torch.zeros((8, 8), dtype=torch.float32)

        my_loc = board_obj.chicken_player.get_location()
        my_loc_map[my_loc[1], my_loc[0]] = 1.0
        opp_loc = board_obj.chicken_enemy.get_location()
        opp_loc_map[opp_loc[1], opp_loc[0]] = 1.0

        for r in range(8):
            for c in range(8):
                loc = (c, r)
                if loc in board_obj.eggs_player: my_eggs_map[r, c] = 1.0
                if loc in board_obj.eggs_enemy: opp_eggs_map[r, c] = 1.0
                if loc in board_obj.turds_player: my_turds_map[r, c] = 1.0
                if loc in board_obj.turds_enemy: opp_turds_map[r, c] = 1.0
                if board_obj.is_cell_in_enemy_turd_zone(loc): opp_turd_zone_map[r, c] = 1.0
        
        # --- NEW CHANNEL 8: RISK MAP ---
        # We combine the white and black probs into one "Danger Map"
        risk_map = torch.tensor(self.white_trapdoor_probs + self.black_trapdoor_probs, dtype=torch.float32)
        
        spatial = torch.stack([
            my_loc_map, opp_loc_map, my_eggs_map, opp_eggs_map,
            my_turds_map, opp_turds_map, opp_turd_zone_map, risk_map
        ]).unsqueeze(0).to(self.device)

        non_spatial = torch.tensor([
            board_obj.turns_left_player / 40.0,
            board_obj.turns_left_enemy / 40.0,
            board_obj.chicken_player.get_turds_left() / 5.0,
            board_obj.chicken_enemy.get_turds_left() / 5.0
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        return spatial, non_spatial

    def play(self, board_obj: game_board.Board, sensor_data, time_left):
        my_current_loc = board_obj.chicken_player.get_location()
        my_spawn = board_obj.chicken_player.get_spawn()
        opponent_current_loc = board_obj.chicken_enemy.get_location()
        opponent_spawn = board_obj.chicken_enemy.get_spawn()

        # --- TRUMAN LOGIC: Update Beliefs ---
        if (self.my_last_loc is not None and my_current_loc == my_spawn and self.my_last_loc != my_spawn):
            self._set_trapdoor_found(self.my_last_loc)
        if (self.opponent_last_loc is not None and opponent_current_loc == opponent_spawn and self.opponent_last_loc != opponent_spawn):
            self._set_trapdoor_found(self.opponent_last_loc)

        (hear_white, feel_white) = sensor_data[0]
        (hear_black, feel_black) = sensor_data[1]
        
        white_update = np.zeros((8, 8)); black_update = np.zeros((8, 8))
        
        for r in range(8):
            for c in range(8):
                loc = (c, r)
                p_h = self._get_hear_prob(my_current_loc, loc)
                p_f = self._get_feel_prob(my_current_loc, loc)
                
                ph_w = p_h if hear_white else (1.0 - p_h)
                pf_w = p_f if feel_white else (1.0 - p_f)
                white_update[r, c] = ph_w * pf_w
                
                ph_b = p_h if hear_black else (1.0 - p_h)
                pf_b = p_f if feel_black else (1.0 - p_f)
                black_update[r, c] = ph_b * pf_b

        self.white_trapdoor_probs *= white_update
        self.black_trapdoor_probs *= black_update
        
        s_w = np.sum(self.white_trapdoor_probs)
        s_b = np.sum(self.black_trapdoor_probs)
        if s_w > 0: self.white_trapdoor_probs /= s_w
        if s_b > 0: self.black_trapdoor_probs /= s_b
        # --------------------------------------

        valid_moves = board_obj.get_valid_moves()
        if not valid_moves: return (Direction.UP, MoveType.PLAIN)

        if self.net is None:
            import random
            return random.choice(valid_moves)

        best_move = valid_moves[0]
        best_score = -float('inf')

        for move in valid_moves:
            direction, m_type = move
            future_board = board_obj.forecast_move(direction, m_type)
            
            if future_board is not None:
                # _featurize now includes the updated risk map!
                spatial, non_spatial = self._featurize(future_board)
                
                with torch.no_grad():
                    score = self.net(spatial, non_spatial).item()
                
                if m_type == MoveType.EGG: score += 0.1 
                if score > best_score:
                    best_score = score
                    best_move = move
        
        # Store state for next turn
        d, _ = best_move
        self.my_last_loc = board_obj.chicken_player.get_next_loc(d)
        self.opponent_last_loc = opponent_current_loc

        return best_move