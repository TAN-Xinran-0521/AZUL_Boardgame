# AZUL Board Game - Python Implementation

This is a Python implementation of the popular board game AZUL (also known as "Azul: Stained Glass of Sintra" or "花砖物语" in Chinese) using Tkinter for the graphical interface. The game features complete rule implementation, a graphical user interface, and an AI opponent using Expectimax algorithm with heuristic evaluation.

## Features

- **Complete Game Rules**: Full implementation of AZUL official rules including factory market, pattern lines, wall grid, floor penalties, and endgame scoring
- **Graphical User Interface**: Intuitive interface built with Tkinter
- **Advanced AI Opponent**: AI decision-making using Expectimax search algorithm combined with sophisticated heuristic evaluation
- **Interactive Gameplay**: Click to select tiles from factories/center, then choose target row or floor
- **Adjustable AI Difficulty**: Customize AI search depth and performance parameters

## Requirements

- Python 3.6+
- Standard libraries only (no additional packages required):
  - tkinter
  - random
  - copy
  - collections

## How to Run

Execute the main program file:
```bash
python AZUL_GUI_buttons_top_final.py
```

## Game Controls

- **Restart**: Reset the current game
- **New Game**: Start a new game
- **End Game**: Force end the current game
- **Faster AI**: Decrease AI search depth (faster decisions, lower quality)
- **Slower AI**: Increase AI search depth (slower decisions, higher quality)

## Game Rules Overview

1. Players select all tiles of one color from either a factory or the center
2. Tiles are placed on the corresponding pattern line or on the floor
3. When all factories and the center are empty, the round ends and scoring occurs
4. The game triggers endgame when a player completes a row on their wall
5. Endgame scoring includes row completion bonuses, column completion bonuses, and color set bonuses

## AI Implementation Details

### Expectimax Algorithm
The AI uses an Expectimax search algorithm with configurable depth to evaluate possible moves. This algorithm:
- Alternates between MAX nodes (AI's turn) and MIN nodes (opponent's turn)
- Includes chance nodes for random factory refills
- Uses heuristic evaluation at depth limits

### Heuristic Evaluation Function
The AI employs a sophisticated heuristic evaluation function that considers multiple strategic factors.

```python
def heuristic_evaluate(self, state, player_idx):
    # Score difference between players
    score_diff = me.score - opp.score
    
    # Row completion potential
    my_row_pot = sum(max(0, 5 - (r+1 - len(me.pattern_lines[r]))) for r in range(5))
    opp_row_pot = sum(max(0, 5 - (r+1 - len(opp.pattern_lines[r]))) for r in range(5))
    
    # Color collection progress
    my_color_pot = sum(tile_count for tile_count in my_color_cnt.values())
    opp_color_pot = sum(tile_count for tile_count in opp_color_cnt.values())
    
    # Floor penalty avoidance
    my_floor = len(me.floor)
    opp_floor = len(opp.floor)
    
    # Blocking value (preventing opponent progress)
    block_value = sum(my_color_cnt[color] * 0.3 for color in COLORS)
    
    # Combined heuristic value
    value = (1.0 * score_diff + 
             0.5 * (my_row_pot - opp_row_pot) + 
             0.2 * (my_color_pot - opp_color_pot) - 
             0.6 * (my_floor - opp_floor) + 
             0.3 * block_value)
    return value
```

### Search Parameters
The AI behavior can be customized through these parameters:
- `SEARCH_DEPTH`: Controls how many moves ahead the AI looks (default: 5)
- `SAMPLES_PER_CHANCE`: Number of samples for random factory refills (default: 6)
- `TOP_K_MOVES`: Number of top moves to consider at each level (default: 7)

## Project Structure

- `Player` class: Manages player state (pattern lines, wall, floor, and score)
- `AzulGame` class: Core game logic and rule implementation
- `AzulGUI` class: Graphical user interface and interaction handling
- Expectimax algorithm with heuristic evaluation: Advanced AI decision implementation

## Customization Options

You can customize the game by modifying global variables at the top of the code:

```python
# Search hyperparameters (adjust based on machine performance)
SEARCH_DEPTH = 5            # Depth (higher = slower)
SAMPLES_PER_CHANCE = 6      # Number of samples for chance nodes
TOP_K_MOVES = 7             # Heuristic top-K moves to retain
```

## Performance Notes

The AI's performance depends on the search parameters:
- Higher `SEARCH_DEPTH` values provide better decisions but significantly increase computation time
- The current implementation uses pruning techniques to maintain reasonable performance
- On standard hardware, depth 5 provides a good balance between intelligence and responsiveness

## License

This project is for entertainment purposes only. 

---
