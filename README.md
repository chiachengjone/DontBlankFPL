#  FPL Strategy Engine 2025/26

A high-performance Streamlit application for Fantasy Premier League strategy optimization, built specifically for the **2025/26 season rules**.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.32+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

##  Features

### 2025/26 Season Rules Implemented

- **5 Free Transfer Rollover**: Plan strategically with up to 5 accumulating free transfers
- **1.25x Captain Multiplier**: Updated optimization for the new captaincy boost
- **CBIT Scoring (DefCon Engine)**: +2 bonus points for 10+ Clearances, Blocks, Interceptions, or Tackles

### Core Functionality

####  Tab 1: My Strategy
- AI-powered squad optimization using Integer Linear Programming (PuLP)
- Adjustable look-ahead window (3-10 gameweeks) with decay weighting
- **Ghost Points Logic**: Automatically penalizes players with blank gameweeks
- Multiple optimization modes:
  - Maximum Points
  - Differential Focus  
  - Value Picks
  - Balanced

####  Tab 2: Rival Scout
- Direct comparison of two Team IDs
- **Jaccard Similarity** calculation for squad overlap
- **Tactical Delta**: Identifies the biggest threat player in your rival's squad
- Visual overlap analysis

####  Tab 3: Feature Factory
- Searchable player database with engineered features
- **Differential Finder Index**: $\text{Differential Score} = \frac{EP}{EO\%} \times \text{Fixture Ease}$
- CBIT Propensity Scores for defenders
- Price sensitivity (xG/£) analysis
- Interactive Plotly visualizations

##  Project Structure

```
DontBlankFPL/
├── app.py              # Streamlit main application
├── fpl_api.py          # FPL API integration & data processing
├── optimizer.py        # PuLP optimization engine
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

##  Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/DontBlankFPL.git
cd DontBlankFPL
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

##  Mathematical Models

### Expected Points Calculation

The engine calculates weighted expected points over N gameweeks:

$$EP_{total} = \sum_{i=1}^{N} EP_{base} \times \text{Decay}^{i-1} \times \text{FDR}_{multiplier}$$

Where:
- Decay factor = 0.95 (configurable)
- FDR multiplier adjusts for fixture difficulty

### Optimization Objective

The PuLP solver maximizes:

$$\max \sum_{p \in P} \left( x_p \cdot EP_p + c_p \cdot EP_p \cdot (1.25 - 1) \right)$$

Subject to constraints:
- Budget: $\sum x_p \cdot \text{price}_p \leq 100$
- Team limit: $\leq 3$ players per team
- Position requirements: GKP(2), DEF(5), MID(5), FWD(3)
- Free transfers: $\leq 5$ rollover cap

### Differential Score

$$\text{Differential} = \frac{EP}{EO\%} \times \text{Fixture Ease} \times 100$$

Flags players with EP > 3.0 and ownership < 5%.

### CBIT Propensity (DefCon Engine)

$$\text{CBIT}_{per90} = \frac{\text{Estimated CBIT Actions}}{\text{Minutes}} \times 90$$

Expected CBIT bonus = $P(\text{CBIT} \geq 10) \times 2$

##  API Integration

### FPL Official API Endpoints
- `/bootstrap-static/` - Player, team, and gameweek data
- `/entry/{team_id}/event/{gw}/picks/` - User squad picks
- `/element-summary/{player_id}/` - Player history and fixtures
- `/fixtures/` - Season fixtures

### The Odds API (Optional)
Include your API key to enable betting odds integration:
- Anytime goalscorer odds → goal probability
- Clean sheet odds → CS probability

Set via environment variable: `ODDS_API_KEY`

##  Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ODDS_API_KEY` | The Odds API key for betting data | None (uses dummy data) |

### Customization Points

In `optimizer.py`:
- `CAPTAIN_MULTIPLIER`: Default 1.25 (2025/26 rule)
- `MAX_FREE_TRANSFERS`: Default 5 (2025/26 rule)
- `CBIT_BONUS_THRESHOLD`: Default 10
- `CBIT_BONUS_POINTS`: Default 2

##  Usage Tips

1. **Start with Load Data**: Always click "Load/Refresh Data" first
2. **Set Your Team ID**: Find it in your FPL URL (e.g., `/entry/123456/`)
3. **Adjust Look-ahead Window**: 
   - 3-5 weeks for short-term gains
   - 6-10 weeks for long-term planning
4. **Use Differential Mode** when chasing in mini-leagues
5. **Check Feature Factory** for hidden gems before transfers

##  Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

##  License

MIT License - see LICENSE file for details.

##  Disclaimer

This tool is for educational and entertainment purposes. It uses publicly available FPL API data. Not affiliated with the Premier League or Fantasy Premier League.

---

Built with  for FPL managers who want an edge in 2025/26
