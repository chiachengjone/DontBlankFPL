import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from understat_api import fetch_understat_league_data, build_team_stats_df

print("Fetching data...")
_, teams_raw = fetch_understat_league_data()
if teams_raw:
    print(f"Fetched {len(teams_raw)} teams.")
    df = build_team_stats_df(teams_raw)
    print("Columns:", df.columns.tolist())
    
    # Check Man City
    mancity = df[df['understat_team'] == 'Manchester City']
    if not mancity.empty:
        row = mancity.iloc[0]
        goals = row['goals_per90'] * row['games_played']
        print(f"Man City: Games={row['games_played']}, Goals/90={row['goals_per90']:.2f}, Total Calculated Goals={goals:.1f}")
        
    # Check Arsenal
    arsenal = df[df['understat_team'] == 'Arsenal']
    if not arsenal.empty:
        row = arsenal.iloc[0]
        goals = row['goals_per90'] * row['games_played']
        print(f"Arsenal: Games={row['games_played']}, Goals/90={row['goals_per90']:.2f}, Total Calculated Goals={goals:.1f}")
else:
    print("No team data available.")
