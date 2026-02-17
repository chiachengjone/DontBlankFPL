import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from understat_api import fetch_understat_league_players, _build_understat_df

print("Fetching player data...")
raw = fetch_understat_league_players()
if raw:
    df = _build_understat_df(raw)
    print("Columns:", df.columns.tolist())
    
    # Check for a player who likely moved or has stats
    # Let's search for duplicate names in different teams
    dupes = df[df.duplicated(subset=['player_name'], keep=False)].sort_values('player_name')
    if not dupes.empty:
        print("\nFound players with multiple entries (checking if split by team):")
        print(dupes[['player_name', 'team_title', 'goals', 'assists', 'xG']].head(10))
    else:
        print("No duplicate player names found (maybe API pre-aggregates?)")
        
    # Check Arsenal total assists
    arsenal = df[df['team_title'] == 'Arsenal']
    total_assists_ars = arsenal['assists'].sum()
    print(f"\nArsenal (Understat) Total Assists: {total_assists_ars}")
    
    # Check Raheem Sterling
    sterling = df[df['player_name'].str.contains("Sterling", case=False)]
    if not sterling.empty:
        print("\nSterling Entries:")
        print(sterling[['player_name', 'team_title', 'goals', 'assists']])
    else:
        print("\nSterling not found.")
else:
    print("No player data available.")
