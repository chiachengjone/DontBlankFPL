import requests
import json

season = 2024
url = f"https://understat.com/getLeagueData/EPL/{season}"

try:
    resp = requests.post(
        url,
        data={},
        headers={
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://understat.com/league/EPL/{season}",
        }
    )
    data = resp.json()
    teams = data.get('teams', {})
    
    if teams:
        first_team_id = list(teams.keys())[0]
        first_team = teams[first_team_id]
        print(f"Keys for team {first_team.get('title')}: {list(first_team.keys())}")
        print("Sample History Item keys:", list(first_team['history'][0].keys()) if first_team.get('history') else "No history")
        
        # Check if there are other keys besides 'id', 'title', 'history'
        # Maybe 'statistics'?
        print("Full Team Object Keys:", first_team.keys())
    else:
        print("No teams found.")

except Exception as e:
    print(f"Error: {e}")
