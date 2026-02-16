import requests
import re
import json

season = 2024
url = f"https://understat.com/league/EPL/{season}"

try:
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    html = resp.text
    
    # Look for statisticsData
    match = re.search(r"var\s+statisticsData\s*=\s*JSON\.parse\('([^']+)'\)", html)
    if match:
        # Decode unicode escapes
        json_str = match.group(1).encode().decode('unicode_escape')
        data = json.loads(json_str)
        print("Keys in statisticsData:", data.keys())
        if 'team' in data:
            first_team_id = list(data['team'].keys())[0]
            print("Sample Team Data:", data['team'][first_team_id])
    else:
        print("statisticsData not found in HTML.")

except Exception as e:
    print(f"Error: {e}")
