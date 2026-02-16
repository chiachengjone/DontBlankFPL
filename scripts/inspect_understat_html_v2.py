import requests
import re
import json

season = 2024
url = f"https://understat.com/league/EPL/{season}"

try:
    resp = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }
    )
    html = resp.text
    print(f"Status Code: {resp.status_code}")
    print(f"HTML Length: {len(html)}")
    
    # Check for specific variable names often used
    vars = ["statisticsData", "teamsData", "playersData"]
    for v in vars:
        if v in html:
            print(f"Found variable name: {v}")
    
    # Broad regex
    matches = re.finditer(r"var\s+([a-zA-Z0-9_]+)\s*=\s*JSON\.parse\('([^']+)'\)", html)
    found = False
    for match in matches:
        found = True
        var_name = match.group(1)
        json_str = match.group(2).encode().decode('unicode_escape')
        try:
            data = json.loads(json_str)
            print(f"--- Found JSON for variable: {var_name} ---")
            if isinstance(data, dict):
                print("Keys:", list(data.keys())[:5])
                # Check for desired keys
                if var_name == 'statisticsData':
                    if 'team' in data:
                         print("Found team keys in statisticsData!")
        except Exception as e:
            print(f"Failed to parse JSON for {var_name}: {e}")

    if not found:
        print("No JSON.parse patterns found.")

except Exception as e:
    print(f"Error: {e}")
