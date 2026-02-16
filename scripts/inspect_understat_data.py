
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_data():
    url = "https://understat.com/getLeagueData/EPL/2024"
    logger.info(f"Fetching {url}...")
    
    resp = requests.post(
        url,
        data={},
        headers={
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://understat.com/league/EPL/2024"
        }
    )
    data = resp.json()
    
    players = data.get('players', [])
    teams = data.get('teams', {})
    
    logger.info(f"Fetched {len(players)} players and {len(teams)} teams.")
    
    if players:
        logger.info("--- Player Keys ---")
        logger.info(list(players[0].keys()))
        logger.info("--- Sample Player ---")
        logger.info(json.dumps(players[0], indent=2))
        
    if teams:
        first_team = list(teams.values())[0]
        logger.info("--- Team Keys ---")
        logger.info(list(first_team.keys()))
        logger.info("--- Sample Team ---")
        # Truncate history for brevity
        if 'history' in first_team:
            first_team['history'] = first_team['history'][:1]
        logger.info(json.dumps(first_team, indent=2))

if __name__ == "__main__":
    inspect_data()
