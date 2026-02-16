
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_fpl_keys():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    logger.info(f"Fetching {url}...")
    
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    data = resp.json()
    
    elements = data.get('elements', [])
    if elements:
        player = elements[0]
        keys = player.keys()
        logger.info("--- Player Keys ---")
        
        target_keys = [
            'corners_and_indirect_freekicks_order',
            'direct_freekicks_order',
            'penalties_order',
            'corners_and_indirect_freekicks_text',
            'direct_freekicks_text',
            'penalties_text'
        ]
        
        found = [k for k in target_keys if k in keys]
        missing = [k for k in target_keys if k not in keys]
        
        logger.info(f"Found: {found}")
        logger.info(f"Missing: {missing}")

if __name__ == "__main__":
    inspect_fpl_keys()
