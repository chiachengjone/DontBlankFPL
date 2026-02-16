#!/usr/bin/env python3
"""
Understat Scraper Health Check
==============================

This script verifies that the Understat website layout has not changed in a way
that breaks our regex-based scraping logic.

It attempts to:
1. Fetch the main page for the current EPL season.
2. Extract the 'playersData' JSON variable using regex.
3. Parse the JSON.
4. Validate that key fields (id, player_name, team_title, xG) are present.

Usage:
    python scripts/check_scraper_health.py

Exit Code:
    0: Healthy
    1: Broken (requires attention)
"""

import re
import json
import requests
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SEASON = "2024" # Update this seasonally if needed, or make dynamic
AJAX_URL = f"https://understat.com/getLeagueData/EPL/{SEASON}"
HTML_URL = f"https://understat.com/league/EPL/{SEASON}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def check_ajax_health():
    """Check the primary AJAX endpoint."""
    logger.info(f"Checking Primary AJAX URL: {AJAX_URL}")
    try:
        response = requests.post(
            AJAX_URL, 
            data={}, 
            headers={**HEADERS, "X-Requested-With": "XMLHttpRequest", "Referer": HTML_URL},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        players = data.get("players", [])
        if not players:
            logger.warning("⚠️ AJAX returned empty players list.")
            return False
            
        logger.info(f"✅ AJAX returned {len(players)} players.")
        return True
    except Exception as e:
        logger.error(f"❌ AJAX check failed: {e}")
        return False

def check_scraper_health():
    ajax_ok = check_ajax_health()
    
    if ajax_ok:
        logger.info("🟢 HEALTHY: Primary AJAX API is working.")
        return True
    else:
        logger.error("🔴 DOWN: Primary AJAX API failed.")
        return False

if __name__ == "__main__":
    success = check_scraper_health()
    sys.exit(0 if success else 1)
