"""
Retrospective Analysis Utilities
Calculates "Manager Report Card" metrics: Captaincy Efficiency, Bench Points, Transfer ROI.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

import concurrent.futures

def calculate_retro_metrics(fetcher, team_id: int, players_df: pd.DataFrame = None, progress_callback=None) -> Dict:
    """
    Calculate all retrospective metrics for a given team.
    Returns dictionary with captain_efficiency, bench_points, transfer_roi.
    """
    try:
        current_gw = fetcher.get_current_gameweek()
        history = fetcher.get_team_history(team_id)
        
        # Helper: Name lookup
        def get_name(pid):
            if players_df is not None and not players_df.empty:
                row = players_df[players_df['id'] == pid]
                if not row.empty:
                    return row.iloc[0]['web_name']
            return f"ID:{pid}"

        # 1. Bench Points
        bench_points_total = 0
        bench_history = []
        
        if 'current' in history:
            for event in history['current']:
                gw = event['event']
                pts = event.get('points_on_bench', 0)
                bench_points_total += pts
                bench_history.append({'gw': gw, 'points': pts})
        
        # 2. Captaincy Efficiency
        cap_points_total = 0
        max_cap_possible_total = 0
        cap_history = []
        
        if 'current' in history:
            completed_events = [e for e in history['current'] if e['event'] < current_gw]
            total_events = len(completed_events)
            
            # Parallel Fetching
            # We need picks and live data for each GW.
            # Let's map GW -> Future
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_picks = {executor.submit(fetcher.get_team_picks, team_id, e['event']): e['event'] for e in completed_events}
                future_live = {executor.submit(fetcher.get_event_live, e['event']): e['event'] for e in completed_events}
                
                # Wait for all
                picks_map = {}
                live_map = {}
                
                for future in concurrent.futures.as_completed(future_picks):
                    gw = future_picks[future]
                    try:
                        picks_map[gw] = future.result()
                    except Exception as e:
                        logger.warning(f"Failed to fetch picks for GW{gw}: {e}")
                        
                for future in concurrent.futures.as_completed(future_live):
                    gw = future_live[future]
                    try:
                        live_map[gw] = future.result()
                    except Exception as e:
                        logger.warning(f"Failed to fetch live data for GW{gw}: {e}")

            # Process Data
            for i, event in enumerate(completed_events):
                gw = event['event']
                
                if progress_callback:
                    progress_callback(i / total_events)
                
                try:
                    picks_data = picks_map.get(gw)
                    if not picks_data: continue
                        
                    picks = picks_data.get('picks', [])
                    if not picks: continue
                        
                    # Identify Captain and Vice
                    cap_entry = next((p for p in picks if p['is_captain']), None)
                    vice_entry = next((p for p in picks if p['is_vice_captain']), None)
                    
                    if not cap_entry: continue
                        
                    live_data = live_map.get(gw)
                    if not live_data: continue
                        
                    elements = {}
                    for e in live_data.get('elements', []):
                        if e.get('stats'):
                            elements[e['id']] = e['stats'].get('total_points', 0)
                    
                    # Calculate actual captain points
                    cap_pid = cap_entry['element']
                    vice_pid = vice_entry['element'] if vice_entry else 0
                    
                    cap_raw_points = elements.get(cap_pid, 0)
                    vice_raw_points = elements.get(vice_pid, 0)
                    
                    # Determine effective captain info
                    if cap_raw_points > 0:
                        real_cap_points = cap_raw_points * 2 
                        current_cap_pid = cap_pid
                    else:
                        if vice_entry:
                            real_cap_points = vice_raw_points * 2
                            current_cap_pid = vice_pid
                        else:
                            real_cap_points = 0
                            current_cap_pid = 0
                        
                    # Find max possible points in squad
                    squad_points = [elements.get(p['element'], 0) for p in picks]
                    max_raw = max(squad_points) if squad_points else 0
                    max_possible = max_raw * 2
                    
                    # Find who was max
                    max_pid = next((pid for pid, pts in elements.items() if pts == max_raw and pid in [p['element'] for p in picks]), 0) 
                    
                    cap_points_total += real_cap_points
                    max_cap_possible_total += max_possible
                    
                    cap_history.append({
                        'gw': gw,
                        'cap_pid': current_cap_pid,
                        'cap_name': get_name(current_cap_pid),
                        'cap_pts': real_cap_points,
                        'max_possible': max_possible,
                        'max_pid': max_pid,
                        'max_name': get_name(max_pid)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error calculating Cap efficiency GW{gw}: {e}")
                    continue

        efficiency_pct = (cap_points_total / max_cap_possible_total * 100) if max_cap_possible_total > 0 else 0.0
        
        return {
            'bench_points_total': bench_points_total,
            'bench_history': bench_history,
            'captain_efficiency': efficiency_pct,
            'captain_history': cap_history
        }

    except Exception as e:
        logger.error(f"Retrospective analysis failed: {e}")
        return {}
