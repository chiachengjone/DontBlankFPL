
import unittest
from unittest.mock import MagicMock
from utils.retro_analysis import calculate_retro_metrics

class TestRetroAnalysis(unittest.TestCase):
    
    def test_retro_metrics_bench_points(self):
        # Setup
        mock_fetcher = MagicMock()
        mock_fetcher.get_current_gameweek.return_value = 5
        mock_fetcher.get_team_history.return_value = {
            'current': [
                {'event': 1, 'points_on_bench': 5},
                {'event': 2, 'points_on_bench': 10},
                {'event': 3, 'points_on_bench': 0}
            ]
        }
        mock_fetcher.get_transfers.return_value = []
        
        # Run
        metrics = calculate_retro_metrics(mock_fetcher, 123, progress_callback=lambda x: None)
        
        # Assert
        self.assertEqual(metrics['bench_points_total'], 15)
        self.assertEqual(len(metrics['bench_history']), 3)



    def test_retro_metrics_captaincy(self):
        # Setup
        mock_fetcher = MagicMock()
        mock_fetcher.get_current_gameweek.return_value = 2
        mock_fetcher.get_team_history.return_value = {
            'current': [{'event': 1}]
        }
        mock_fetcher.get_transfers.return_value = []
        
        # GW 1 Picks: Cap (1), Vice (2)
        mock_fetcher.get_team_picks.return_value = {
            'picks': [
                {'element': 1, 'is_captain': True, 'is_vice_captain': False, 'multiplier': 2},
                {'element': 2, 'is_captain': False, 'is_vice_captain': True, 'multiplier': 1}
            ]
        }
        
        # GW 1 Live Data: Player 1 (5pts), Player 2 (10pts)
        # Cap scored 5 * 2 = 10.
        # Max possible was Player 2 (10) * 2 = 20.
        mock_fetcher.get_event_live.return_value = {
            'elements': [
                {'id': 1, 'stats': {'total_points': 5}},
                {'id': 2, 'stats': {'total_points': 10}}
            ]
        }
        
        # Run
        metrics = calculate_retro_metrics(mock_fetcher, 123)
        
        # Assert
        self.assertEqual(metrics['captain_efficiency'], 50.0) # 10 / 20 * 100
        self.assertEqual(len(metrics['captain_history']), 1)
        self.assertEqual(metrics['captain_history'][0]['cap_pts'], 10)
        self.assertEqual(metrics['captain_history'][0]['max_possible'], 20)

    def test_retro_metrics_names(self):
        # Setup
        mock_fetcher = MagicMock()
        mock_fetcher.get_current_gameweek.return_value = 2
        mock_fetcher.get_team_history.return_value = {'current': [{'event': 1}]}
        mock_fetcher.get_transfers.return_value = []
        
        mock_fetcher.get_team_picks.return_value = {
            'picks': [{'element': 1, 'is_captain': True, 'is_vice_captain': False, 'multiplier': 2}]
        }
        mock_fetcher.get_event_live.return_value = {
            'elements': [{'id': 1, 'stats': {'total_points': 5}}]
        }
        
        # Mock Players DF
        import pandas as pd
        players_df = pd.DataFrame([
            {'id': 1, 'web_name': 'Haaland'},
            {'id': 2, 'web_name': 'Salah'}
        ])
        
        # Run
        metrics = calculate_retro_metrics(mock_fetcher, 123, players_df)
        
        # Assert
        self.assertEqual(metrics['captain_history'][0]['cap_name'], 'Haaland')

if __name__ == '__main__':
    unittest.main()
