"""
Genetic Algorithm Optimizer - Evolutionary squad optimization.
Alternative to Integer Linear Programming using nature-inspired algorithms.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 100
    n_generations: int = 50
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elitism_rate: float = 0.1
    tournament_size: int = 5
    
    # Penalties
    invalid_squad_penalty: float = -1000.0
    budget_violation_penalty: float = -500.0
    position_violation_penalty: float = -300.0


@dataclass
class Individual:
    """Represents a single squad (chromosome)."""
    squad: List[int]  # 15 player IDs
    starting_xi: List[int]  # 11 starting players
    captain: int
    vice_captain: int
    fitness: float = 0.0
    
    # Validation flags
    is_valid: bool = False
    violations: List[str] = None
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class GeneticOptimizer:
    """
    Genetic Algorithm optimizer for FPL squad selection.
    
    Uses evolutionary principles:
    - Selection: Tournament selection
    - Crossover: Multi-point crossover
    - Mutation: Random player swaps
    - Elitism: Keep best individuals
    """
    
    def __init__(
        self,
        players_df: pd.DataFrame,
        config: Optional[GeneticConfig] = None
    ):
        self.players_df = players_df.copy()
        self.config = config or GeneticConfig()
        
        # Prepare player data
        self._prepare_data()
        
        # Population
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history = []
    
    def _prepare_data(self):
        """Prepare player data for optimization."""
        df = self.players_df.copy()
        
        # Ensure numeric
        df['now_cost'] = pd.to_numeric(df['now_cost'], errors='coerce').fillna(5.0)
        df['expected_points'] = pd.to_numeric(
            df.get('ep_next', df.get('expected_points', 2.0)),
            errors='coerce'
        ).fillna(2.0)
        
        # Position mapping
        position_map = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        df['position_id'] = df['position'].map(position_map).fillna(2)
        
        # Create lookup structures
        self.player_ids = df['id'].tolist()
        self.player_lookup = df.set_index('id').to_dict('index')
        
        # Group by position for easier squad construction
        self.players_by_position = {
            'GKP': df[df['position'] == 'GKP']['id'].tolist(),
            'DEF': df[df['position'] == 'DEF']['id'].tolist(),
            'MID': df[df['position'] == 'MID']['id'].tolist(),
            'FWD': df[df['position'] == 'FWD']['id'].tolist()
        }
        
        # Group by team for 3-per-team constraint
        self.players_by_team = df.groupby('team')['id'].apply(list).to_dict()
    
    def _create_random_valid_squad(self) -> List[int]:
        """Create a random but valid 15-player squad."""
        squad = []
        
        # Ensure position constraints: 2 GKP, 5 DEF, 5 MID, 3 FWD
        constraints = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
        for pos, count in constraints.items():
            available = self.players_by_position[pos].copy()
            
            # Select random players from position
            selected = np.random.choice(available, size=count, replace=False).tolist()
            squad.extend(selected)
        
        # Shuffle to randomize order
        np.random.shuffle(squad)
        
        return squad
    
    def _is_squad_valid(self, squad: List[int]) -> Tuple[bool, List[str]]:
        """
        Validate squad against FPL rules.
        Returns (is_valid, list_of_violations)
        """
        violations = []
        
        # Check squad size
        if len(squad) != 15:
            violations.append(f"Squad size must be 15, got {len(squad)}")
        
        # Check unique players
        if len(set(squad)) != len(squad):
            violations.append("Duplicate players in squad")
        
        # Check position constraints
        positions = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        for pid in squad:
            player = self.player_lookup.get(pid, {})
            pos = player.get('position', 'Unknown')
            if pos in positions:
                positions[pos] += 1
        
        if positions['GKP'] != 2:
            violations.append(f"Must have 2 GKP, got {positions['GKP']}")
        if positions['DEF'] != 5:
            violations.append(f"Must have 5 DEF, got {positions['DEF']}")
        if positions['MID'] != 5:
            violations.append(f"Must have 5 MID, got {positions['MID']}")
        if positions['FWD'] != 3:
            violations.append(f"Must have 3 FWD, got {positions['FWD']}")
        
        # Check budget
        total_cost = sum(self.player_lookup.get(pid, {}).get('now_cost', 10) for pid in squad)
        if total_cost > 100.0:
            violations.append(f"Budget exceeded: {total_cost:.1f}m > 100.0m")
        
        # Check max 3 per team
        team_counts = {}
        for pid in squad:
            team = self.player_lookup.get(pid, {}).get('team', 0)
            team_counts[team] = team_counts.get(team, 0) + 1
        
        for team, count in team_counts.items():
            if count > 3:
                violations.append(f"Max 3 players per team, team {team} has {count}")
        
        return len(violations) == 0, violations
    
    def _select_starting_xi(self, squad: List[int]) -> List[int]:
        """Select best starting 11 from squad based on expected points."""
        # Get player data
        squad_data = [(pid, self.player_lookup.get(pid, {})) for pid in squad]
        
        # Sort by expected points
        squad_data.sort(key=lambda x: x[1].get('expected_points', 0), reverse=True)
        
        # Build starting XI with formation constraints
        # 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD
        starting = []
        positions = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        max_starting = {'GKP': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
        min_starting = {'GKP': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
        
        # First pass: ensure minimums
        for pid, player in squad_data:
            pos = player.get('position', 'MID')
            if positions[pos] < min_starting[pos]:
                starting.append(pid)
                positions[pos] += 1
        
        # Second pass: fill remaining spots with best players
        for pid, player in squad_data:
            if len(starting) >= 11:
                break
            if pid not in starting:
                pos = player.get('position', 'MID')
                if positions[pos] < max_starting[pos]:
                    starting.append(pid)
                    positions[pos] += 1
        
        return starting[:11]
    
    def _fitness_function(self, individual: Individual) -> float:
        """
        Calculate fitness score for an individual.
        Higher is better.
        """
        squad = individual.squad
        starting_xi = individual.starting_xi
        captain = individual.captain
        
        # Validate squad
        is_valid, violations = self._is_squad_valid(squad)
        individual.is_valid = is_valid
        individual.violations = violations
        
        if not is_valid:
            # Heavy penalty for invalid squads
            penalty = self.config.invalid_squad_penalty * len(violations)
            return penalty
        
        # Calculate expected points
        total_ep = 0.0
        
        for pid in starting_xi:
            player = self.player_lookup.get(pid, {})
            ep = player.get('expected_points', 2.0)
            
            # Captain gets 1.25x multiplier
            if pid == captain:
                ep *= 1.25
            
            total_ep += ep
        
        # Bonus for team balance (diversity)
        team_counts = {}
        for pid in squad:
            team = self.player_lookup.get(pid, {}).get('team', 0)
            team_counts[team] = team_counts.get(team, 0) + 1
        
        diversity_bonus = len(team_counts) * 0.5  # More teams = more diverse
        
        # Bonus for budget efficiency
        total_cost = sum(self.player_lookup.get(pid, {}).get('now_cost', 5) for pid in squad)
        remaining_budget = 100.0 - total_cost
        budget_bonus = remaining_budget * 0.1  # Small bonus for keeping bank
        
        fitness = total_ep + diversity_bonus + budget_bonus
        
        return fitness
    
    def initialize_population(self):
        """Create initial population of random valid squads."""
        self.population = []
        
        attempts = 0
        max_attempts = self.config.population_size * 10
        
        while len(self.population) < self.config.population_size and attempts < max_attempts:
            attempts += 1
            
            try:
                squad = self._create_random_valid_squad()
                starting_xi = self._select_starting_xi(squad)
                captain = starting_xi[0] if starting_xi else squad[0]
                vice = starting_xi[1] if len(starting_xi) > 1 else squad[1]
                
                individual = Individual(
                    squad=squad,
                    starting_xi=starting_xi,
                    captain=captain,
                    vice_captain=vice
                )
                
                individual.fitness = self._fitness_function(individual)
                
                if individual.is_valid or len(self.population) < self.config.population_size // 2:
                    self.population.append(individual)
            
            except Exception as e:
                continue
        
        # If we couldn't generate enough, duplicate best ones
        while len(self.population) < self.config.population_size:
            if self.population:
                self.population.append(deepcopy(self.population[0]))
            else:
                break
    
    def selection(self) -> Individual:
        """Tournament selection."""
        tournament = np.random.choice(
            self.population,
            size=min(self.config.tournament_size, len(self.population)),
            replace=False
        )
        return max(tournament, key=lambda ind: ind.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Multi-point crossover between two parents.
        Preserves position constraints.
        """
        if np.random.random() > self.config.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        # Crossover by position to maintain validity
        child1_squad = []
        child2_squad = []
        
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            p1_pos = [pid for pid in parent1.squad if self.player_lookup.get(pid, {}).get('position') == pos]
            p2_pos = [pid for pid in parent2.squad if self.player_lookup.get(pid, {}).get('position') == pos]
            
            # Randomly split position players
            split = len(p1_pos) // 2
            
            child1_squad.extend(p1_pos[:split] + p2_pos[split:])
            child2_squad.extend(p2_pos[:split] + p1_pos[split:])
        
        # Remove duplicates (take first occurrence)
        child1_squad = list(dict.fromkeys(child1_squad))
        child2_squad = list(dict.fromkeys(child2_squad))
        
        # If squad incomplete, fill with random valid players
        while len(child1_squad) < 15:
            random_pid = np.random.choice(self.player_ids)
            if random_pid not in child1_squad:
                child1_squad.append(random_pid)
        
        while len(child2_squad) < 15:
            random_pid = np.random.choice(self.player_ids)
            if random_pid not in child2_squad:
                child2_squad.append(random_pid)
        
        # Create children
        child1 = Individual(
            squad=child1_squad[:15],
            starting_xi=self._select_starting_xi(child1_squad[:15]),
            captain=0,
            vice_captain=0
        )
        child1.captain = child1.starting_xi[0] if child1.starting_xi else child1_squad[0]
        child1.vice_captain = child1.starting_xi[1] if len(child1.starting_xi) > 1 else child1_squad[1]
        
        child2 = Individual(
            squad=child2_squad[:15],
            starting_xi=self._select_starting_xi(child2_squad[:15]),
            captain=0,
            vice_captain=0
        )
        child2.captain = child2.starting_xi[0] if child2.starting_xi else child2_squad[0]
        child2.vice_captain = child2.starting_xi[1] if len(child2.starting_xi) > 1 else child2_squad[1]
        
        return child1, child2
    
    def mutate(self, individual: Individual):
        """
        Mutate individual by replacing random players.
        Maintains position constraints.
        """
        if np.random.random() > self.config.mutation_rate:
            return
        
        # Mutate 1-3 players
        n_mutations = np.random.randint(1, 4)
        
        for _ in range(n_mutations):
            # Select random player to replace
            idx = np.random.randint(0, len(individual.squad))
            old_pid = individual.squad[idx]
            old_player = self.player_lookup.get(old_pid, {})
            old_pos = old_player.get('position', 'MID')
            
            # Find replacement from same position
            available = [
                pid for pid in self.players_by_position[old_pos]
                if pid not in individual.squad
            ]
            
            if available:
                new_pid = np.random.choice(available)
                individual.squad[idx] = new_pid
        
        # Reselect starting XI and captain
        individual.starting_xi = self._select_starting_xi(individual.squad)
        individual.captain = individual.starting_xi[0] if individual.starting_xi else individual.squad[0]
        individual.vice_captain = individual.starting_xi[1] if len(individual.starting_xi) > 1 else individual.squad[1]
    
    def evolve(self) -> Individual:
        """
        Run genetic algorithm optimization.
        Returns best individual found.
        """
        print(f"Initializing population of {self.config.population_size}...")
        self.initialize_population()
        
        print(f"Evolving for {self.config.n_generations} generations...")
        
        for generation in range(self.config.n_generations):
            # Sort by fitness
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            
            # Track best
            best_fitness = self.population[0].fitness
            self.history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean([ind.fitness for ind in self.population]),
                'valid_count': sum(1 for ind in self.population if ind.is_valid)
            })
            
            if generation % 10 == 0:
                valid_pct = (self.history[-1]['valid_count'] / len(self.population)) * 100
                print(f"Gen {generation}: Best={best_fitness:.2f}, Avg={self.history[-1]['avg_fitness']:.2f}, Valid={valid_pct:.0f}%")
            
            # Elitism: keep top individuals
            n_elite = int(self.config.population_size * self.config.elitism_rate)
            new_population = self.population[:n_elite]
            
            # Generate new population
            while len(new_population) < self.config.population_size:
                # Selection
                parent1 = self.selection()
                parent2 = self.selection()
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                self.mutate(child1)
                self.mutate(child2)
                
                # Evaluate
                child1.fitness = self._fitness_function(child1)
                child2.fitness = self._fitness_function(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.config.population_size]
        
        # Final sort
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        self.best_individual = self.population[0]
        
        print(f"\nOptimization complete!")
        print(f"Best fitness: {self.best_individual.fitness:.2f}")
        print(f"Valid: {self.best_individual.is_valid}")
        
        return self.best_individual
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get convergence history as DataFrame."""
        return pd.DataFrame(self.history)


def create_genetic_optimizer(
    players_df: pd.DataFrame,
    population_size: int = 100,
    n_generations: int = 50
) -> GeneticOptimizer:
    """Factory function to create genetic optimizer."""
    config = GeneticConfig(
        population_size=population_size,
        n_generations=n_generations
    )
    return GeneticOptimizer(players_df, config)
