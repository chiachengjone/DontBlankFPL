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
    """
    
    def __init__(
        self,
        players_df: pd.DataFrame,
        config: Optional[GeneticConfig] = None,
        locked_ids: List[int] = None,
        excluded_ids: List[int] = None
    ):
        self.players_df = players_df.copy()
        self.config = config or GeneticConfig()
        self.locked_ids = set(locked_ids) if locked_ids else set()
        self.excluded_ids = set(excluded_ids) if excluded_ids else set()
        
        # Prepare player data
        self._prepare_data()
        
        # Population
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history = []
    
    def _prepare_data(self):
        """Prepare player data for optimization."""
        # Filter out excluded players immediately if possible, 
        # but we need to keep them in lookup just in case? 
        # Actually safer to just filter them out from "available" lists.
        
        df = self.players_df.copy()
        
        # Ensure numeric
        df['now_cost'] = pd.to_numeric(df['now_cost'], errors='coerce').fillna(5.0)
        df['expected_points'] = pd.to_numeric(
            df.get('consensus_ep', df.get('expected_points', df.get('ep_next', 2.0))),
            errors='coerce'
        ).fillna(2.0)
        
        # Position mapping
        position_map = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        df['position_id'] = df['position'].map(position_map).fillna(2)
        
        # Create lookup structures
        self.player_ids = df['id'].tolist()
        self.player_lookup = df.set_index('id').to_dict('index')
        
        # Group by position for easier squad construction
        # Exclude 'excluded_ids' from these pools
        self.players_by_position = {
            'GKP': [pid for pid in df[df['position'] == 'GKP']['id'].tolist() if pid not in self.excluded_ids],
            'DEF': [pid for pid in df[df['position'] == 'DEF']['id'].tolist() if pid not in self.excluded_ids],
            'MID': [pid for pid in df[df['position'] == 'MID']['id'].tolist() if pid not in self.excluded_ids],
            'FWD': [pid for pid in df[df['position'] == 'FWD']['id'].tolist() if pid not in self.excluded_ids]
        }
    
    def _create_random_valid_squad(self) -> List[int]:
        """Create a random but valid 15-player squad, respecting locks."""
        squad = list(self.locked_ids)
        
        # Track current state
        positions = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        for pid in squad:
            pos = self.player_lookup.get(pid, {}).get('position', 'MID')
            if pos in positions:
                positions[pos] += 1
        
        # Constraints
        target_counts = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
        for pos, count in target_counts.items():
            needed = count - positions.get(pos, 0)
            if needed <= 0:
                continue
                
            available = [p for p in self.players_by_position[pos] if p not in squad]
            
            # Select random players from position
            if len(available) >= needed:
                selected = np.random.choice(available, size=needed, replace=False).tolist()
                squad.extend(selected)
            else:
                # Not enough players to satisfy constraint (edge case)
                squad.extend(available)
        
        # Shuffle keys, not the list itself if we want to keep structure? 
        # Actually shuffle is fine, structure is just list of IDs
        np.random.shuffle(squad)
        
        # Trim if we somehow have > 15 (e.g. too many locked)
        # But we assume locks are < 15.
        
        return squad[:15]

    def _is_squad_valid(self, squad: List[int]) -> Tuple[bool, List[str]]:
        """Check if squad satisfies all FPL constraints."""
        violations = []
        
        # 1. Size check
        if len(squad) != 15:
            violations.append(f"Invalid size: {len(squad)}")
            return False, violations
            
        # 2. Budget check
        total_cost = sum(self.player_lookup[pid]['now_cost'] for pid in squad)
        if total_cost > 100.0:
            violations.append(f"Budget exceeded: {total_cost:.1f}")
            
        # 3. Team constraints (max 3 per team)
        teams = [self.player_lookup[pid]['team'] for pid in squad]
        for team in set(teams):
            if teams.count(team) > 3:
                violations.append(f"Too many players from team {team}")
                
        # 4. Position constraints
        positions = [self.player_lookup[pid]['position'] for pid in squad]
        if positions.count('GKP') != 2: violations.append("Must have 2 GKPs")
        if positions.count('DEF') != 5: violations.append("Must have 5 DEFs")
        if positions.count('MID') != 5: violations.append("Must have 5 MIDs")
        if positions.count('FWD') != 3: violations.append("Must have 3 FWDs")
        
        # 5. Lock/Exclude constraints
        if not self.locked_ids.issubset(set(squad)):
             violations.append("Missing locked players")
        
        if self.excluded_ids.intersection(set(squad)):
            violations.append("Contains excluded players")

        return len(violations) == 0, violations

    def _select_starting_xi(self, squad: List[int]) -> List[int]:
        """Select best valid starting XI from squad."""
        # Split by position
        gkp = [p for p in squad if self.player_lookup[p]['position'] == 'GKP']
        def_p = [p for p in squad if self.player_lookup[p]['position'] == 'DEF']
        mid = [p for p in squad if self.player_lookup[p]['position'] == 'MID']
        fwd = [p for p in squad if self.player_lookup[p]['position'] == 'FWD']
        
        # Sort by xP
        gkp.sort(key=lambda p: self.player_lookup[p]['expected_points'], reverse=True)
        def_p.sort(key=lambda p: self.player_lookup[p]['expected_points'], reverse=True)
        mid.sort(key=lambda p: self.player_lookup[p]['expected_points'], reverse=True)
        fwd.sort(key=lambda p: self.player_lookup[p]['expected_points'], reverse=True)
        
        # Base formation: 1 GK, 3 DEF, 2 MID, 1 FWD (Minimums)
        xi = [gkp[0]] + def_p[:3] + mid[:2] + fwd[:1]
        
        # Remaining pool (sorted by xP)
        remaining = def_p[3:] + mid[2:] + fwd[1:]
        remaining.sort(key=lambda p: self.player_lookup[p]['expected_points'], reverse=True)
        
        # Fill remaining 4 spots
        xi.extend(remaining[:4])
        
        # Check if valid formation (min 3 DEF, 1 FWD is guaranteed by base formation)
        # Max limits implied by squad size? No, need to ensure we don't play e.g. 6 MIDs (impossible since only 5 in squad)
        # With 15-man squad constraints (5 DEF, 5 MID), we can't violate max formation rules usually
        # (Max 5 DEF, Max 5 MID, Max 3 FWD). 
        # Wait, max 5 DEF is standard. Max 5 MID is standard. Max 3 FWD is standard.
        # So essentially any combination of 10 outfielders from 5-5-3 set is valid?
        # Formations: 3-5-2, 3-4-3, 4-5-1, 4-4-2, 4-3-3, 5-4-1, 5-3-2, 5-2-3 (Total 10)
        # Yes.
        
        return xi

    def _fitness_function(self, individual: Individual) -> float:
        """Calculate fitness of an individual."""
        # Validity check
        is_valid, violations = self._is_squad_valid(individual.squad)
        individual.is_valid = is_valid
        
        if not is_valid:
            return self.config.invalid_squad_penalty - (len(violations) * 50)
            
        # Base Points: Starting XI xP
        xi_xp = sum(self.player_lookup[pid]['expected_points'] for pid in individual.starting_xi)
        
        # Captain Bonus (doubled points for captain)
        cap_pid = individual.captain
        cap_xp = self.player_lookup[cap_pid]['expected_points']
        
        # Bench weighting (10% of bench points)
        bench = set(individual.squad) - set(individual.starting_xi)
        bench_xp = sum(self.player_lookup[pid]['expected_points'] for pid in bench)
        
        # Total
        score = xi_xp + cap_xp + (bench_xp * 0.1)
        
        # Budget utilization bonus (optional, maybe not needed if xP is king)
        
        return score

    def initialize_population(self):
        """Initialize random population."""
        self.population = []
        failure_count = 0
        while len(self.population) < self.config.population_size:
            try:
                squad = self._create_random_valid_squad()
                xi = self._select_starting_xi(squad)
                cap = xi[0] if xi else squad[0]
                vc = xi[1] if len(xi) > 1 else squad[1]
                
                ind = Individual(squad, xi, cap, vc)
                ind.fitness = self._fitness_function(ind)
                
                self.population.append(ind)
            except Exception:
                failure_count += 1
                if failure_count > 1000:
                    break

    def selection(self) -> Individual:
        """Tournament selection."""
        candidates = np.random.choice(self.population, size=self.config.tournament_size, replace=False)
        return max(candidates, key=lambda ind: ind.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Crossover two parents to create two children."""
        if np.random.random() > self.config.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
            
        # Merge squads to pool
        pool = list(set(parent1.squad + parent2.squad))
        
        # Try to form 2 valid split squads from the pool
        # This is hard to ensure validity directly. 
        # Standard approach: Take half from P1, half from P2, fill rest.
        
        child1_squad = self._repair_squad(parent1.squad[:7] + parent2.squad[7:])
        child2_squad = self._repair_squad(parent2.squad[:7] + parent1.squad[7:])
        
        c1 = Individual(child1_squad, self._select_starting_xi(child1_squad), child1_squad[0], child1_squad[1])
        c2 = Individual(child2_squad, self._select_starting_xi(child2_squad), child2_squad[0], child2_squad[1])
        
        return c1, c2

    def _repair_squad(self, squad_in: List[int]) -> List[int]:
        """Repair a squad to valid state."""
        # 1. Deduplicate
        squad = list(set(squad_in))
        
        # 2. Ensure locks
        for pid in self.locked_ids:
            if pid not in squad:
                squad.append(pid)
        
        # 3. Categorize
        pos_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        final_squad = []
        
        # Add locked players first
        locked = list(self.locked_ids)
        for pid in locked:
            pos = self.player_lookup.get(pid, {}).get('position', 'MID')
            if pos_counts[pos] < {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}[pos]:
                final_squad.append(pid)
                pos_counts[pos] += 1
                
        # Fill remaining slots with players from input squad
        for pid in squad:
            if pid in final_squad: continue
            pos = self.player_lookup.get(pid, {}).get('position', 'MID')
            limit = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}[pos]
            if pos_counts[pos] < limit:
                final_squad.append(pid)
                pos_counts[pos] += 1
                
        # If still missing, fill with free agents
        target = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        for pos, count in target.items():
            needed = count - pos_counts[pos]
            if needed > 0:
                available = [
                    p for p in self.players_by_position[pos] 
                    if p not in final_squad
                ]
                # Sort by xP to pick good fillers
                # available.sort(key=lambda p: self.player_lookup[p]['expected_points'], reverse=True)
                # Randomize slightly
                if len(available) >= needed:
                    selected = np.random.choice(available, size=needed, replace=False).tolist()
                    final_squad.extend(selected)
                    
        return final_squad[:15]
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get history as DataFrame."""
        return pd.DataFrame(self.history)

    def mutate(self, individual: Individual):
        """
        Mutate individual by replacing random players.
        Maintains position constraints and respects locked players.
        """
        if np.random.random() > self.config.mutation_rate:
            return
        
        # Mutate 1-3 players
        n_mutations = np.random.randint(1, 4)
        
        # Candidates for mutation (not locked)
        mutable_indices = [i for i, pid in enumerate(individual.squad) if pid not in self.locked_ids]
        
        if not mutable_indices:
            return

        for _ in range(n_mutations):
            if not mutable_indices:
                break
                
            # Select random player to replace
            idx = np.random.choice(mutable_indices)
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
            
            # Remove from mutable indices to avoid double mutation in one pass? 
            # Or allow it. Let's allowing it is fine but simpler loop structure ok.

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
        
        stagnation_counter = 0
        best_fitness_history = -99999.0
        
        for generation in range(self.config.n_generations):
            # Sort by fitness
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            
            # Track best
            current_best = self.population[0].fitness
            
            # Stagnation check
            if abs(current_best - best_fitness_history) < 0.1:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                best_fitness_history = current_best
                
            # Diversity Injection (Cataclysm)
            if stagnation_counter > 10:
                # print("Stagnation detected! Injecting diversity...")
                # Replace bottom 50% with new random squads
                n_replace = self.config.population_size // 2
                for i in range(self.config.population_size - n_replace, self.config.population_size):
                    try:
                        squad = self._create_random_valid_squad()
                        xi = self._select_starting_xi(squad)
                        cap = xi[0] if xi else squad[0]
                        vc = xi[1] if len(xi) > 1 else squad[1]
                        ind = Individual(squad, xi, cap, vc)
                        ind.fitness = self._fitness_function(ind)
                        self.population[i] = ind
                    except:
                        pass
                stagnation_counter = 0
            
            self.history.append({
                'generation': generation,
                'best_fitness': current_best,
                'avg_fitness': np.mean([ind.fitness for ind in self.population]),
                'valid_count': sum(1 for ind in self.population if ind.is_valid)
            })
            
            if generation % 10 == 0:
                valid_pct = (self.history[-1]['valid_count'] / len(self.population)) * 100
                print(f"Gen {generation}: Best={current_best:.2f}, Avg={self.history[-1]['avg_fitness']:.2f}, Valid={valid_pct:.0f}%")
            
            # Elitism
            n_elite = int(self.config.population_size * self.config.elitism_rate)
            new_population = self.population[:n_elite]
            
            # Generate new population
            while len(new_population) < self.config.population_size:
                parent1 = self.selection()
                parent2 = self.selection()
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                child1.fitness = self._fitness_function(child1)
                child2.fitness = self._fitness_function(child2)
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.config.population_size]
        
        # Final sort
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        self.best_individual = self.population[0]
        
        return self.best_individual

    # ... [Keep get_optimization_history] ...

def create_genetic_optimizer(
    players_df: pd.DataFrame,
    population_size: int = 100,
    n_generations: int = 50,
    locked_ids: List[int] = None,
    excluded_ids: List[int] = None
) -> GeneticOptimizer:
    """Factory function to create genetic optimizer."""
    config = GeneticConfig(
        population_size=population_size,
        n_generations=n_generations
    )
    return GeneticOptimizer(players_df, config, locked_ids, excluded_ids)
