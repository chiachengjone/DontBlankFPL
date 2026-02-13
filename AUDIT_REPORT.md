# FPL Strategy Engine - Comprehensive Audit Report
*Generated: February 14, 2026*

---

## 🔴 CRITICAL ISSUES

### 1. **Poisson EP Integration Has Fatal Flaw**
**Location:** `poisson_ep.py` line 635-670  
**Issue:** The new Poisson model uses **FDR as proxy for opponent xG/xGA**, which is circular reasoning and inaccurate.

```python
# Current (WRONG):
fdr = fx.get('team_h_difficulty', 3)
xga_mult = 0.5 + (fdr / 5) * 1.0  # FDR is subjective FPL rating, not actual xGA
```

**Impact:** EP calculations are based on FPL's own subjective difficulty ratings, not real defensive stats. This defeats the purpose of using Understat data.

**Fix Required:**
- Fetch actual team-level xGA/xG data from Understat team stats endpoint
- Build `team_xGA_per90` and `team_xG_per90` from historical match data
- Use real opponent defensive weakness instead of FDR proxy

---

### 2. **CBIT Calculation is a Placeholder**
**Location:** `fpl_api.py` lines 286-299  
**Issue:** CBIT (Clearances, Blocks, Interceptions, Tackles) is estimated from **clean sheets**, not actual defensive actions.

```python
df['estimated_cbit'] = np.where(
    is_defender,
    df['clean_sheets'].fillna(0) * 2.5,  # ❌ Not real CBIT data
    0.0
)
```

**Impact:** 
- Havertz showing 1.00 CBIT is because the formula applies to ALL players initially
- No actual Opta defensive action data is being used
- CBIT propensity is meaningless for evaluating defensive contributions

**Fix Required:**
- Integrate with Opta or FBref API for real defensive action counts
- Or scrape from FPL's own player detail pages which show tackles/interceptions
- Until then, set CBIT to 0 for non-GKP/DEF to avoid misleading data

---

### 3. **Understat Enrichment Fails Silently**
**Location:** `fpl_api.py` lines 590-594  
**Issue:** If Understat fetch fails, the app continues with zero values.

```python
try:
    from understat_api import enrich_with_understat
    df = enrich_with_understat(df)
except Exception:
    pass  # ❌ Silent failure - user has no idea EP is degraded
```

**Impact:** 
- Users don't know when advanced EP model isn't working
- Poisson model still runs with bad/zero xG data
- No logging of why enrichment failed

**Fix Required:**
- Add visual indicator in UI showing "Understat: ✅ Active" or "⚠️ Unavailable"
- Log the actual exception for debugging
- Show warning banner when using fallback FPL data

---

### 4. **Double Gameweek Handling is Broken**
**Location:** `poisson_ep.py` lines 663-667  
**Issue:** DGW is handled by **simple multiplication**, not iterating fixtures.

```python
if fx_count > 1:
    xp_result['xp_total'] *= fx_count  # ❌ Wrong - assumes identical fixtures
    xp_result['xp_goals'] *= fx_count
```

**Impact:**
- DGW against City + Brentford is treated same as two games vs Brentford
- Massively overestimates DGW value for bad fixtures
- Should iterate each fixture independently

**Fix Required:**
- Pass full fixture list to `calculate_gameweek_xp()`
- Calculate λ for each opponent separately
- Sum independent Poisson outcomes per fixture

---

### 5. **Starts Percentage Calculation is Nonsensical**
**Location:** `fpl_api.py` lines 271-278  
**Issue:** `starts_pct` uses `total_points` as denominator estimate?

```python
total_gws = pd.to_numeric(df.get('total_points', 0), errors='coerce').fillna(0)
df['starts_pct'] = np.where(
    df['starts'] > 0,
    np.minimum(df['starts'] / max(1, df['starts'].max() * 0.9), 1.0),  # ❌ What is this?
    0.1
)
```

**Impact:** Appearance probability in Poisson model is wildly inaccurate.

**Fix Required:**
- Use actual: `starts / total_gameweeks_played` where total GWs is from bootstrap
- Or better: `(minutes / 90) / team_games_played`

---

## 🟠 MAJOR WEAKNESSES

### 6. **No Real Team-Level xGA Data**
**Location:** Entire Poisson model  
**Issue:** The model claims to use "opponent xGA / league avg" but has no source for this.

**Current State:**
- `LEAGUE_AVG_XGA_PER_MATCH = 1.35` is hardcoded
- No team-specific defensive stats are fetched
- FDR is used as proxy (circular logic)

**Fix Required:**
- Add `fetch_understat_team_stats()` function
- Scrape team-level xG for/against from Understat team pages
- Calculate league average dynamically
- Build lookup: `team_id -> {xG_for_per90, xGA_per90}`

---

### 7. **Threat Momentum Uses Wrong Direction Formula**
**Location:** `fpl_api.py` lines 686-696  
**Issue:** Compares form-weighted xG to season average, but the math is wrong.

```python
season_avg_xgi = (df['us_xG_per90'].fillna(0) + df['us_xA_per90'].fillna(0))
df['threat_direction'] = np.where(
    season_avg_xgi > 0,
    (df['threat_momentum'] - season_avg_xgi) / season_avg_xgi.clip(lower=0.01),  # ❌
    0
)
```

**Problem:** 
- `threat_momentum` is form-weighted, `season_avg_xgi` is not
- Comparing weighted to unweighted is meaningless
- Should compare recent 5 games to prior 10 games

**Fix Required:**
- Calculate rolling average xGI for last 5 games
- Compare to rolling average for previous 10 games
- That gives true "heating up" vs "cooling off" signal

---

### 8. **Matchup Quality Formula is Incomplete**
**Location:** `fpl_api.py` lines 714-728  
**Issue:** Uses `fixture_ease` (derived from FDR) instead of opponent xGA.

```python
opp_weakness = 0.5 + df['fixture_ease'].clip(0, 1)  # ❌ FDR again
df['matchup_quality'] = player_threat * opp_weakness
```

**Should Be:**
```python
# Get opponent's actual xGA from team stats
opp_xga = opponent_stats[opponent_id]['xGA_per90']
opp_weakness = opp_xga / LEAGUE_AVG_XGA
df['matchup_quality'] = player_xg_per90 * opp_weakness
```

---

### 9. **Variance-Adjusted Differential Logic is Inverted**
**Location:** `fpl_api.py` lines 652-658  
**Issue:** Negative bonus for underperformance is backwards.

```python
variance_bonus = (-df['us_goal_overperf'].fillna(0) * 0.5) + ...
```

**Current Logic:**
- Player has scored 5 goals, xG is 8 → overperf = -3 → variance_bonus = +1.5 ✅
- Player has scored 8 goals, xG is 5 → overperf = +3 → variance_bonus = -1.5 ❌

**Problem:** Players outperforming xG are being PENALIZED as differentials, when they're actually hot.

**Fix:** Remove the negative sign OR flip the interpretation.

---

### 10. **No Injury Impact on Expected Points**
**Location:** `poisson_ep.py` & `fpl_api.py`  
**Issue:** Injury status affects appearance probability but not the main EP calculation flow.

**Current:**
- Poisson model has `injury_status` parameter but it's not passed from FPL API data
- Main EP calculation ignores `chance_of_playing_next_round`

**Fix Required:**
- Add `df['injury_status']` from bootstrap data (status field)
- Multiply EP by injury probability: `ep * (chance_of_playing / 100)`
- Show injury indicators in all tables

---

## 🟡 MODERATE ISSUES

### 11. **No Rotation Risk Modeling**
**Issue:** Template squads include benched players with zero adjustment.

**Fix:** Add `minutes_volatility` metric:
```python
df['rotation_risk'] = 1 - (df['minutes_per_game'] / 90).clip(0, 1)
df['adjusted_ep'] = df['expected_points'] * (1 - df['rotation_risk'] * 0.3)
```

---

### 12. **Captain Multiplier is Wrong**
**Location:** Multiple files  
**Issue:** `CAPTAIN_MULTIPLIER = 1.25` should be `2.0` for standard captaincy.

The 1.25x is for Vice-Captain fallback or some 2025/26 rule change that isn't documented.

---

### 13. **Bonus Points Model Lacks Data**
**Location:** `poisson_ep.py` lines 302-345  
**Issue:** Claims to use "haul probability" for bonus but has no historical bonus data.

**Reality:** 
- No BPS (Bonus Points System) thresholds are modeled
- No position-specific bonus rates from historical data
- Just arbitrary multipliers

**Fix:** Scrape FPL's bonus point data or use historical correlation:
- 2+ goal involvements → 70% chance of 2+ bonus (for attackers)
- CS + attacking return → 80% chance bonus (for defenders)

---

### 14. **Optimizer Doesn't Use Poisson Variance**
**Location:** `optimizer.py`  
**Issue:** Squad optimization uses point estimates, not uncertainty.

**Better Approach:**
- Use Poisson model's probability distributions
- Optimize for `mean - k*std` (downside-adjusted returns)
- Or maximize probability of top 10k finish (requires Monte Carlo integration)

---

### 15. **No Chip Strategy**
**Issue:** Bench Boost, Triple Captain, Wildcard not modeled in planning horizon.

**Fix:** Add chip simulation in multi-week planner:
- BB: Add bench EP to total
- TC: 3x best captain expected
- FH: Unlimited transfers for 1 GW

---

### 16. **Fixture Ease Uses Exponential Decay Incorrectly**
**Location:** `fpl_api.py` lines 546-560  
**Issue:** Uses `0.95 ** i` but doesn't explain why 5% decay per week.

**Better:** 
- GW+1: 100% weight
- GW+2: 70% weight (harder to predict)
- GW+3: 50% weight
- GW+4+: 30% weight

Linear decay makes more sense than exponential.

---

## 🟢 MINOR ISSUES / UX IMPROVEMENTS

### 17. **No Data Freshness Indicator**
Users can't tell if data is 5 minutes old or 5 hours old. Add timestamp to status bar.

---

### 18. **Search in Heatmap is Slow**
The player search implemented in analytics tab does full re-render. Use `st.experimental_fragment` for performance.

---

### 19. **No Export to CSV**
Users can't export optimized squads or differential picks to CSV.

---

### 20. **Error Messages are Generic**
"Error loading players: 'expected_points'" tells user nothing. Show full traceback in expander.

---

### 21. **No Dark Mode Toggle**
App is always dark. Some users prefer light mode.

---

### 22. **Trends Use Static Data**
"Ownership Trends" uses current ownership %, not historical trend. Need time-series ownership data.

---

### 23. **No Mini-League Comparison**
Rival Scout only works with single team ID. Should support mini-league batch analysis.

---

### 24. **Genetic Algorithm Output Not Human-Readable**
Shows squad IDs but not player names in chromosome display.

---

### 25. **No Mobile Optimization**
Charts are too wide for mobile. Use responsive breakpoints.

---

## 🏗️ ARCHITECTURAL ISSUES

### 26. **Tight Coupling Between Tabs and Data Pipeline**
Each tab imports processor directly. Should use dependency injection or context.

---

### 27. **No Unit Tests**
Zero test coverage. Critical calculations have no validation.

**Recommendation:** Add `tests/` folder with:
- `test_poisson_ep.py` - validate λ calculations
- `test_fpl_api.py` - mock API responses
- `test_optimizer.py` - verify constraint satisfaction

---

### 28. **No Caching Strategy for Understat**
Understat data is fetched every 5 minutes but rarely changes. Cache for 1 hour minimum.

---

### 29. **Pandas PerformanceWarnings**
`.iterrows()` is used extensively (slow). Use `.apply()` or vectorized operations.

---

### 30. **No Database**
All data is in-memory. Loses historical tracking, can't do time-series analysis.

**Recommendation:** Add SQLite or PostgreSQL for:
- Historical EP predictions vs actual
- Ownership changes over time
- Transfer histories

---

## 📊 STATISTICAL METHODOLOGY CONCERNS

### 31. **No Backtesting Validation**
The Poisson model has never been tested against historical data. We don't know if it's actually better than FPL's model.

**Fix:** Implement backtesting:
1. Load 2024/25 season data
2. Calculate Poisson EP for each GW
3. Compare to actual points scored
4. Measure MAE (Mean Absolute Error) vs FPL baseline

---

### 32. **Correlation Not Considered**
Poisson model treats goals/assists as independent. But:
- Team correlation: If Saka scores, Havertz likely assisted
- Opponent correlation: If City concedes, all their defenders lose CS

**Advanced Fix:** Use Copula models or multivariate distributions.

---

### 33. **No Adjustment for Home/Away Splits**
Home advantage is modeled globally (15%) but should be team-specific:
- Liverpool at Anfield: 25% boost
- Bournemouth at home: 5% boost

---

### 34. **Sample Size Bias**
New signings have 1-2 games of data. Model treats them same as players with 38 games.

**Fix:** Bayesian shrinkage - regress small samples toward positional mean.

---

### 35. **No Fatigue Modeling**
Players in Europa League or playing 3 games in 7 days should have reduced EP.

---

## ✅ THINGS THAT ARE ACTUALLY GOOD

1. **Clean Code Structure** - Well-organized modules
2. **Advanced Metrics** - EPPM, threat momentum are innovative
3. **Fuzzy Name Matching** - Levenshtein matching works well
4. **Multiple Optimization Strategies** - Good variety (greedy, differential, value)
5. **Monte Carlo Integration** - Proper uncertainty quantification
6. **Genetic Algorithm** - Clever approach to constraint satisfaction
7. **Streamlit UI** - Clean, professional design
8. **Comprehensive Feature Engineering** - Lots of useful derived metrics
9. **Modular Design** - Easy to extend with new tabs
10. **Session State Management** - Proper use of st.session_state

---

## 🎯 PRIORITY FIX LIST (Do These First)

1. **Fix Poisson FDR proxy** → Use real team xGA data
2. **Fix CBIT calculation** → Set to 0 for non-defenders OR fetch real data
3. **Fix DGW multiplication** → Iterate fixtures independently
4. **Add Understat status indicator** → Show when using fallback
5. **Fix starts_pct calculation** → Use minutes / team_games
6. **Add injury probability** → Multiply EP by chance_of_playing
7. **Invert variance-adjusted differential** → Stop penalizing hot players
8. **Add backtesting** → Validate model accuracy
9. **Add unit tests** → Prevent regressions
10. **Fix captain multiplier** → Should be 2.0 not 1.25

---

## 💡 RECOMMENDED ENHANCEMENTS

### Short-term (Next 2 Weeks)
- Fix all 🔴 Critical Issues
- Add data freshness timestamp
- Export to CSV functionality
- Error message improvements

### Medium-term (Next Month)
- Implement real team xGA/xG fetching
- Add backtesting framework
- Unit test coverage >50%
- Historical data tracking

### Long-term (Next Quarter)
- Database integration (PostgreSQL)
- Time-series ownership tracking
- Mini-league batch analysis
- Mobile-responsive design
- API for external integrations

---

## 📈 SUGGESTED METRICS TO ADD

1. **Model Calibration Score** - Are your 5-point predictions actually averaging 5 points?
2. **Hit Rate** - % of weeks your top pick outperformed ownership-weighted average
3. **Differential Success** - Do your <5% picks actually outperform?
4. **Transfer Efficiency** - Points gained per transfer made
5. **Captain ROI** - Did your captain picks beat the template?

---

## 🔬 ADVANCED FEATURES TO CONSIDER

1. **Bayesian Updating** - Update EP predictions as GW progresses (live scores)
2. **Sentiment Analysis** - Scrape Reddit/Twitter for injury rumors
3. **Weather API** - Rain/wind affects scoring
4. **Referee Stats** - Some refs give more cards/pens
5. **xT (Expected Threat)** - Beyond just xG/xA
6. **Machine Learning Ensemble** - Combine Poisson + XGBoost + LightGBM

---

## 🎓 LEARNING RESOURCES

For improving the model:
- [FPL Analytics Handbook](https://github.com/vaastav/Fantasy-Premier-League)
- [Expected Goals Philosophy](https://www.amazon.com/Expected-Goals-Philosophy-James-Yorke/dp/1801500053)
- [FPL Review's Model Explanation](https://fplreview.com/expected-points-model/)
- [Mathematical Soccer](https://www.soccer-rating.com)

---

## 🔚 CONCLUSION

**Overall Assessment: B+ (Very Good, with fixable flaws)**

**Strengths:**
- Ambitious scope with advanced features
- Solid engineering practices (mostly)
- Professional UI/UX
- Innovative metrics (EPPM, threat momentum)

**Weaknesses:**
- Critical bugs in new Poisson model (FDR proxy, DGW)
- CBIT data is fake
- No validation/testing
- Statistical methodology needs refinement

**Recommended Next Steps:**
1. Fix the 10 priority issues above
2. Add backtesting to validate Poisson model
3. Implement proper team-level xGA fetching
4. Add unit tests for critical calculations

With these fixes, this would be a **production-grade** FPL tool competitive with FPL Review and LiveFPL.

---

*End of Audit Report*
