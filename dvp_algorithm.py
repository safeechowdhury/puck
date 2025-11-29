"""
Defense vs Position (DvP) Algorithm Design

This module implements the Schedule-Adjusted Defense v Position rankings
which measures how much an opposing team's defense inflates (+) or suppresses (-) 
a position's production compared to that opponent's season average.

Algorithm Steps:
1. Calculate team defensive performance by position
2. Calculate opponent offensive averages by position  
3. Adjust for schedule strength
4. Compute DvP differential
"""

import polars as pl
from typing import Dict, List, Tuple

def calculate_team_defense_by_position(
    df: pl.DataFrame, 
    stat_col: str, 
    season_id: int
) -> pl.DataFrame:
    """
    Calculate how each team performs defensively against each position.
    
    Args:
        df: Comprehensive boxscores data
        stat_col: Stat column to analyze (e.g., 'shots', 'points', 'goals')
        season_id: Current season ID
        
    Returns:
        DataFrame with team defense metrics by position
    """
    # Filter for current season
    df_season = df.filter(pl.col("season") == season_id)
    
    # Calculate defensive performance (what opponents do against each team)
    team_defense = df_season.group_by(["team", "position"]).agg([
        pl.col(stat_col).mean().alias(f"avg_{stat_col}_allowed"),
        pl.len().alias("games_count")
    ])
    
    return team_defense

def calculate_position_offensive_averages(
    df: pl.DataFrame, 
    stat_col: str, 
    season_id: int
) -> pl.DataFrame:
    """
    Calculate league-wide offensive averages by position.
    
    Args:
        df: Comprehensive boxscores data
        stat_col: Stat column to analyze
        season_id: Current season ID
        
    Returns:
        DataFrame with position offensive averages
    """
    # Filter for current season
    df_season = df.filter(pl.col("season") == season_id)
    
    # Calculate position averages across all teams
    position_averages = df_season.group_by("position").agg([
        pl.col(stat_col).mean().alias(f"league_avg_{stat_col}"),
        pl.len().alias("total_games")
    ])
    
    return position_averages

def calculate_schedule_strength(
    df: pl.DataFrame, 
    stat_col: str, 
    season_id: int
) -> pl.DataFrame:
    """
    Calculate schedule strength for each team based on opponents' offensive quality.
    
    Args:
        df: Comprehensive boxscores data
        stat_col: Stat column to analyze
        season_id: Current season ID
        
    Returns:
        DataFrame with schedule strength factors for each team
    """
    # Filter for current season
    df_season = df.filter(pl.col("season") == season_id)
    
    # Calculate each team's offensive quality by position
    team_offensive = df_season.group_by(["team", "position"]).agg([
        pl.col(stat_col).mean().alias(f"team_offensive_{stat_col}")
    ])
    
    # For each team, calculate average offensive quality of their opponents
    # This requires joining with schedule data to get matchups
    schedule_strength = df_season.group_by("team").agg([
        pl.col("opponent").unique().alias("opponents_played")
    ])
    
    # Explode opponents to get one row per opponent
    schedule_strength = schedule_strength.explode("opponents_played")
    
    # Join with team offensive data
    schedule_strength = schedule_strength.join(
        team_offensive,
        left_on="opponents_played",
        right_on="team",
        how="left"
    )
    
    # Calculate average opponent offensive quality by position
    schedule_strength = schedule_strength.group_by(["team", "position"]).agg([
        pl.col(f"team_offensive_{stat_col}").mean().alias("avg_opponent_offensive")
    ])
    
    # Normalize schedule strength (1.0 = average schedule)
    league_avg = df_season.group_by("position").agg([
        pl.col(stat_col).mean().alias(f"position_avg_{stat_col}")
    ])
    
    schedule_strength = schedule_strength.join(
        league_avg,
        on="position",
        how="left"
    ).with_columns([
        (pl.col("avg_opponent_offensive") / pl.col(f"position_avg_{stat_col}"))
        .alias("schedule_strength_factor")
    ])
    
    return schedule_strength.select(["team", "position", "schedule_strength_factor"])

def calculate_dvp_metrics(
    df: pl.DataFrame, 
    stat_col: str, 
    season_id: int
) -> pl.DataFrame:
    """
    Calculate Schedule-Adjusted Defense v Position metrics.
    
    Args:
        df: Comprehensive boxscores data
        stat_col: Stat column to analyze
        season_id: Current season ID
        
    Returns:
        DataFrame with DvP metrics for each team-position combination
    """
    # Step 1: Calculate team defensive performance
    team_defense = calculate_team_defense_by_position(df, stat_col, season_id)
    
    # Step 2: Calculate position offensive averages
    position_averages = calculate_position_offensive_averages(df, stat_col, season_id)
    
    # Step 3: Calculate schedule strength
    schedule_strength = calculate_schedule_strength(df, stat_col, season_id)
    
    # Step 4: Combine metrics
    dvp = team_defense.join(
        position_averages,
        on="position",
        how="left"
    ).join(
        schedule_strength,
        on=["team", "position"],
        how="left"
    )
    
    # Step 5: Calculate raw DvP (actual vs expected)
    dvp = dvp.with_columns([
        # Raw DvP: what team allows vs league average
        (pl.col(f"avg_{stat_col}_allowed") - pl.col(f"league_avg_{stat_col}"))
        .alias("raw_dvp")
    ])
    
    # Step 6: Adjust for schedule strength
    dvp = dvp.with_columns([
        # Schedule-adjusted DvP: remove bias from easy/difficult schedules
        (pl.col("raw_dvp") / pl.col("schedule_strength_factor"))
        .alias("schedule_adjusted_dvp")
    ])
    
    # Step 7: Calculate rankings
    dvp = dvp.with_columns([
        pl.col("schedule_adjusted_dvp").rank(method="dense", descending=True)
        .over("position")
        .alias("dvp_rank")
    ])
    
    # Step 8: Format for display
    dvp = dvp.with_columns([
        pl.col("schedule_adjusted_dvp").round(2).alias("dvp_value"),
        pl.when(pl.col("schedule_adjusted_dvp") > 0)
        .then(pl.lit("+") + pl.col("schedule_adjusted_dvp").round(1).cast(pl.Utf8))
        .otherwise(pl.col("schedule_adjusted_dvp").round(1).cast(pl.Utf8))
        .alias("dvp_formatted")
    ])
    
    return dvp.select([
        "team", "position", "dvp_value", "dvp_rank", "dvp_formatted",
        f"avg_{stat_col}_allowed", f"league_avg_{stat_col}", 
        "schedule_strength_factor", "games_count"
    ])

def get_matchup_dvp(
    dvp_df: pl.DataFrame, 
    team: str, 
    opponent: str, 
    position: str, 
    stat_col: str
) -> Dict:
    """
    Get DvP information for a specific matchup.
    
    Args:
        dvp_df: DvP metrics DataFrame
        team: Team to analyze (defense)
        opponent: Opponent team (offense)
        position: Position to analyze
        stat_col: Stat column
        
    Returns:
        Dictionary with DvP information
    """
    # Get opponent's offensive average for this position
    opponent_offensive = dvp_df.filter(
        (pl.col("team") == opponent) & 
        (pl.col("position") == position)
    )
    
    # Get team's defensive DvP for this position
    team_dvp = dvp_df.filter(
        (pl.col("team") == team) & 
        (pl.col("position") == position)
    )
    
    if opponent_offensive.is_empty() or team_dvp.is_empty():
        return {"dvp_value": 0.0, "dvp_rank": 16, "dvp_formatted": "0.0"}
    
    opponent_avg = opponent_offensive[f"avg_{stat_col}_allowed"][0]
    dvp_value = team_dvp["dvp_value"][0]
    dvp_rank = team_dvp["dvp_rank"][0]
    dvp_formatted = team_dvp["dvp_formatted"][0]
    
    return {
        "dvp_value": dvp_value,
        "dvp_rank": dvp_rank,
        "dvp_formatted": dvp_formatted,
        "opponent_avg": opponent_avg
    }

# Position mapping for display
POSITION_MAPPING = {
    "C": "Center",
    "L": "Left Wing", 
    "R": "Right Wing",
    "D": "Defense",
    "G": "Goalie"
}

def get_position_display(position_code: str) -> str:
    """Convert position code to display name."""
    return POSITION_MAPPING.get(position_code, position_code)