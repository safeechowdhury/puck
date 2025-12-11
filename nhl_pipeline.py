"""
NHL Data Pipeline - Production-Ready Async Implementation
Supports both historical backfills and incremental refreshes

Updated with:
- Robust Power Play Goal calculation logic
- Validation tools
- Season-aware team fetching for accurate historical backfills
- Season-aware player roster fetching (e.g., data/20232024/players.parquet)
- Current player roster fetching (data/players.parquet)
- Archiving logic for refresh updates
- Efficient httpx session management
"""

import asyncio
import httpx
import polars as pl
import polars.selectors as cs
from pathlib import Path
from datetime import date, timedelta, datetime
from typing import Optional, List, Dict, Set, Tuple
import logging
import json
from dataclasses import dataclass
import argparse

# Configure logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
# Prevent duplicate logs if root logger is configured elsewhere
logger.propagate = False

# ============================================================================
# CONFIGURATION
# ============================================================================

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15'
    ),
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-CA,en-US;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://www.nhl.com/',
}

OUTPUT_DIR = Path("data")
MAX_CONCURRENT_REQUESTS = 16
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# ============================================================================
# SCHEMAS
# ============================================================================

SCHEDULE_SCHEMA = {
    "game_id": pl.Int64,
    "date": pl.Utf8,
    "season": pl.Int64,
    "game_type": pl.Int64,
    "away_abbrev": pl.Utf8,
    "away_score": pl.Int64,
    "home_abbrev": pl.Utf8,
    "home_score": pl.Int64,
    "start_time": pl.Utf8,
    "venue": pl.Utf8,
}

BOXSCORE_SCHEMA = {
    "game_id": pl.Int64,
    "player_id": pl.Int64,
    "player_name": pl.Utf8,
    "team": pl.Utf8,
    "opponent": pl.Utf8,
    "location": pl.Utf8,
    "position": pl.Utf8,
    "plus_minus": pl.Int32,
    "toi": pl.Utf8,
    "shifts": pl.Int32,
    "shots_against": pl.Int32,
    "saves": pl.Int32,
    "goals_against": pl.Int32,
    "save_pct": pl.Float64,
    "decision": pl.Utf8,
    "game_winning_goal": pl.Int32,
    "ot_loss": pl.Int32,
}

PBP_RAW_SCHEMA = {
    "game_id": pl.Int64,
    "event_type": pl.Utf8,
    "situation_code": pl.Utf8,
    "event_owner_team_id": pl.Int64,
    "home_team_id": pl.Int64,
    "away_team_id": pl.Int64,
    "scoring_player_id": pl.Int64,
    "assist1_player_id": pl.Int64,
    "assist2_player_id": pl.Int64,
    "shooting_player_id": pl.Int64,
    "blocking_player_id": pl.Int64,
    "hitting_player_id": pl.Int64,
    "committed_player_id": pl.Int64,
    "winning_player_id": pl.Int64,
    "losing_player_id": pl.Int64,
    "duration": pl.Int32,
    "player_id": pl.Int64,
    "home_score": pl.Int32,
    "away_score": pl.Int32,
    "period_type": pl.Utf8,
}

PLAYER_ROSTER_SCHEMA = {
    "team_abbrev": pl.Utf8,
    "player_id": pl.Int64,
    "full_player_name": pl.Utf8,
    "position_code": pl.Utf8,
    "headshot_url": pl.Utf8,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_season_output_dir(season_id: int) -> Path:
    """Get output directory for a specific season."""
    season_dir = OUTPUT_DIR / str(season_id)
    season_dir.mkdir(parents=True, exist_ok=True)
    return season_dir


def get_player_id_from_details(play: dict, key: str) -> Optional[int]:
    """Safely extracts a player ID from nested details."""
    return play.get("details", {}).get(key)


async def _get_season_standings_end_date(
    session: httpx.AsyncClient, 
    season_id: int
) -> Optional[str]:
    """
    Fetch the 'standingsEnd' date for a specific season from the season metadata.
    """
    url = "https://api-web.nhle.com/v1/standings-season"
    try:
        response = await session.get(url, headers=HEADERS)
        response.raise_for_status()
        seasons_data = response.json().get("seasons", [])
        
        for season in seasons_data:
            if season.get("id") == season_id:
                return season.get("standingsEnd")
                
        logger.warning(f"Could not find standingsEnd date for season {season_id}")
        return None
    except Exception as e:
        logger.error(f"Error fetching season end dates: {e}")
        return None


async def get_current_season_id(session: httpx.AsyncClient) -> int:
    """
    Fetches the current season ID from the 'standings-season' endpoint.
    This is typically the last season in the returned list.
    """
    url = "https://api-web.nhle.com/v1/standings-season"
    try:
        response = await session.get(url, headers=HEADERS)
        response.raise_for_status()
        seasons_data = response.json().get("seasons", [])
        
        if seasons_data:
            # The "current" season is the last one in the list
            current_season = seasons_data[-1].get("id")
            if current_season:
                return current_season
                
    except Exception as e:
        logger.warning(f"Could not dynamically fetch current season: {e}. Falling back.")
    
    # Fallback to a recent, likely "current" season
    logger.warning("Falling back to hardcoded season 20252026")
    return 20252026


# ============================================================================
# SCHEDULE FETCHING
# ============================================================================

async def fetch_active_teams(
    session: httpx.AsyncClient, 
    standings_date: Optional[str] = None
) -> List[str]:
    """
    Fetch list of active team abbreviations from standings
    on a specific date (if provided) or from 'now'.
    """
    today_str = date.today().strftime("%Y-%m-%d")
    # Use standings_date if provided (for historical), otherwise default to today
    date_to_fetch = standings_date if standings_date else today_str
    
    url = f"https://api-web.nhle.com/v1/standings/{date_to_fetch}"
    log_msg = f"Fetching teams from 'standings/{date_to_fetch}'"
    
    logger.info(log_msg)
    
    try:
        response = await session.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        teams = set()
        for standing in data.get("standings", []):
            team_abbrev = standing.get("teamAbbrev", {}).get("default")
            if team_abbrev:
                teams.add(team_abbrev)
        
        if not teams:
             raise ValueError("No teams found in standings data.")
             
        return sorted(list(teams))
        
    except Exception as e:
        logger.error(f"Error fetching active teams: {e}")
        # Fallback to known teams
        return [
            "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
            "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
            "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "VAN", "VGK", "WPG",
            "WSH", "ARI", "UTA"
        ]


async def fetch_team_schedule(
    team_abbrev: str, 
    season_id: int, 
    session: httpx.AsyncClient
) -> List[Dict]:
    """Fetch schedule for a single team."""
    url = f"https://api-web.nhle.com/v1/club-schedule-season/{team_abbrev}/{season_id}"
    
    try:
        response = await session.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        games = []
        for game in data.get("games", []):
            games.append({
                "game_id": game.get("id"),
                "date": game.get("gameDate"),
                "season": season_id,
                "game_type": game.get("gameType"),
                "away_abbrev": game.get("awayTeam", {}).get("abbrev"),
                "away_score": game.get("awayTeam", {}).get("score", 0),
                "home_abbrev": game.get("homeTeam", {}).get("abbrev"),
                "home_score": game.get("homeTeam", {}).get("score", 0),
                "start_time": game.get("startTimeUTC"),
                "venue": game.get("venue", {}).get("default"),
            })
        
        return games
    except Exception as e:
        logger.warning(f"Error fetching schedule for {team_abbrev}: {e}")
        return []


async def fetch_season_game_ids(
    season_id: int,
    teams: List[str],  # <-- ADDED
    session: httpx.AsyncClient,  # <-- ADDED
    game_types: List[str] = ["R", "P"]
) -> pl.DataFrame:
    """Fetch all game IDs for a season."""
    logger.info(f"Fetching schedule for season {season_id} for {len(teams)} teams")

    # This function now expects the session and team list to be provided
    
    tasks = [
        fetch_team_schedule(team, season_id, session) 
        for team in teams
    ]
    results = await asyncio.gather(*tasks)
    
    all_games = []
    seen_game_ids = set()
    
    for team_games in results:
        for game in team_games:
            game_id = game["game_id"]
            if game_id not in seen_game_ids:
                seen_game_ids.add(game_id)
                all_games.append(game)
    
    game_type_map = {"R": 2, "P": 3}
    allowed_types = [game_type_map.get(gt, 2) for gt in game_types]
    all_games = [g for g in all_games if g["game_type"] in allowed_types]
    
    logger.info(f"Found {len(all_games)} unique games")
    
    if not all_games:
        return pl.DataFrame(schema=SCHEDULE_SCHEMA)
    
    df = pl.DataFrame(all_games, schema=SCHEDULE_SCHEMA)
    return df.sort("date", "game_id")


def add_days_rest_to_schedule(df_schedule: pl.DataFrame) -> pl.DataFrame:
    """
    Add days_rest columns to schedule for both home and away teams.
    This shows how many days rest each team had before the game (capped at 3+).
    """
    if df_schedule.is_empty():
        return df_schedule
    
    # Create a flattened view of all games per team
    df_home = df_schedule.select(["date", "game_id", "home_abbrev"]).rename({"home_abbrev": "team"})
    df_away = df_schedule.select(["date", "game_id", "away_abbrev"]).rename({"away_abbrev": "team"})
    df_team_games = pl.concat([df_home, df_away]).sort("team", "date")
    
    # Calculate days since last game for each team
    df_team_games = df_team_games.with_columns([
        pl.col("date").shift(1).over("team").alias("prev_date")
    ]).with_columns([
        pl.when(pl.col("prev_date").is_null())
        .then(pl.lit(None).cast(pl.Int32))
        .otherwise(
            pl.min_horizontal(
                (pl.col("date").str.to_date() - pl.col("prev_date").str.to_date()).dt.total_days() - 1,
                pl.lit(3)
            ).cast(pl.Int32)
        )
        .alias("days_rest")
    ]).drop("prev_date")
    
    # Get home team days rest
    df_home_rest = (
        df_team_games
        .join(df_schedule.select(["game_id", "home_abbrev"]), 
              left_on=["game_id", "team"], 
              right_on=["game_id", "home_abbrev"],
              how="inner")
        .select(["game_id", "days_rest"])
        .rename({"days_rest": "home_days_rest"})
    )
    
    # Get away team days rest  
    df_away_rest = (
        df_team_games
        .join(df_schedule.select(["game_id", "away_abbrev"]),
              left_on=["game_id", "team"],
              right_on=["game_id", "away_abbrev"],
              how="inner")
        .select(["game_id", "days_rest"])
        .rename({"days_rest": "away_days_rest"})
    )
    
    # Join back to schedule
    result = (
        df_schedule
        .join(df_home_rest, on="game_id", how="left")
        .join(df_away_rest, on="game_id", how="left")
    )
    
    return result


# ============================================================================
# ASYNC DATA FETCHING
# ============================================================================

async def fetch_team_roster(
    team_abbrev: str,
    season_id: int,  # <-- ADDED
    session: httpx.AsyncClient,
    semaphore: asyncio.Semaphore
) -> List[Dict]:
    """Fetch roster for a single team for a specific season."""
    url = f"https://api-web.nhle.com/v1/roster/{team_abbrev}/{season_id}" # <-- DYNAMIC
    
    async with semaphore:
        try:
            response = await session.get(url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            
            roster = []
            for position_group in ['forwards', 'defensemen', 'goalies']:
                if position_group in data:
                    for player in data.get(position_group, []):
                        # Use robust .get() chains for safety
                        first_name = (player.get('firstName') or {}).get('default', '')
                        last_name = (player.get('lastName') or {}).get('default', '')
                        
                        roster.append({
                            'team_abbrev': team_abbrev,
                            'player_id': player.get('id'),
                            'full_player_name': f"{first_name} {last_name}".strip(),
                            'position_code': player.get('positionCode'),
                            'headshot_url': player.get('headshot'),
                        })
            return roster
        except Exception as e:
            # Updated log message
            logger.warning(
                f"Error fetching roster for {team_abbrev} (Season {season_id}): {e}"
            )
            return []


async def update_player_rosters(
    session: httpx.AsyncClient,
    season_id: int,
    teams: List[str],
    is_current: bool = False
) -> pl.DataFrame:
    """
    Fetches rosters for all active teams for a season and saves to parquet.
    If is_current=True, saves to data/players.parquet.
    Otherwise, saves to data/{season_id}/players.parquet.
    """
    logger.info(f"Updating player rosters for season {season_id}...")
    
    if not teams:
        logger.error(f"No teams provided for season {season_id}, cannot update player rosters.")
        return pl.DataFrame(schema=PLAYER_ROSTER_SCHEMA)
        
    logger.info(f"Extracting rosters for {len(teams)} teams...")
    
    # 2. Create tasks to fetch all rosters concurrently
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [
        # Pass the season_id to the fetch function
        fetch_team_roster(team, season_id, session, semaphore) 
        for team in teams
    ]
    results = await asyncio.gather(*tasks)
    
    # 3. Flatten list of lists into a single list of players
    all_rosters = [player for team_roster in results for player in team_roster]
    
    if not all_rosters:
        logger.warning(f"No player data was fetched for season {season_id}.")
        return pl.DataFrame(schema=PLAYER_ROSTER_SCHEMA)
        
    # 4. Create DataFrame
    df = pl.DataFrame(all_rosters, schema=PLAYER_ROSTER_SCHEMA)
    
    # 5. Save file to the dynamic path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if is_current:
        output_file = OUTPUT_DIR / "players.parquet"
        log_msg = f"Saved CURRENT player rosters: {output_file} ({len(df)} players)"
    else:
        season_dir = get_season_output_dir(season_id)
        output_file = season_dir / "players.parquet"
        log_msg = f"Saved HISTORICAL player rosters: {output_file} ({len(df)} players)"
    
    df.write_parquet(output_file)
    logger.info(log_msg)
    return df


async def fetch_game_boxscore(
    game_id: int, 
    session: httpx.AsyncClient,
    semaphore: asyncio.Semaphore
) -> Optional[pl.DataFrame]:
    """Fetch essential boxscore data for a single game."""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    
    async with semaphore:
        try:
            response = await session.get(url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            
            if "playerByGameStats" not in data:
                return None
            
            away_team = (data.get("awayTeam") or {}).get("abbrev", "")
            home_team = (data.get("homeTeam") or {}).get("abbrev", "")
            
            player_stats = []
            
            for team_type in ["awayTeam", "homeTeam"]:
                team_data = data["playerByGameStats"].get(team_type, {})
                if not team_data:
                    continue
                
                team_abbrev = away_team if team_type == "awayTeam" else home_team
                opponent_abbrev = home_team if team_type == "awayTeam" else away_team
                location = "away" if team_type == "awayTeam" else "home"
                
                # Process skaters
                for position_group in ["forwards", "defense"]:
                    for player in team_data.get(position_group, []):
                        player_stats.append({
                            "game_id": game_id,
                            "player_id": player.get("playerId"),
                            "player_name": (player.get("name") or {}).get("default", ""),
                            "team": team_abbrev,
                            "opponent": opponent_abbrev,
                            "location": location,
                            "position": player.get("position"),
                            "plus_minus": player.get("plusMinus", 0),
                            "toi": player.get("toi", "0:00"),
                            "shifts": player.get("shifts", 0),
                            "shots_against": None,
                            "saves": None,
                            "goals_against": None,
                            "save_pct": None,
                            "decision": None,
                            "game_winning_goal": player.get("gameWinningGoals", 0),
                            "ot_loss": 0,
                        })
                
                # Process goalies
                for goalie in team_data.get("goalies", []):
                    player_stats.append({
                        "game_id": game_id,
                        "player_id": goalie.get("playerId"),
                        "player_name": (goalie.get("name") or {}).get("default", ""),
                        "team": team_abbrev,
                        "opponent": opponent_abbrev,
                        "location": location,
                        "position": "G",
                        "plus_minus": 0,
                        "toi": goalie.get("toi", "0:00"),
                        "shifts": 0,
                        "shots_against": goalie.get("shotsAgainst", 0),
                        "saves": goalie.get("saves", 0),
                        "goals_against": goalie.get("goalsAgainst", 0),
                        "save_pct": goalie.get("savePctg", 0.0),
                        "decision": goalie.get("decision", ""),
                        "game_winning_goal": 0,
                        "ot_loss": 1 if goalie.get("decision") == "O" else 0,
                    })
            
            if not player_stats:
                return None
            
            return pl.DataFrame(player_stats, schema=BOXSCORE_SCHEMA)
            
        except Exception as e:
            logger.warning(f"Error fetching boxscore for game {game_id}: {e}")
            return None


async def fetch_game_pbp(
    game_id: int, 
    session: httpx.AsyncClient,
    semaphore: asyncio.Semaphore
) -> Optional[pl.DataFrame]:
    """Fetch raw PBP events including Team IDs for PP calculation."""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    
    async with semaphore:
        try:
            response = await session.get(url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            
            if "plays" not in data or not data["plays"]:
                return None
            
            away_team_id = data.get("awayTeam", {}).get("id")
            home_team_id = data.get("homeTeam", {}).get("id")
            
            events = []
            for play in data["plays"]:
                details = play.get("details", {})
                
                events.append({
                    "game_id": game_id,
                    "event_type": play.get("typeDescKey", ""),
                    "situation_code": play.get("situationCode", ""),
                    "event_owner_team_id": details.get("eventOwnerTeamId"), 
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "scoring_player_id": get_player_id_from_details(play, "scoringPlayerId"),
                    "assist1_player_id": get_player_id_from_details(play, "assist1PlayerId"),
                    "assist2_player_id": get_player_id_from_details(play, "assist2PlayerId"),
                    "shooting_player_id": get_player_id_from_details(play, "shootingPlayerId"),
                    "blocking_player_id": get_player_id_from_details(play, "blockingPlayerId"),
                    "hitting_player_id": get_player_id_from_details(play, "hittingPlayerId"),
                    "committed_player_id": get_player_id_from_details(play, "committedByPlayerId"),
                    "winning_player_id": get_player_id_from_details(play, "winningPlayerId"),
                    "losing_player_id": get_player_id_from_details(play, "losingPlayerId"),
                    "duration": details.get("duration", 0),
                    "player_id": details.get("playerId"),
                    "home_score": details.get("homeScore"),
                    "away_score": details.get("awayScore"),
                    "period_type": play.get("periodDescriptor", {}).get("periodType", "REG"),
                })
            
            if not events:
                return None
            
            return pl.DataFrame(events, schema=PBP_RAW_SCHEMA)
            
        except Exception as e:
            logger.warning(f"Error fetching PBP for game {game_id}: {e}")
            return None


async def fetch_game_data_with_retry(
    game_id: int,
    session: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    fetch_func,
    max_retries: int = MAX_RETRIES
) -> Optional[pl.DataFrame]:
    """Fetch game data with retry logic."""
    for attempt in range(max_retries + 1):
        result = await fetch_func(game_id, session, semaphore)
        if result is not None:
            return result
        
        if attempt < max_retries:
            logger.warning(f"Retrying game {game_id} (attempt {attempt + 1})")
            await asyncio.sleep(RETRY_DELAY)
    
    logger.error(f"Failed to fetch data for game {game_id} after {max_retries + 1} attempts")
    return None


# ============================================================================
# PBP AGGREGATION
# ============================================================================

def aggregate_pbp_stats(raw_pbp_df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate raw PBP events into player-level statistics.
    Includes robust logic for PPG, PPA, and PPP with enhanced error handling.
    """
    if raw_pbp_df.is_empty():
        return pl.DataFrame()
    
    # FILTER: Exclude Shootout events
    raw_pbp_df = raw_pbp_df.filter(pl.col("period_type") != "SO")
    if raw_pbp_df.is_empty():
        return pl.DataFrame()

    # --- PRE-CALCULATE POWER PLAY LOGIC (BEFORE UNPIVOT) ---
    # Situation Code Format: [AwayGoalie][AwaySkaters][HomeSkaters][HomeGoalie]
    
    df_with_ppg_flag = raw_pbp_df.with_columns([
        # Parse skater counts with robust error handling
        pl.when(pl.col("situation_code").str.len_chars() >= 4)
          .then(pl.col("situation_code").str.slice(1, 1).cast(pl.Int32, strict=False))
          .otherwise(None)
          .fill_null(5)
          .alias("away_skaters"),
        
        pl.when(pl.col("situation_code").str.len_chars() >= 4)
          .then(pl.col("situation_code").str.slice(2, 1).cast(pl.Int32, strict=False))
          .otherwise(None)
          .fill_null(5)
          .alias("home_skaters"),
        
        # Determine if event owner is Away Team
        (pl.col("event_owner_team_id") == pl.col("away_team_id")).alias("is_away_event")
    ]).with_columns(
        # Set a flag on the EVENT if it's a Power Play Goal
        (
            (pl.col("event_type") == "goal") &
            (
                # Away Scored AND Away > Home AND Home < 5 skaters
                (
                    pl.col("is_away_event") &
                    (pl.col("away_skaters") > pl.col("home_skaters")) &
                    (pl.col("home_skaters") < 5)
                ) |
                # Home Scored AND Home > Away AND Away < 5 skaters
                (
                    (~pl.col("is_away_event")) &
                    (pl.col("home_skaters") > pl.col("away_skaters")) &
                    (pl.col("away_skaters") < 5)
                )
            )
        ).alias("is_ppg_event")
    ).with_columns([
        # --- GAME WINNING GOAL LOGIC ---
        # 1. Identify the winning team (max score at end of game)
        pl.col("home_score").max().over("game_id").alias("final_home_score"),
        pl.col("away_score").max().over("game_id").alias("final_away_score")
    ]).with_columns([
        # 2. Determine winning team ID
        pl.when(pl.col("final_home_score") > pl.col("final_away_score"))
        .then(pl.col("home_team_id"))
        .when(pl.col("final_away_score") > pl.col("final_home_score"))
        .then(pl.col("away_team_id"))
        .otherwise(None) # Tie/Shootout logic handled separately or ignored for GWG
        .alias("winning_team_id"),
        
        # 3. Determine the "GWG Score Threshold"
        # The goal that makes the score LoserScore + 1 for the winner
        pl.when(pl.col("final_home_score") > pl.col("final_away_score"))
        .then(pl.col("final_away_score") + 1)
        .when(pl.col("final_away_score") > pl.col("final_home_score"))
        .then(pl.col("final_home_score") + 1)
        .otherwise(None)
        .alias("gwg_score_threshold")
    ]).with_columns(
        # 4. Flag the specific goal event
        (
            (pl.col("event_type") == "goal") &
            (pl.col("event_owner_team_id") == pl.col("winning_team_id")) &
            (
                # Check if this goal reached the threshold
                # For home winner: home_score == threshold
                # For away winner: away_score == threshold
                (
                    (pl.col("winning_team_id") == pl.col("home_team_id")) & 
                    (pl.col("home_score") == pl.col("gwg_score_threshold"))
                ) |
                (
                    (pl.col("winning_team_id") == pl.col("away_team_id")) & 
                    (pl.col("away_score") == pl.col("gwg_score_threshold"))
                )
            )
        ).alias("is_gwg_event")
    )

    # --- UNPIVOT TO LONG FORMAT ---
    df_long = df_with_ppg_flag.unpivot(
        on=[
            "scoring_player_id", "assist1_player_id", "assist2_player_id",
            "shooting_player_id", "blocking_player_id", "hitting_player_id",
            "committed_player_id", "winning_player_id", "losing_player_id",
            "player_id"
        ],
        index=[
            "game_id", "event_type", "situation_code", "duration",
            "event_owner_team_id", "home_team_id", "away_team_id",
            "is_ppg_event", "is_gwg_event"  # Carry the flags through
        ],
        variable_name="event_role",
        value_name="player_id"
    ).drop_nulls(subset=["player_id"])
    
    # FIX: Ensure event_role is string type to prevent type mismatch in aggregation
    df_long = df_long.with_columns([
        pl.col("event_role").cast(pl.Utf8).alias("event_role")
    ])
    
    if df_long.is_empty():
        return pl.DataFrame()
    
    # --- AGGREGATION ---
    
    df_pbp_stats = df_long.group_by(["game_id", "player_id"]).agg([
        # Goals
        pl.col("event_type").filter(
            (pl.col("event_type") == "goal") & 
            (pl.col("event_role") == "scoring_player_id")
        ).count().alias("goals"),
        
        # Assists
        pl.col("event_type").filter(
            (pl.col("event_type") == "goal") & 
            (pl.col("event_role").str.contains("assist"))
        ).count().alias("assists"),
        
        # Saves (by shooter)
        pl.col("event_type").filter(
            (pl.col("event_type") == "shot-on-goal") & 
            (pl.col("event_role") == "shooting_player_id")
        ).count().alias("saves_on_shooter"),
        
        # Missed Shots
        pl.col("event_type").filter(
            (pl.col("event_type") == "missed-shot") & 
            (pl.col("event_role") == "shooting_player_id")
        ).count().alias("missed_shots"),
        
        # Blocked Shots (Defensive)
        pl.col("event_type").filter(
            (pl.col("event_type") == "blocked-shot") & 
            (pl.col("event_role") == "blocking_player_id")
        ).count().alias("blocked_shots"),
        
        # Offensive Blocked Shots
        pl.col("event_type").filter(
            (pl.col("event_type") == "blocked-shot") & 
            (pl.col("event_role") == "shooting_player_id")
        ).count().alias("offensive_blocked_shots"),
        
        # Hits
        pl.col("event_type").filter(
            (pl.col("event_type") == "hit") & 
            (pl.col("event_role") == "hitting_player_id")
        ).count().alias("hits"),
        
        # Giveaways
        pl.col("event_type").filter(
            (pl.col("event_type") == "giveaway") & 
            (pl.col("event_role") == "player_id")
        ).count().alias("giveaways"),
        
        # Takeaways
        pl.col("event_type").filter(
            (pl.col("event_type") == "takeaway") & 
            (pl.col("event_role") == "player_id")
        ).count().alias("takeaways"),
        
        # PIM
        pl.col("duration").filter(
            (pl.col("event_type") == "penalty") & 
            (pl.col("event_role") == "committed_player_id")
        ).sum().alias("pim"),
        
        # Power Play Goals (PPG)
        pl.col("is_ppg_event").filter(
            (pl.col("is_ppg_event")) &
            (pl.col("event_role") == "scoring_player_id")
        ).count().alias("ppg"),
        
        # Power Play Assists (PPA)
        pl.col("is_ppg_event").filter(
            (pl.col("is_ppg_event")) &
            (pl.col("event_role").str.contains("assist"))
        ).count().alias("ppa"),
        
        # Game Winning Goal (GWG)
        pl.col("is_gwg_event").filter(
            (pl.col("is_gwg_event")) &
            (pl.col("event_role") == "scoring_player_id")
        ).count().alias("game_winning_goal"),
        
        # Faceoffs
        pl.col("event_type").filter(
            (pl.col("event_type") == "faceoff") & 
            (pl.col("event_role") == "winning_player_id")
        ).count().alias("faceoff_wins"),
        pl.col("event_type").filter(
            (pl.col("event_type") == "faceoff") & 
            (pl.col("event_role") == "losing_player_id")
        ).count().alias("faceoff_losses"),
    ]).fill_null(0)
    
    # --- DERIVED METRICS ---
    df_pbp_stats = df_pbp_stats.with_columns([
        # Points
        (pl.col("goals").cast(pl.Int32) + pl.col("assists").cast(pl.Int32)).alias("points"),
        
        # Power Play Points (PPP)
        (pl.col("ppg").cast(pl.Int32) + pl.col("ppa").cast(pl.Int32)).alias("ppp"),
        
        # Shots on Goal (Goals + Saves)
        (pl.col("goals").cast(pl.Int32) + pl.col("saves_on_shooter").cast(pl.Int32)).alias("shots"),
        
        # Faceoff Percentage
        (pl.col("faceoff_wins") / (pl.col("faceoff_wins") + pl.col("faceoff_losses")))
            .fill_nan(0.0).alias("faceoff_pct"),
    ])

    # Shot Attempts (Corsi)
    df_pbp_stats = df_pbp_stats.with_columns(
        (
            pl.col("shots").cast(pl.Int32) + 
            pl.col("missed_shots").cast(pl.Int32) + 
            pl.col("offensive_blocked_shots").cast(pl.Int32)
        ).alias("shot_attempts")
    )
    
    # Drop intermediate column
    return df_pbp_stats.drop("saves_on_shooter")


# ============================================================================
# DATA COMBINATION
# ============================================================================

def combine_data(df_box: pl.DataFrame, df_pbp: pl.DataFrame, df_schedule: pl.DataFrame) -> pl.DataFrame:
    """Merge boxscore and aggregated PBP data with schedule information."""
    if df_box.is_empty():
        return pl.DataFrame()
        
    # Drop boxscore's GWG (which is 0) to allow PBP's calculated GWG to take precedence
    if "game_winning_goal" in df_box.columns:
        df_box = df_box.drop("game_winning_goal")
    
    pbp_cols = [
        "goals", "assists", "points", "shots", "shot_attempts",
        "blocked_shots", "offensive_blocked_shots", "missed_shots",
        "hits", "giveaways", "takeaways", "pim", 
        "ppg", "ppa", "ppp", "game_winning_goal",
        "faceoff_wins", "faceoff_losses", "faceoff_pct"
    ]
    
    if not df_pbp.is_empty():
        df_final = df_box.join(df_pbp, on=["game_id", "player_id"], how="left")
        
        # VALIDATION: Check if PBP goals exist and compare with boxscore
        if "goals" in df_final.columns:
            total_goals = df_final.filter(pl.col("position") != "G")["goals"].sum()
            logger.info(f"Validation: Total goals from PBP: {total_goals}")
            
            # Check for any discrepancies (goals should never be null after join)
            null_goals = df_final.filter(pl.col("goals").is_null()).height
            if null_goals > 0:
                logger.warning(f"{null_goals} players have null goals after PBP join")
    else:
        # Optimized: Create all columns at once instead of in a loop
        df_final = df_box.with_columns([
            pl.lit(0).cast(pl.Float64 if "pct" in col else pl.Int32).alias(col)
            for col in pbp_cols
        ])
    
    # Fill nulls for stats (batch operation)
    int_cols = [c for c in pbp_cols if "pct" not in c]
    df_final = df_final.with_columns([
        pl.col(c).fill_null(0).cast(pl.Int32) for c in int_cols
    ] + [
        pl.col("faceoff_pct").fill_null(0.0).cast(pl.Float64)
    ])
    
    # ========================================================================
    # OPTIMIZED SECTION: Use pre-calculated days_rest from df_schedule
    # ========================================================================
    
    # Add date and pre-calculated days_rest from schedule
    df_schedule_info = df_schedule.select([
        "game_id", 
        "date", 
        "home_days_rest", 
        "away_days_rest"
    ])
    df_final = df_final.join(df_schedule_info, on="game_id", how="left")
    
    # Map home/away days_rest to the player's team
    df_final = df_final.with_columns(
        pl.when(pl.col("location") == "home")
        .then(pl.col("home_days_rest"))
        .otherwise(pl.col("away_days_rest"))
        .alias("days_rest")
    ).drop(["home_days_rest", "away_days_rest"])
    
    # ========================================================================
    # End of optimized section
    # ========================================================================
    
    # Optimized TOI conversion (vectorized, pure Polars)
    df_final = df_final.with_columns(
        pl.when(pl.col("toi").str.contains(":"))
        .then(
            pl.col("toi").str.split(":")
            .list.get(0).cast(pl.Int32, strict=False).fill_null(0) +
            (pl.col("toi").str.split(":")
             .list.get(1).cast(pl.Int32, strict=False).fill_null(0) / 60.0)
        )
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("toi_minutes")
    )
    
    # Final column order
    final_columns_order = [
        "game_id", "date", "player_id", "player_name", "team", "opponent", "location",
        "position", "days_rest", "toi", "toi_minutes", "shifts", 
        "goals", "assists", "points", "plus_minus", "pim", 
        "shots", "shot_attempts", "missed_shots", "offensive_blocked_shots", 
        "blocked_shots", "hits", "giveaways", "takeaways", 
        "faceoff_wins", "faceoff_losses", "faceoff_pct", 
        "ppg", "ppa", "ppp", "game_winning_goal",
        "shots_against", "saves", "goals_against", "save_pct", "decision", "ot_loss"
    ]
    
    existing_cols = [c for c in final_columns_order if c in df_final.columns]
    return df_final.select(existing_cols)


# ============================================================================
# VALIDATION & DIAGNOSTICS
# ============================================================================

def validate_data_integrity(df: pl.DataFrame) -> List[str]:
    """Check for data anomalies."""
    issues = []
    if df.is_empty():
        return ["DataFrame is empty"]
    
    # TOI Anomalies
    if "toi_minutes" in df.columns and "position" in df.columns:
        high_toi_skaters = df.filter((pl.col("position") != "G") & (pl.col("toi_minutes") > 35))
        if not high_toi_skaters.is_empty():
            issues.append(f"Found {len(high_toi_skaters)} skaters with > 35 mins TOI")
            
        high_toi_goalies = df.filter((pl.col("position") == "G") & (pl.col("toi_minutes") > 66))
        if not high_toi_goalies.is_empty():
            issues.append(f"Found {len(high_toi_goalies)} goalies with > 66 mins TOI")

    # Shift Consistency
    if "shifts" in df.columns and "toi_minutes" in df.columns:
        shifts_no_toi = df.filter((pl.col("shifts") > 0) & (pl.col("toi_minutes") == 0))
        if not shifts_no_toi.is_empty():
             issues.append(f"Found {len(shifts_no_toi)} players with shifts but 0 TOI")
             
    return issues

async def validate_goals_for_player(player_name: str, days_back: int = 7):
    """
    Diagnostic function to validate goals calculation for a specific player.
    Compares boxscore goals with PBP-derived goals.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"GOAL VALIDATION FOR: {player_name}")
    logger.info('='*60)
    
    async with httpx.AsyncClient(timeout=30.0) as session:
        # Fetch recent games
        df_recent = await fetch_recent_games(session, days_back)
        
        if df_recent.is_empty():
            logger.warning("No recent games found")
            return
        
        # Fetch data
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        box_tasks = [fetch_game_data_with_retry(row["game_id"], session, semaphore, fetch_game_boxscore) for row in df_recent.iter_rows(named=True)]
        pbp_tasks = [fetch_game_data_with_retry(row["game_id"], session, semaphore, fetch_game_pbp) for row in df_recent.iter_rows(named=True)]
        
        box_results = await asyncio.gather(*box_tasks)
        pbp_results = await asyncio.gather(*pbp_tasks)
    
    valid_box = [df for df in box_results if df is not None]
    valid_pbp = [df for df in pbp_results if df is not None]
    
    if not valid_box:
        logger.warning("No boxscore data")
        return
    
    df_box = pl.concat(valid_box)
    df_pbp_raw = pl.concat(valid_pbp) if valid_pbp else pl.DataFrame(schema=PBP_RAW_SCHEMA)
    
    # Filter for the player
    df_player_box = df_box.filter(pl.col("player_name").str.contains(player_name))
    
    if df_player_box.is_empty():
        logger.warning(f"Player '{player_name}' not found in recent games")
        logger.info("\nSearching for similar names...")
        similar = df_box.filter(
            pl.col("player_name").str.to_lowercase().str.contains(player_name.lower())
        )["player_name"].unique().to_list()
        if similar:
            logger.info(f"Did you mean one of these? {similar}")
        return
    
    player_id = df_player_box["player_id"][0]
    logger.info(f"Found player ID: {player_id}")
    logger.info(f"Player: {df_player_box['player_name'][0]}")
    
    # Show boxscore data (note: boxscore API doesn't include goals!)
    logger.info(f"\nBoxscore data for {player_name} (Note: Boxscore API doesn't return goals/assists/shots):")
    print(df_player_box.select(["game_id", "player_name", "team", "toi", "shifts", "plus_minus", "decision", "ot_loss"]))
    
    # Now check PBP for this player
    if not df_pbp_raw.is_empty():
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing PBP events for player_id {player_id}...")
        logger.info('='*60)
        
        # Check ALL columns where this player could appear
        df_player_events = df_pbp_raw.filter(
            (pl.col("scoring_player_id") == player_id) |
            (pl.col("assist1_player_id") == player_id) |
            (pl.col("assist2_player_id") == player_id) |
            (pl.col("shooting_player_id") == player_id)
        )
        
        logger.info(f"Total events involving player: {len(df_player_events)}")
        
        # Goals specifically
        df_goals = df_player_events.filter(
            (pl.col("event_type") == "goal") &
            (pl.col("scoring_player_id") == player_id) &
            (pl.col("period_type") != "SO")
        )
        
        logger.info(f"\nGOALS BY GAME (from raw PBP):")
        if len(df_goals) > 0:
            goals_by_game = df_goals.group_by("game_id").agg(pl.count().alias("goals")).sort("game_id")
            print(goals_by_game)
            logger.info(f"\nTotal goals in raw PBP: {len(df_goals)}")
            
            logger.info(f"\nGoal events detail:")
            print(df_goals.select(["game_id", "event_type", "scoring_player_id", "situation_code"]).head(10))
        else:
            logger.warning("No goals found in raw PBP for this player!")
        
        # Now aggregate using our function
        logger.info(f"\n{'='*60}")
        logger.info("Running aggregate_pbp_stats()...")
        logger.info('='*60)
        df_pbp_agg = aggregate_pbp_stats(df_pbp_raw)
        
        df_player_pbp = df_pbp_agg.filter(pl.col("player_id") == player_id)
        
        if not df_player_pbp.is_empty():
            logger.info(f"\nAGGREGATED PBP STATS:")
            print(df_player_pbp.select([
                "game_id", "goals", "assists", "points", "shots", 
                "ppg", "ppa", "ppp", "game_winning_goal"
            ]))
            
            # Validate totals
            total_aggregated_goals = df_player_pbp["goals"].sum()
            logger.info(f"\nVALIDATION:")
            logger.info(f"  Raw PBP goals: {len(df_goals)}")
            logger.info(f"  Aggregated goals: {total_aggregated_goals}")
            
            if len(df_goals) == total_aggregated_goals:
                logger.info("  MATCH! Goals calculated correctly.")
            else:
                logger.error(f"  MISMATCH! Difference: {len(df_goals) - total_aggregated_goals}")
                logger.error("  This indicates a bug in the aggregation logic!")
            
            # Run integrity checks
            logger.info(f"\nINTEGRITY CHECKS:")
            issues = validate_data_integrity(df_player_pbp)
            if issues:
                for issue in issues:
                    logger.warning(f"  - {issue}")
            else:
                logger.info("  No data integrity issues found.")
        else:
            logger.error("Player not found in aggregated PBP!")
            logger.error("This means the unpivot or group_by is filtering out this player")
    else:
        logger.warning("No PBP data available")


# ============================================================================
# CHECKPOINT/RESUME LOGIC
# ============================================================================

def get_missing_game_ids(
    schedule_df: pl.DataFrame,
    season_dir: Path
) -> Tuple[Set[int], bool, bool]:
    """Determine which games need to be fetched."""
    all_game_ids = set(schedule_df["game_id"].to_list())
    
    pbp_file = season_dir / "raw_pbp_events.parquet"
    box_file = season_dir / "essential_boxscores.parquet"
    final_file = season_dir / "comprehensive_boxscores.parquet"
    
    has_raw_data = pbp_file.exists() and box_file.exists()
    has_final_data = final_file.exists()
    
    if not has_raw_data:
        return all_game_ids, False, False
    
    try:
        df_pbp_existing = pl.read_parquet(pbp_file)
        df_box_existing = pl.read_parquet(box_file)
        
        existing_pbp_games = set(df_pbp_existing["game_id"].unique().to_list())
        existing_box_games = set(df_box_existing["game_id"].unique().to_list())
        
        complete_games = existing_pbp_games & existing_box_games
        missing_games = all_game_ids - complete_games
        
        logger.info(f"Found {len(complete_games)} complete games, {len(missing_games)} missing")
        return missing_games, True, has_final_data
        
    except Exception as e:
        logger.warning(f"Error reading existing data: {e}")
        return all_game_ids, False, False


# ============================================================================
# ORCHESTRATION
# ============================================================================

async def run_backfill(season_ids: List[int], game_types: List[str] = ["R", "P"], test: bool = False):
    """Main backfill orchestrator."""
    logger.info(f"Starting backfill for seasons: {season_ids}")
    
    # Create one session to be re-used for all tasks in this backfill
    async with httpx.AsyncClient(timeout=30.0) as session:
        for season_id in season_ids:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing season {season_id}")
            logger.info('='*60)
            
            season_dir = get_season_output_dir(season_id)
            
            # --- 1. Get Active Teams for this specific season ---
            logger.info(f"Fetching active teams for {season_id}...")
            standings_end_date = await _get_season_standings_end_date(session, season_id)
            teams_for_season = await fetch_active_teams(session, standings_end_date)
            
            if not teams_for_season:
                logger.error(f"No active teams found for {season_id}. Skipping season.")
                continue
            
            # --- 2. Update Player Rosters for this season ---
            player_file = season_dir / "players.parquet"
            if not player_file.exists():
                logger.info(f"Player roster file missing for {season_id}, fetching...")
                await update_player_rosters(
                    session, season_id, teams_for_season, is_current=False
                )
            else:
                logger.info(f"Player roster file already exists for {season_id}.")

            # --- 3. Fetch or Load Schedule ---
            schedule_file = season_dir / "schedule.parquet"
            if schedule_file.exists():
                logger.info("Loading existing schedule")
                df_schedule = pl.read_parquet(schedule_file)
                
                if "home_days_rest" not in df_schedule.columns or "away_days_rest" not in df_schedule.columns:
                    logger.warning("Existing schedule missing 'days_rest'. Recalculating...")
                    df_schedule = add_days_rest_to_schedule(df_schedule)
                    df_schedule.write_parquet(schedule_file)
            else:
                logger.info("Fetching schedule from API")
                # Pass session and teams to the updated function
                df_schedule = await fetch_season_game_ids(
                    season_id, teams_for_season, session, game_types
                )
                if df_schedule.is_empty():
                    logger.warning(f"No games found for season {season_id}")
                    continue
                
                logger.info("Calculating days rest for schedule...")
                df_schedule = add_days_rest_to_schedule(df_schedule)
                df_schedule.write_parquet(schedule_file)
                logger.info(f"Saved schedule: {len(df_schedule)} games")

            # --- 4. Process Games (rest of the logic) ---
            today_str = date.today().strftime("%Y-%m-%d")
            df_schedule_played = df_schedule.filter(pl.col("date") < today_str)
            logger.info(f"Found {len(df_schedule_played)} games on or before yesterday")
            
            if test:
                logger.info("TEST MODE: Selecting one game per team")
                df_schedule_played = df_schedule_played.group_by("home_abbrev").head(1).head(32)
            
            missing_game_ids, has_raw_data, has_final_data = get_missing_game_ids(
                df_schedule_played, season_dir
            )
            
            if not missing_game_ids and has_final_data:
                logger.info("All data already complete for this season")
                continue
            
            if not missing_game_ids and has_raw_data:
                logger.info("Raw data complete, regenerating final aggregations")
                df_pbp_raw = pl.read_parquet(season_dir / "raw_pbp_events.parquet")
                df_box = pl.read_parquet(season_dir / "essential_boxscores.parquet")
                
                logger.info("Aggregating PBP stats...")
                df_pbp_agg = aggregate_pbp_stats(df_pbp_raw)
                
                logger.info("Combining data...")
                df_final = combine_data(df_box, df_pbp_agg, df_schedule)
                
                df_final.write_parquet(season_dir / "comprehensive_boxscores.parquet")
                logger.info(f"Saved final data: {len(df_final)} player-games")
                continue
            
            games_to_fetch = df_schedule_played.filter(pl.col("game_id").is_in(list(missing_game_ids)))
            logger.info(f"Fetching data for {len(games_to_fetch)} games")
            
            # Use the existing session
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            
            box_tasks = [fetch_game_data_with_retry(row["game_id"], session, semaphore, fetch_game_boxscore) for row in games_to_fetch.iter_rows(named=True)]
            pbp_tasks = [fetch_game_data_with_retry(row["game_id"], session, semaphore, fetch_game_pbp) for row in games_to_fetch.iter_rows(named=True)]
            
            logger.info("Fetching boxscores...")
            box_results = await asyncio.gather(*box_tasks)
            logger.info("Fetching play-by-play data...")
            pbp_results = await asyncio.gather(*pbp_tasks)
            
            valid_box = [df for df in box_results if df is not None]
            valid_pbp = [df for df in pbp_results if df is not None]
            
            if not valid_box:
                logger.warning("No boxscore data retrieved")
                continue
                
            df_box_new = pl.concat(valid_box)
            df_pbp_new = pl.concat(valid_pbp) if valid_pbp else pl.DataFrame(schema=PBP_RAW_SCHEMA)
            
            if has_raw_data:
                logger.info("Merging with existing data")
                df_box_existing = pl.read_parquet(season_dir / "essential_boxscores.parquet")
                df_box_new = pl.concat([df_box_existing, df_box_new]).unique(subset=["game_id", "player_id"])
                
                if not df_pbp_new.is_empty():
                    df_pbp_existing = pl.read_parquet(season_dir / "raw_pbp_events.parquet")
                    df_pbp_new = pl.concat([df_pbp_existing, df_pbp_new])
            
            df_box_new.write_parquet(season_dir / "essential_boxscores.parquet")
            logger.info(f"Saved boxscores: {len(df_box_new)} player-games")
            
            if not df_pbp_new.is_empty():
                df_pbp_new.write_parquet(season_dir / "raw_pbp_events.parquet")
                logger.info(f"Saved raw PBP: {len(df_pbp_new)} events")
                
            logger.info("Aggregating PBP stats...")
            df_pbp_agg = aggregate_pbp_stats(df_pbp_new)
            
            logger.info("Combining boxscore and PBP data...")
            df_final = combine_data(df_box_new, df_pbp_agg, df_schedule)
            
            df_final.write_parquet(season_dir / "comprehensive_boxscores.parquet")
            logger.info(f"Saved comprehensive boxscores: {len(df_final)} player-games")
            logger.info(f"\nSeason {season_id} complete!")


async def fetch_recent_games(session: httpx.AsyncClient, days_back: int = 7) -> pl.DataFrame:
    """Fetch completed games from the last N days."""
    logger.info(f"Fetching games from last {days_back} days")
    all_games = []
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_back - 1)
    
    # Remove the internal session, as one is now provided
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = f"https://api-web.nhle.com/v1/score/{date_str}"
        try:
            response = await session.get(url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            
            games_on_date = 0
            for game in data.get("games", []):
                if game.get("gameState") in ["OFF", "FINAL"]:
                    all_games.append({
                        "game_id": game.get("id"),
                        "date": date_str,
                        "season": game.get("season"),
                        "game_type": game.get("gameType"),
                        "away_abbrev": (game.get("awayTeam") or {}).get("abbrev"),
                        "away_score": (game.get("awayTeam") or {}).get("score"),
                        "home_abbrev": (game.get("homeTeam") or {}).get("abbrev"),
                        "home_score": (game.get("homeTeam") or {}).get("score"),
                    })
                    games_on_date += 1
            logger.info(f"  {date_str}: {games_on_date} games")
                    
        except Exception as e:
            logger.warning(f"Error fetching games for {date_str}: {e}")
        
        current_date += timedelta(days=1)
        await asyncio.sleep(0.1)
        
    if not all_games:
        logger.warning("No games found")
        return pl.DataFrame(schema=SCHEDULE_SCHEMA)
        
    df = pl.DataFrame(all_games, schema=SCHEDULE_SCHEMA)
    logger.info(f"Found {len(df)} completed games")
    return df


async def run_refresh(days_back: int = 7):
    """Refresh data for recent games."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting data refresh (last {days_back} days)")
    logger.info('='*60)
    
    # Create one session to be re-used for all tasks in this refresh
    async with httpx.AsyncClient(timeout=30.0) as session:
    
        # --- NEW: Update current player rosters ---
        try:
            logger.info("Updating CURRENT player rosters...")
            # 1. Get the current season ID
            current_season_id = await get_current_season_id(session)
            logger.info(f"Determined current season: {current_season_id}")
            
            # 2. Get current active teams (standings_date=None)
            teams = await fetch_active_teams(session, None)
            
            # 3. Update the data/players.parquet file
            if teams:
                await update_player_rosters(
                    session, current_season_id, teams, is_current=True
                )
            else:
                logger.error("No teams found, cannot update player rosters.")
                
        except Exception as e:
            logger.error(f"Failed to update player rosters: {e}")
        # --- END NEW ---
        
        # Pass the session to fetch_recent_games
        df_recent = await fetch_recent_games(session, days_back)
        
        if df_recent.is_empty():
            logger.warning("No recent games to refresh")
            return
            
        logger.info("Calculating days rest for recent games...")
        df_recent = add_days_rest_to_schedule(df_recent)
            
        seasons = df_recent["season"].unique().to_list()
        
        # Use the existing session for game data fetching
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        logger.info("Fetching boxscores...")
        box_tasks = [fetch_game_data_with_retry(row["game_id"], session, semaphore, fetch_game_boxscore) for row in df_recent.iter_rows(named=True)]
        box_res = await asyncio.gather(*box_tasks)
        
        logger.info("Fetching play-by-play data...")
        pbp_tasks = [fetch_game_data_with_retry(row["game_id"], session, semaphore, fetch_game_pbp) for row in df_recent.iter_rows(named=True)]
        pbp_res = await asyncio.gather(*pbp_tasks)
        
        valid_box = [df for df in box_res if df is not None]
        valid_pbp = [df for df in pbp_res if df is not None]
        
        if not valid_box:
            logger.warning("No boxscore data retrieved")
            return
            
        df_box_new = pl.concat(valid_box)
        df_pbp_new = pl.concat(valid_pbp) if valid_pbp else pl.DataFrame(schema=PBP_RAW_SCHEMA)
        
        logger.info(f"Retrieved {len(df_box_new)} player-game records")
        
        logger.info("Aggregating PBP stats...")
        df_pbp_agg = aggregate_pbp_stats(df_pbp_new)
        
        for season_id in seasons:
            logger.info(f"\nUpserting data for season {season_id}")
            season_dir = get_season_output_dir(season_id)
            
            season_games_to_refresh = df_recent.filter(pl.col("season") == season_id)["game_id"].to_list()
            
            full_schedule_file = season_dir / "schedule.parquet"
            if not full_schedule_file.exists():
                logger.error(f"Cannot refresh season {season_id}: schedule.parquet not found.")
                logger.error("Please run a full backfill for this season first.")
                continue
                
            logger.info(f"Loading full schedule for season {season_id} for context...")
            df_full_schedule = pl.read_parquet(full_schedule_file)
            
            df_combined_full_context = combine_data(df_box_new, df_pbp_agg, df_full_schedule)
            
            df_new_season = df_combined_full_context.filter(pl.col("game_id").is_in(season_games_to_refresh))
            
            main_file = season_dir / "comprehensive_boxscores.parquet"
            
            if main_file.exists():
                df_existing = pl.read_parquet(main_file)
                logger.info(f"Loaded {len(df_existing)} existing records")
                
                df_filtered = df_existing.filter(~pl.col("game_id").is_in(season_games_to_refresh))
                logger.info(f"Kept {len(df_filtered)} records from existing data")
                
                df_final = pl.concat([df_filtered, df_new_season]).sort("game_id", "player_id")
                
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_file = season_dir / f"archive_comprehensive_boxscores_{timestamp}.parquet"
                    logger.info(f"Archiving {main_file} to {archive_file}")
                    main_file.rename(archive_file)
                except Exception as e:
                    logger.error(f"Failed to archive existing file: {e}")
                
            else:
                logger.warning(f"No existing data found for season {season_id}, creating new file")
                df_final = df_new_season
                
            df_final.write_parquet(main_file)
            logger.info(f"Saved updated data to: {main_file} ({len(df_final)} records)")
        
        logger.info(f"\nRefresh complete!")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="NHL Data Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    backfill_parser = subparsers.add_parser("backfill", help="Run a full backfill")
    backfill_parser.add_argument("seasons", type=int, nargs="+", help="Season(s) to backfill (e.g., 20232024)")
    backfill_parser.add_argument("--test", action="store_true", help="Run in test mode")
    
    refresh_parser = subparsers.add_parser("refresh", help="Refresh recent data")
    refresh_parser.add_argument("--days", type=int, default=7, help="Days to look back (default: 7)")
    
    validate_parser = subparsers.add_parser("validate", help="Validate goals for a player")
    validate_parser.add_argument("player_name", type=str, help="Player name to validate (e.g., 'McDavid')")
    validate_parser.add_argument("--days", type=int, default=7, help="Days to look back (default: 7)")
    
    args = parser.parse_args()
    
    if args.command == "backfill":
        await run_backfill(args.seasons, test=args.test)
    elif args.command == "refresh":
        await run_refresh(days_back=args.days)
    elif args.command == "validate":
        await validate_goals_for_player(args.player_name, days_back=args.days)

if __name__ == "__main__":
    asyncio.run(main())
