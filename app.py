import streamlit as st
import polars as pl
import altair as alt
import datetime
import glob
from pathlib import Path
from zoneinfo import ZoneInfo
from dvp_algorithm import calculate_dvp_metrics, get_matchup_dvp, get_position_display


def get_hockey_date():
    """
    Returns the current 'hockey date' in Eastern Time.
    
    NHL games that run late (especially West Coast games) can end after midnight ET.
    This function considers games as 'today' until 5 AM ET the next morning.
    For example, at 2 AM ET on January 10th, the hockey date is still January 9th.
    """
    eastern = ZoneInfo("America/New_York")
    now_eastern = datetime.datetime.now(eastern)
    
    # If it's before 5 AM, consider it still the previous day's games
    if now_eastern.hour < 5:
        return (now_eastern - datetime.timedelta(days=1)).date()
    return now_eastern.date()

# --- 1. Refined Minimalist Styling with Mobile Responsiveness ---
def load_minimalist_style():
    """Injects custom CSS for a black theme with responsive adjustments."""
    st.markdown("""
        <style>
        /* Base */
        html, body, [class*="st-"], .main, .block-container {
            background-color: #000000 !important;
            color: #FAFAFA !important;
        }
        
        /* Sidebar Border */
        .st-emotion-cache-16txtl3 {
            border-right: 1px solid rgba(255, 193, 7, 0.3);
        }

        /* Highlight selected radio button */
        div[role="radiogroup"] > label:has(input:checked) {
            outline: 2px solid #FFC107 !important;
            outline-offset: 2px;
            border-radius: 5px;
            padding: 4px;
            background-color: rgba(255, 193, 7, 0.1);
        }
        
        /* Base Dataframe font */
        .stDataFrame {
            font-size: 13px;
        }

        /* --- MOBILE OPTIMIZATIONS --- */
        @media only screen and (max-width: 600px) {
            /* Reduce padding on mobile so content uses full width */
            .block-container {
                padding-top: 2rem !important;
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
            
            /* Make dataframe font smaller on mobile to fit more columns */
            .stDataFrame { font-size: 11px; }
            
            /* Adjust metric text size */
            div[data-testid="stMetricValue"] {
                font-size: 1.2rem !important;
            }
            
            /* Center images on mobile */
            div[data-testid="stImage"] {
                display: flex;
                justify-content: center;
            }

            /* Adjust chart padding */
            canvas {
                max-width: 100% !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)

# --- 2. Data Loading (Optimized) ---
@st.cache_data
def load_all_data():
    """Loads all parquet files with optimized Polars operations."""
    script_dir = Path(__file__).parent
    boxscore_files = glob.glob(str(script_dir / "data/**/comprehensive_boxscores.parquet"), recursive=True)
    schedule_files = glob.glob(str(script_dir / "data/**/schedule.parquet"), recursive=True)

    if not boxscore_files:
        st.error("No 'comprehensive_boxscores.parquet' files found in 'data/' directory.")
        return pl.DataFrame(), pl.DataFrame()

    try:
        boxscores_df = pl.concat([pl.scan_parquet(f) for f in boxscore_files]).collect()
    except Exception:
        st.error("Failed to load boxscore files.")
        return pl.DataFrame(), pl.DataFrame()

    schedule_df = pl.DataFrame()
    if schedule_files:
        try:
            schedule_df = pl.concat([
                pl.scan_parquet(f).select([
                    'game_id', 'date', 'season', 'game_type', 
                    'home_abbrev', 'away_abbrev', 'home_score', 'away_score',
                    'start_time', 'venue'
                ]) for f in schedule_files
            ]).unique(subset=["game_id"]).collect()
        except Exception:
            st.warning("Failed to load schedule data.")

    roster_file = script_dir / "data/players.parquet"
    if roster_file.exists():
        try:
            roster_names = pl.scan_parquet(roster_file).select([
                "player_id", "full_player_name"
            ]).rename({"full_player_name": "roster_name"}).drop_nulls("player_id").unique("player_id").collect()
            
            boxscores_df = boxscores_df.join(roster_names, on="player_id", how="left").with_columns(
                pl.coalesce(pl.col("roster_name"), pl.col("player_name")).alias("player_name")
            ).drop("roster_name")
        except Exception as e:
            st.warning(f"Could not standardize player names: {e}")

    numeric_cols = [
        'toi_minutes', 'shifts', 'goals', 'assists', 'points', 'plus_minus', 'pim', 
        'shots', 'shot_attempts', 'missed_shots', 'offensive_blocked_shots', 
        'blocked_shots', 'hits', 'giveaways', 'takeaways', 'faceoff_wins', 
        'faceoff_losses', 'ppg', 'ppa', 'ppp', 
        'shots_against', 'saves', 'goals_against'
    ]
    string_cols = ["player_name", "team", "opponent", "location", "position", "decision"]

    if boxscores_df["date"].dtype == pl.Utf8:
        boxscores_df = boxscores_df.with_columns(
            pl.col("date").str.to_date(format="%Y-%m-%d", strict=False)
        )

    cast_exprs = [
        pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0) 
        for col in numeric_cols if col in boxscores_df.columns
    ] + [
        pl.col(col).fill_null("Unknown") 
        for col in string_cols if col in boxscores_df.columns
    ]
    
    full_df = boxscores_df.with_columns(cast_exprs).drop_nulls("date")

    if not schedule_df.is_empty():
        schedule_df = schedule_df.with_columns(
            pl.col("date").str.to_date(format="%Y-%m-%d", strict=False)
        ).drop_nulls("date")
        
        full_df = full_df.join(
            schedule_df.select([
                'game_id', 'season', 'game_type', 
                'home_score', 'away_score', 'home_abbrev', 'away_abbrev'
            ]), 
            on="game_id", 
            how="left"
        )
        
        full_df = full_df.with_columns([
            pl.when((pl.col("location") == "home") & (pl.col("home_score") > pl.col("away_score")))
              .then(1)
              .when((pl.col("location") == "away") & (pl.col("away_score") > pl.col("home_score")))
              .then(1)
              .otherwise(0)
              .fill_null(0)
              .alias("win"),
            pl.col("season").fill_null(0).cast(pl.Int64),
            pl.col("game_type").fill_null(2).cast(pl.Int32)
        ])
    else:
        # Fallback if schedule is missing
        full_df = full_df.with_columns([
            pl.lit(0).alias("win"),
            pl.lit(2).alias("game_type"),
            pl.when(pl.col("date").dt.month() >= 9)
              .then((pl.col("date").dt.year() * 10000 + (pl.col("date").dt.year() + 1)).cast(pl.Int64))
              .otherwise(((pl.col("date").dt.year() - 1) * 10000 + pl.col("date").dt.year()).cast(pl.Int64))
              .alias("season")
        ])

    if "days_rest" not in full_df.columns:
        full_df = full_df.with_columns(pl.lit(99).alias("days_rest"))
    else:
        full_df = full_df.with_columns(pl.col("days_rest").fill_null(99).cast(pl.Int32))
    
    full_df = full_df.with_columns((pl.col("days_rest") == 0).alias("b2b"))

    return full_df.sort("date", descending=True), schedule_df

@st.cache_data
def load_player_rosters():
    """Loads the current player roster file."""
    script_dir = Path(__file__).parent
    roster_file = script_dir / "data/players.parquet"
    if not roster_file.exists():
        st.error("data/players.parquet not found.")
        return pl.DataFrame()
    try:
        roster_df = pl.scan_parquet(roster_file).select([
            "team_abbrev", "player_id", "full_player_name", "position_code", "headshot_url"
        ]).rename({"full_player_name": "player_name"}).collect()
        return roster_df
    except Exception as e:
        st.error(f"Failed to load player roster: {e}")
        return pl.DataFrame()

# --- 3. Charting Functions ---
def create_primary_chart(data: pl.DataFrame, y_col: str, title: str, threshold: float):
    """Creates the main Altair bar chart with responsive settings."""
    if data.is_empty():
        return
        
    max_y = data[y_col].max()
    y_domain_max = max_y + 1 if max_y is not None else 1
    
    chart_data = data.with_columns(
        (pl.col('date').dt.strftime('%m/%d') + " vs " + pl.col('opponent')).alias('display_date')
    )
    
    is_integer_stat = y_col in [
        'shots', 'shot_attempts', 'points', 'goals', 'assists', 
        'blocked_shots', 'hits', 'giveaways', 'takeaways', 'shifts',
        'saves', 'shots_against', 'goals_against', 'plus_minus', 'pim',
        'faceoff_wins', 'faceoff_losses', 'ppg', 'ppa', 'ppp', 
        'missed_shots', 'offensive_blocked_shots'
    ]
    stat_format = "d" if is_integer_stat else ".1f"
    
    tooltip = [
        alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"), 
        alt.Tooltip(f"{y_col}:Q", title=title, format=stat_format)
    ]

    base = alt.Chart(chart_data).properties(
        width='container', 
        height=350 
    )

    color_condition = alt.condition(
        f"datum['{y_col}'] >= {threshold}",
        alt.value("#2E8B57"), # Green
        alt.value("#CD5C5C")  # Red for bars (Keep this as is, only line changes)
    )

    text = base.mark_text(
        align='center',
        baseline='bottom',
        dx=0,
        dy=-4,
        fontSize=10,
        color="white" 
    ).encode(
        x=alt.X("display_date:N", sort=None, title=None, axis=alt.Axis(labels=True, ticks=True, domain=False, labelAngle=-90, labelFontSize=9)),
        y=alt.Y(f"{y_col}:Q", title=None, scale=alt.Scale(domain=[0, y_domain_max])),
        text=alt.Text(f"{y_col}:Q", format=stat_format),
        tooltip=tooltip
    )

    bars = base.mark_bar(opacity=0.9, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("display_date:N", sort=None, title=None, axis=alt.Axis(labels=True, ticks=True, domain=False, labelAngle=-90, labelFontSize=9)),
        y=alt.Y(f"{y_col}:Q", title=title, axis=alt.Axis(labels=True, ticks=True, domain=False, titleFontSize=11), scale=alt.Scale(domain=[0, y_domain_max])),
        color=color_condition,
        tooltip=tooltip
    )

    # --- REVISION: Set Threshold Line to Red ---
    thresh_line = alt.Chart(pl.DataFrame({'y': [threshold]})).mark_rule(
        strokeDash=[5,5],
        size=2,
        color="#FF0000"  # Explicitly Red
    ).encode(y='y:Q')

    chart = (bars + text + thresh_line).configure_view(
        strokeWidth=0
    ).properties(
        autosize=alt.AutoSizeParams(type='fit', contains='padding')
    )
    st.altair_chart(chart, use_container_width=True, theme="streamlit")

def create_secondary_chart(data: pl.DataFrame, y_col: str, title: str, avg_val: float):
    """Creates a secondary bar chart with proper height."""
    if data.is_empty():
        return
    
    chart_data = data.with_columns(
        pl.col('date').dt.strftime('%m/%d').alias('display_date')
    )
    
    is_integer_stat = y_col in [
        'shots', 'shot_attempts', 'points', 'goals', 'assists', 
        'blocked_shots', 'hits', 'giveaways', 'takeaways', 'shifts',
        'saves', 'shots_against', 'goals_against', 'plus_minus', 'pim',
        'faceoff_wins', 'faceoff_losses', 'ppg', 'ppa', 'ppp', 
        'missed_shots', 'offensive_blocked_shots'
    ]
    stat_format = "d" if is_integer_stat else ".1f"
    
    tooltip = [
        alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"), 
        alt.Tooltip(f"{y_col}:Q", format=stat_format)
    ]
    
    base = alt.Chart(chart_data).properties(
        width='container',
        height=140
    )
    
    bars = base.mark_bar(
        opacity=0.85,
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2
    ).encode(
        x=alt.X("display_date:N", sort=None, title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(f"{y_col}:Q", title=None, axis=alt.Axis(labels=True, ticks=True, domain=False, labelFontSize=9)),
        tooltip=tooltip
    )
    
    avg_line = alt.Chart(pl.DataFrame({'y': [avg_val]})).mark_rule(
        strokeDash=[3,3],
        size=1
    ).encode(y='y:Q')
    
    chart = (bars + avg_line).configure_view(
        strokeWidth=0
    ).properties(
        autosize=alt.AutoSizeParams(type='fit', contains='padding')
    )
    
    st.altair_chart(chart, use_container_width=True, theme="streamlit")

# --- 4. Compute DvP Metrics ---
@st.cache_data
def compute_dvp_data(full_df: pl.DataFrame, schedule_df: pl.DataFrame, stat_col: str, season_id: int):
    """Compute DvP metrics for the current season."""
    # Join full_df with schedule to get season info
    df_with_season = full_df.join(
        schedule_df.select(["game_id", "season"]),
        on="game_id",
        how="left"
    )
    
    # Calculate DvP metrics
    dvp_df = calculate_dvp_metrics(df_with_season, stat_col, season_id)
    return dvp_df

# --- 5. Compute Player Stats ---
@st.cache_data
def compute_player_stats_table(full_df: pl.DataFrame, player_roster_df: pl.DataFrame,
                               selected_date, teams: list, player_type: str,
                               stat_col: str, threshold: float):
    """Compute stats for all players in specified teams - optimized with Polars."""
    hockey_today = get_hockey_date()
    is_future_game = selected_date >= hockey_today
    
    # Get player list
    if is_future_game:
        player_list_df = player_roster_df.filter(pl.col("team_abbrev").is_in(teams))
    else:
        player_list_df = (
            full_df
            .filter((pl.col("date") == selected_date) & pl.col("team").is_in(teams))
            .select(["player_id", "player_name", "position", "team"])
            .rename({"position": "position_code", "team": "team_abbrev"})
            .unique(subset=["player_id"])
        )
    
    if player_list_df.is_empty():
        return pl.DataFrame()
    
    position_filter = pl.col("position_code") == "G" if player_type == "Goalie" else pl.col("position_code") != "G"
    player_list_df = player_list_df.filter(position_filter)
    
    if player_list_df.is_empty():
        return pl.DataFrame()
    
    player_ids = player_list_df.get_column("player_id").unique().to_list()
    
    # Get all historical data
    player_data = (
        full_df
        .filter(pl.col("player_id").is_in(player_ids) & (pl.col("date") <= selected_date))
        .sort("date", descending=True)
    )
    
    if player_data.is_empty():
        return pl.DataFrame(schema=player_list_df.schema)
    
    if selected_date.month >= 9:
        current_season = int(f"{selected_date.year}{selected_date.year + 1}")
    else:
        current_season = int(f"{selected_date.year - 1}{selected_date.year}")
    
    player_data = player_data.with_columns(
        (pl.col(stat_col) >= threshold).alias("is_hit")
    )
    
    # Season stats
    season_stats = (
        player_data
        .filter(pl.col("season") == current_season)
        .group_by("player_id")
        .agg([
            pl.len().alias("season_games"),
            pl.col("is_hit").sum().alias("season_hits")
        ])
        .with_columns((pl.col("season_hits") / pl.col("season_games") * 100).alias("season_pct"))
    )
    
    # H2H stats
    if len(teams) == 2:
        h2h_stats = (
            player_data
            .join(player_list_df.select(["player_id", "team_abbrev"]), on="player_id", how="left")
            .filter(
                ((pl.col("team_abbrev") == teams[0]) & (pl.col("opponent") == teams[1])) |
                ((pl.col("team_abbrev") == teams[1]) & (pl.col("opponent") == teams[0]))
            )
            .group_by("player_id")
            .agg([
                pl.len().alias("h2h_games"),
                pl.col("is_hit").sum().alias("h2h_hits")
            ])
            .with_columns((pl.col("h2h_hits") / pl.col("h2h_games") * 100).alias("h2h_pct"))
        )
    else:
        h2h_stats = pl.DataFrame()
    
    # Last N stats
    last_n_stats = (
        player_data
        .with_columns(pl.col("date").rank("dense", descending=True).over("player_id").alias("game_rank"))
        .group_by("player_id")
        .agg([
            pl.when(pl.col("game_rank") <= 5).then(pl.col("is_hit")).sum().alias("l5_hits"),
            pl.when(pl.col("game_rank") <= 5).then(1).sum().alias("l5_games"),
            pl.when(pl.col("game_rank") <= 10).then(pl.col("is_hit")).sum().alias("l10_hits"),
            pl.when(pl.col("game_rank") <= 10).then(1).sum().alias("l10_games"),
            pl.when(pl.col("game_rank") <= 20).then(pl.col("is_hit")).sum().alias("l20_hits"),
            pl.when(pl.col("game_rank") <= 20).then(1).sum().alias("l20_games"),
        ])
        .with_columns([
            (pl.col("l5_hits") / pl.col("l5_games") * 100).alias("l5_pct"),
            (pl.col("l10_hits") / pl.col("l10_games") * 100).alias("l10_pct"),
            (pl.col("l20_hits") / pl.col("l20_games") * 100).alias("l20_pct"),
        ])
    )
    
    # Location stats
    location_stats = (
        player_data
        .filter(pl.col("season") == current_season)
        .group_by(["player_id", "location"])
        .agg([
            pl.col("is_hit").sum().alias("hits"),
            pl.len().alias("games")
        ])
        .with_columns((pl.col("hits") / pl.col("games") * 100).alias("hit_pct"))
        .pivot(index="player_id", on="location", values=["hit_pct", "games"])
    )
    
    if "hit_pct_away" in location_stats.columns:
        location_stats = location_stats.rename({"hit_pct_away": "away_season_pct"})
    else:
        location_stats = location_stats.with_columns(pl.lit(None).alias("away_season_pct"))
    
    if "games_away" in location_stats.columns:
        location_stats = location_stats.rename({"games_away": "away_season_games"})
    else:
        location_stats = location_stats.with_columns(pl.lit(0).alias("away_season_games"))

    if "hit_pct_home" in location_stats.columns:
        location_stats = location_stats.rename({"hit_pct_home": "home_season_pct"})
    else:
        location_stats = location_stats.with_columns(pl.lit(None).alias("home_season_pct"))
        
    if "games_home" in location_stats.columns:
        location_stats = location_stats.rename({"games_home": "home_season_games"})
    else:
        location_stats = location_stats.with_columns(pl.lit(0).alias("home_season_games"))
    
    # Combine all stats
    result_df = (
        player_list_df
        .select(["player_id", "player_name", "team_abbrev", "position_code"])
        .join(season_stats, on="player_id", how="left")
        .join(last_n_stats, on="player_id", how="left")
        .join(location_stats, on="player_id", how="left")
    )
    
    if not h2h_stats.is_empty():
        result_df = result_df.join(h2h_stats, on="player_id", how="left")
    
    # Fill nulls for h2h if needed
    if "h2h_games" not in result_df.columns:
        result_df = result_df.with_columns([
            pl.lit(0).alias("h2h_games"),
            pl.lit(0.0).alias("h2h_pct")
        ])
    else:
        result_df = result_df.with_columns([
            pl.col("h2h_games").fill_null(0),
            pl.col("h2h_pct").fill_null(0.0)
        ])
        
    result_df = result_df.with_columns([
        pl.col(c).fill_null(0) for c in ["season_games", "l5_games", "l10_games", "l20_games", "away_season_games", "home_season_games"]
    ])
    
    # Format display
    display_df = result_df.with_columns([
        pl.lit(threshold).alias("Line"),
        pl.when(pl.col("season_games") > 0)
          .then(pl.format("{}% ({}g)", pl.col("season_pct").round(0).cast(pl.Int32), pl.col("season_games")))
          .otherwise(pl.lit("n/a")).alias("Season (Formatted)"),
        pl.when(pl.col("h2h_games") > 0)
          .then(pl.format("{}% ({}g)", pl.col("h2h_pct").round(0).cast(pl.Int32), pl.col("h2h_games")))
          .otherwise(pl.lit("n/a")).alias("H2H (Formatted)"),
        pl.when(pl.col("l5_games") >= 5)
          .then(pl.format("{}%", pl.col("l5_pct").round(0).cast(pl.Int32)))
          .otherwise(pl.lit("n/a")).alias("L5 (Formatted)"),
        pl.when(pl.col("l10_games") >= 10)
          .then(pl.format("{}%", pl.col("l10_pct").round(0).cast(pl.Int32)))
          .otherwise(pl.lit("n/a")).alias("L10 (Formatted)"),
        pl.when(pl.col("l20_games") >= 20)
          .then(pl.format("{}%", pl.col("l20_pct").round(0).cast(pl.Int32)))
          .otherwise(pl.lit("n/a")).alias("L20 (Formatted)"),
        pl.when((pl.col("away_season_pct").is_not_null()) & (pl.col("away_season_games") > 0))
          .then(pl.format("{}% ({}g)", pl.col("away_season_pct").round(0).cast(pl.Int32), pl.col("away_season_games")))
          .otherwise(pl.lit("n/a")).alias("Away (Season) (Formatted)"),
        pl.when((pl.col("home_season_pct").is_not_null()) & (pl.col("home_season_games") > 0))
          .then(pl.format("{}% ({}g)", pl.col("home_season_pct").round(0).cast(pl.Int32), pl.col("home_season_games")))
          .otherwise(pl.lit("n/a")).alias("Home (Season) (Formatted)"),
          
        pl.when(pl.col("season_games") > 0).then(pl.col("season_pct")).otherwise(pl.lit(None)).alias("Season %"),
        pl.when(pl.col("h2h_games") > 0).then(pl.col("h2h_pct")).otherwise(pl.lit(None)).alias("H2H %"),
        pl.when(pl.col("l5_games") >= 5).then(pl.col("l5_pct")).otherwise(pl.lit(None)).alias("L5 %"),
        pl.when(pl.col("l10_games") >= 10).then(pl.col("l10_pct")).otherwise(pl.lit(None)).alias("L10 %"),
        pl.when(pl.col("l20_games") >= 20).then(pl.col("l20_pct")).otherwise(pl.lit(None)).alias("L20 %"),
        pl.when((pl.col("away_season_pct").is_not_null()) & (pl.col("away_season_games") > 0)).then(pl.col("away_season_pct")).otherwise(pl.lit(None)).alias("Away %"),
        pl.when((pl.col("home_season_pct").is_not_null()) & (pl.col("home_season_games") > 0)).then(pl.col("home_season_pct")).otherwise(pl.lit(None)).alias("Home %"),
    ]).select([
        "player_id", "player_name", "team_abbrev", "position_code", "Line",
        "Season (Formatted)", "Season %", 
        "H2H (Formatted)", "H2H %", 
        "L5 (Formatted)", "L5 %",
        "L10 (Formatted)", "L10 %", 
        "L20 (Formatted)", "L20 %", 
        "Away (Season) (Formatted)", "Away %", 
        "Home (Season) (Formatted)", "Home %"
    ]).rename({
        "player_name": "Player",
        "team_abbrev": "Team",
        "position_code": "Pos"
    })
    
    return display_df

# --- 5. Main Application ---
def render_game_selection_table(schedule_df, selected_date):
    """Renders a table of games for the selected date to allow selection."""
    st.header(f"Games for {selected_date.strftime('%Y-%m-%d')}")
    
    daily_games = schedule_df.filter(pl.col("date") == selected_date)
    
    if daily_games.is_empty():
        st.info("No games scheduled for this date.")
        return

    # Process for display
    display_df = daily_games.with_columns([
        pl.col("start_time").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ").dt.convert_time_zone("America/New_York").dt.strftime("%I:%M %p").alias("Time (EST)"),
        (pl.col("away_abbrev") + " @ " + pl.col("home_abbrev")).alias("Matchup"),
        pl.col("venue").fill_null("Unknown").alias("Venue")
    ]).sort("start_time").select([
        "Time (EST)", "home_abbrev", "away_abbrev", "Venue", "Matchup"
    ]).rename({
        "home_abbrev": "Home",
        "away_abbrev": "Away"
    })

    event = st.dataframe(
        display_df.select(["Time (EST)", "Home", "Away", "Venue"]),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    if event.selection.rows:
        selected_row_idx = event.selection.rows[0]
        selected_matchup = display_df.row(selected_row_idx, named=True)["Matchup"]
        st.session_state.selected_matchup = selected_matchup
        st.rerun()

def render_sidebar(full_df, schedule_df, player_roster_df):
    """Renders the sidebar controls."""
    st.sidebar.header("Controls")

    # 1. Date
    selected_date = st.sidebar.date_input("Date", st.session_state.get("selected_date", get_hockey_date()))
    st.session_state.selected_date = selected_date

    # 2. Matchup (sorted by game_id)
    schedule_for_date = schedule_df.filter(pl.col("date") == selected_date)
    matchup_options = ["Select matchup..."]
    if not schedule_for_date.is_empty():
        matchups = (schedule_for_date
                    .sort("game_id")
                    .select((pl.col("away_abbrev") + " @ " + pl.col("home_abbrev")).alias("matchup"))
                    .unique(maintain_order=True)
                    .get_column("matchup")
                    .to_list())
        matchup_options.extend(matchups)
    
    selected_matchup_index = 0
    if "selected_matchup" in st.session_state:
        try:
            selected_matchup_index = matchup_options.index(st.session_state.selected_matchup)
        except ValueError:
            st.session_state.selected_matchup = None
            
    selected_matchup = st.sidebar.selectbox("Matchup", matchup_options, index=selected_matchup_index)
    st.session_state.selected_matchup = selected_matchup
    
    if selected_matchup == "Select matchup...":
        st.session_state.selected_player_id = None
        return selected_date, None, None, None, None, None, None, None
    
    # 3. Player Type
    player_type = st.sidebar.selectbox("Player Type", ["Skater", "Goalie"], index=0)
    
    # 4. Prop
    if player_type == "Goalie":
        stat_options = {"Saves": "saves", "Shots Against": "shots_against", "Goals Against": "goals_against"}
        default_stat = "Saves"
    else:
        stat_options = {
            "Shots on Goal": "shots", "Shot Attempts": "shot_attempts", "Points": "points",
            "Goals": "goals", "Assists": "assists", "Time on Ice": "toi_minutes",
            "Blocked Shots": "blocked_shots", "Hits": "hits",
            "Plus/Minus": "plus_minus",
        }
        default_stat = "Shots on Goal"
    
    selected_stat_label = st.sidebar.selectbox("Prop", list(stat_options.keys()), 
                                               index=list(stat_options.keys()).index(default_stat))
    selected_stat = stat_options[selected_stat_label]
    
    # 5. Threshold
    if player_type == "Goalie":
        default_val = {"goals_against": 2.5, "saves": 24.5, "shots_against": 25.5}.get(selected_stat, 2.0)
    else:
        default_val = 1.5 if selected_stat == 'shots' else (1.5 if selected_stat == 'blocked_shots' else 0.5)
    
    threshold = st.sidebar.number_input("Threshold", min_value=0.5, step=1.0, value=default_val)
    
    # 6. Player (build list)
    hockey_today = get_hockey_date()
    is_future_game = selected_date >= hockey_today
    away_team, home_team = selected_matchup.split(" @ ")
    
    if is_future_game:
        temp_df = player_roster_df.filter(pl.col("team_abbrev").is_in([home_team, away_team]))
    else:
        temp_df = full_df.filter(
            (pl.col("date") == selected_date) &
            (((pl.col("opponent") == away_team) & (pl.col("team") == home_team)) |
             ((pl.col("opponent") == home_team) & (pl.col("team") == away_team)))
        )
        if not temp_df.is_empty():
            temp_df = temp_df.select(["player_id", "player_name", "position"]).rename({"position": "position_code"})

    player_list_df = pl.DataFrame()
    if not temp_df.is_empty():
        if player_type == "Goalie":
            player_list_df = temp_df.filter(pl.col("position_code") == "G")
        else:
            player_list_df = temp_df.filter(pl.col("position_code") != "G")
    
    player_options = [("", "Select player...")]
    if not player_list_df.is_empty():
        player_list_df = player_list_df.unique(subset=["player_id"]).sort("player_name")
        for row in player_list_df.iter_rows(named=True):
            player_options.append((row["player_id"], row["player_name"]))
    
    # Check session state
    current_player_id = st.session_state.get("selected_player_id")
    default_index = 0
    if current_player_id:
        valid_player_ids = [p[0] for p in player_options]
        if current_player_id in valid_player_ids:
            default_index = valid_player_ids.index(current_player_id)
        else:
            st.session_state.selected_player_id = None
    
    selected_player_id = st.sidebar.selectbox(
        "Player",
        options=[p[0] for p in player_options],
        index=default_index,
        format_func=lambda pid: next((name for p_id, name in player_options if p_id == pid), "Select player...")
    )
    
    if selected_player_id != current_player_id:
        st.session_state.selected_player_id = selected_player_id if selected_player_id else None
    
    player_pos_map = {}
    if not player_list_df.is_empty():
        player_pos_map = dict(zip(player_list_df["player_id"], player_list_df["position_code"]))
    
    return (selected_date, selected_matchup, player_type, selected_stat, threshold, 
            selected_stat_label, selected_player_id, player_pos_map.get(selected_player_id))

def render_matchup_overview(full_df, player_roster_df, schedule_df, selected_date, selected_matchup,
                           player_type, stat_col, threshold, stat_label):
    """Render matchup overview table with Mobile optimization and DvP."""
    st.header("Matchup Overview")
    
    # 1. Mobile View Toggle
    col_head, col_opt = st.columns([3, 1])
    with col_head:
        st.subheader(f"{stat_label} - Line: {threshold}")
    with col_opt:
        mobile_view = st.checkbox("Mobile View", value=True, help="Reduces columns for small screens")

    away_team, home_team = selected_matchup.split(" @ ")
    teams = [home_team, away_team]
    
    # Compute DvP data for current season
    current_season = 20242025  # This should be dynamic based on selected_date
    with st.spinner("Computing DvP metrics..."):
        dvp_df = compute_dvp_data(full_df, schedule_df, stat_col, current_season)
    
    with st.spinner("Computing player statistics..."):
        stats_df = compute_player_stats_table(
            full_df, player_roster_df, selected_date, teams, player_type, stat_col, threshold
        )
    
    if stats_df.is_empty():
        st.warning("No player data available for this matchup.")
        return
    
    # Add DvP information to stats_df
    # For each player, get the DvP info for their position vs the opponent
    dvp_info = []
    for row in stats_df.iter_rows(named=True):
        player_team = row["Team"]
        player_pos = row["Pos"]
        opponent_team = away_team if player_team == home_team else home_team
        
        # Get DvP info
        matchup_dvp = get_matchup_dvp(dvp_df, player_team, opponent_team, player_pos, stat_col)
        dvp_info.append({
            "Player": row["Player"],
            "Team": row["Team"],
            "Pos": row["Pos"],
            "DvP": matchup_dvp["dvp_formatted"],
            "DvP_Rank": matchup_dvp["dvp_rank"]
        })
    
    # Convert DvP info to DataFrame and join back
    dvp_additions = pl.DataFrame(dvp_info)
    stats_df = stats_df.join(
        dvp_additions.select(["Player", "DvP", "DvP_Rank"]),
        on="Player",
        how="left"
    )
    
    stats_df = stats_df.sort("L10 %", descending=True, nulls_last=True)
    
    # 2. Dynamic Column Configuration
    base_config = {
        "Player": st.column_config.TextColumn("Player", width="small" if mobile_view else "medium"),
        "Team": st.column_config.TextColumn("Team", width="small"),
        "Pos": st.column_config.TextColumn("Pos", width="small"),
        "Line": st.column_config.NumberColumn("Line", format="%.1f"),
        "DvP": st.column_config.TextColumn("DvP", width="small", help="Defense vs Position - how opponent defense affects this position"),
        "DvP_Rank": st.column_config.NumberColumn("DvP Rank", width="small", format="%d", help="Rank of opponent defense vs this position (1=best, 32=worst)"),
        "player_id": None
    }

    l_stats = {
        "L5 (Formatted)": st.column_config.TextColumn("L5"),
        "L5 %": st.column_config.NumberColumn("L5 %", format="%.0f%%"),
        "L10 (Formatted)": st.column_config.TextColumn("L10"),
        "L10 %": st.column_config.NumberColumn("L10 %", format="%.0f%%"),
        "L20 (Formatted)": st.column_config.TextColumn("L20"),
        "L20 %": st.column_config.NumberColumn("L20 %", format="%.0f%%"),
    }
    
    season_stats = {
        "Season (Formatted)": st.column_config.TextColumn("Season"),
        "Season %": st.column_config.NumberColumn("Season %", format="%.0f%%"),
        "H2H (Formatted)": st.column_config.TextColumn("H2H"),
        "H2H %": st.column_config.NumberColumn("H2H %", format="%.0f%%"),
        "Away (Season) (Formatted)": st.column_config.TextColumn("Away"),
        "Away %": st.column_config.NumberColumn("Away %", format="%.0f%%"),
        "Home (Season) (Formatted)": st.column_config.TextColumn("Home"),
        "Home %": st.column_config.NumberColumn("Home %", format="%.0f%%"),
    }

    if mobile_view:
        # Reduced columns for mobile (include DvP)
        visible_cols = ["player_id", "Player", "Team", "Pos", "Line", "DvP", "DvP_Rank", "L5 (Formatted)", "L10 (Formatted)", "Season %"]
        final_config = {**base_config, **l_stats, "Season %": st.column_config.NumberColumn("Szn %", format="%.0f%%")}
    else:
        # All columns for desktop (include DvP)
        visible_cols = stats_df.columns
        final_config = {**base_config, **l_stats, **season_stats}

    # 3. Render Dataframe with container width
    event = st.dataframe(
        stats_df.select([c for c in visible_cols if c in stats_df.columns]),
        width="stretch",
        height=600,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config=final_config
    )
    
    if event.selection.rows:
        selected_row_idx = event.selection.rows[0]
        selected_player_id = stats_df.row(selected_row_idx, named=True)["player_id"]
        st.session_state.selected_player_id = selected_player_id
        st.rerun()

def render_main_content(full_df, schedule_df, player_roster_df, selected_date, 
                        selected_player_id, player_position, selected_stat, threshold,
                        stat_label, selected_matchup):
    """Renders the main content area of the application."""
    # Get player info
    player_meta = player_roster_df.filter(pl.col("player_id") == selected_player_id)
    if player_meta.is_empty():
        player_meta_hist = full_df.filter(pl.col("player_id") == selected_player_id).select(["player_id", "player_name", "team"]).unique("player_id")
        if not player_meta_hist.is_empty():
            selected_player_name = player_meta_hist.item(0, "player_name")
            player_team = player_meta_hist.item(0, "team")
            player_headshot = None
        else:
            st.error("Player not found.")
            st.stop()
    else:
        selected_player_name = player_meta.item(0, "player_name")
        player_headshot = player_meta.item(0, "headshot_url")
        player_team = player_meta.item(0, "team_abbrev")

    # --- Player Header ---
    opponent_team, game_location = None, None
    away_team, home_team = selected_matchup.split(" @ ")

    if player_team == home_team:
        opponent_team = away_team
        game_location = "vs"
    elif player_team == away_team:
        opponent_team = home_team
        game_location = "@"
    
    col_img, col_info = st.columns([1, 5])
    with col_img:
        st.image(player_headshot if player_headshot and "nhle.com" in player_headshot else "https://placehold.co/120x120/eee/ccc?text=?", width=120)
    with col_info:
        st.title(selected_player_name)
        if player_team and opponent_team:
            st.subheader(f"{player_team} {game_location} {opponent_team}")
        st.subheader(f"{stat_label} - Line: {threshold}")

    st.divider()

    # --- Filter Bar ---
    if selected_date.month >= 9:
        current_season_id = int(f"{selected_date.year}{selected_date.year + 1}")
        prev_season_id = int(f"{selected_date.year - 1}{selected_date.year}")
        curr_season_label = f"{str(selected_date.year)[-2:]}/{str(selected_date.year + 1)[-2:]}"
        prev_season_label = f"{str(selected_date.year - 1)[-2:]}/{str(selected_date.year)[-2:]}"
    else:
        current_season_id = int(f"{selected_date.year - 1}{selected_date.year}")
        prev_season_id = int(f"{selected_date.year - 2}{selected_date.year - 1}")
        curr_season_label = f"{str(selected_date.year - 1)[-2:]}/{str(selected_date.year)[-2:]}"
        prev_season_label = f"{str(selected_date.year - 2)[-2:]}/{str(selected_date.year - 1)[-2:]}"

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        filter_options = [
            f"{curr_season_label} (This Season)",
            "H2H",
            "L5",
            "L10",
            "L20",
            "L30",
            f"{prev_season_label} (Last Season)"
        ]
        game_filter = st.selectbox("Game Filter", filter_options, index=0)
        
    with f_col2:
        location_filter = st.selectbox("Location", ["Any", "Home", "Away"], index=0)
    with f_col3:
        rest_days_filter = st.selectbox("Rest Days", ["Any", "0", "1", "2", "3+"], index=0)
        
    # --- Data Filtering ---
    analysis_df = full_df.filter(pl.col("player_id") == selected_player_id).sort("date", descending=True)
    
    if "This Season" in game_filter:
        analysis_df = analysis_df.filter(pl.col("season") == current_season_id)
    elif "Last Season" in game_filter:
        analysis_df = analysis_df.filter(pl.col("season") == prev_season_id)
    elif game_filter == "H2H":
        if opponent_team:
            analysis_df = analysis_df.filter(pl.col("opponent") == opponent_team)
        else:
            st.warning("No opponent determined for H2H filter.")
    elif game_filter.startswith("L"):
        try:
            limit = int(game_filter[1:])
            analysis_df = analysis_df.head(limit)
        except:
            pass

    attr_filter_expr = []
    if location_filter != "Any":
        attr_filter_expr.append(pl.col("location") == location_filter.lower())
    if rest_days_filter != "Any":
        attr_filter_expr.append(pl.col("days_rest") == int(rest_days_filter) if rest_days_filter != "3+" else pl.col("days_rest") >= 3)
    
    if attr_filter_expr:
        analysis_df = analysis_df.filter(pl.all_horizontal(attr_filter_expr))
    
    if analysis_df.is_empty():
        st.warning(f"No data found for {selected_player_name} with the selected filters.")
        st.stop()

    # --- KPI Section ---
    st.divider()
    def calc_hit_rate(df: pl.DataFrame, stat_col: str, thresh: float) -> tuple[float, int]:
        if df.is_empty(): 
            return 0.0, 0
        hit_count = df.filter(pl.col(stat_col) >= thresh).height
        return (hit_count / df.height) * 100, df.height

    if selected_date.month >= 9:
        current_season = int(f"{selected_date.year}{selected_date.year + 1}")
    else:
        current_season = int(f"{selected_date.year - 1}{selected_date.year}")
        
    season_df = analysis_df.filter(pl.col("season") == current_season)
    
    season_hit, season_games = calc_hit_rate(season_df, selected_stat, threshold)
    h2h_df = analysis_df.filter(pl.col("opponent") == opponent_team) if opponent_team else pl.DataFrame()
    h2h_hit, h2h_games = calc_hit_rate(h2h_df, selected_stat, threshold)
    l5_hit, l5_games = calc_hit_rate(analysis_df.head(5), selected_stat, threshold)
    l10_hit, l10_games = calc_hit_rate(analysis_df.head(10), selected_stat, threshold)
    
    # Responsive grid for KPIs (2 sets of 2 cols stacks better on mobile)
    kpi_row1 = st.columns(2)
    kpi_row1[0].metric(f"Season ({season_games} g)", f"{season_hit:.0f}%")
    kpi_row1[1].metric(f"vs {opponent_team} ({h2h_games} g)" if opponent_team else "H2H", 
                       f"{h2h_hit:.0f}%" if h2h_games > 0 else "n/a")
    
    kpi_row2 = st.columns(2)
    kpi_row2[0].metric(f"Last 5 ({l5_games} g)", f"{l5_hit:.0f}%")
    kpi_row2[1].metric(f"Last 10 ({l10_games} g)", f"{l10_hit:.0f}%")

    # --- Charting Section ---
    st.divider()
    chart_df = analysis_df.head(25).sort("date")
    create_primary_chart(chart_df, selected_stat, selected_stat.replace('_', ' ').title(), threshold)

    # --- Secondary Stats ---
    if player_position != "G":
        st.divider()
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            avg_val = chart_df['shot_attempts'].mean()
            st.metric("Avg. Shot Attempts", f"{avg_val:.1f}")
            create_secondary_chart(chart_df, "shot_attempts", "Shot Attempts", avg_val)
        with s_col2:
            avg_val = chart_df['missed_shots'].mean()
            st.metric("Avg. Missed Shots", f"{avg_val:.1f}")
            create_secondary_chart(chart_df, "missed_shots", "Missed Shots", avg_val)

        s_col3, s_col4 = st.columns(2)
        with s_col3:
            avg_val = chart_df['toi_minutes'].mean()
            st.metric("Avg. Time on Ice (Mins)", f"{avg_val:.1f}")
            create_secondary_chart(chart_df, "toi_minutes", "Time on Ice", avg_val)
        with s_col4:
            avg_val = chart_df['shifts'].mean()
            st.metric("Avg. Shifts", f"{avg_val:.1f}")
            create_secondary_chart(chart_df, "shifts", "Shifts", avg_val)

    # --- Game Logs ---
    st.divider()
    with st.expander("View Game Logs"):
        st.dataframe(analysis_df.head(50), use_container_width=True, hide_index=True)

def main():
    st.set_page_config(layout="wide", page_title="NHL Dashboard")
    load_minimalist_style()

    # Initialize session state
    if 'selected_player_id' not in st.session_state:
        st.session_state.selected_player_id = None
    if 'selected_matchup' not in st.session_state:
        st.session_state.selected_matchup = None

    # --- Data Loading ---
    full_df, schedule_df = load_all_data()
    player_roster_df = load_player_rosters()

    if full_df.is_empty() or player_roster_df.is_empty():
        st.error("Failed to load necessary data. Please check file paths and data integrity.")
        st.stop()

    # --- Sidebar ---
    sidebar_result = render_sidebar(full_df, schedule_df, player_roster_df)
    
    # Check if matchup selected
    if sidebar_result[1] is None or sidebar_result[1] == "Select matchup...":
        render_game_selection_table(schedule_df, sidebar_result[0])
        st.session_state.selected_player_id = None
        return
    
    (selected_date, selected_matchup, player_type, stat, 
     thresh, stat_label, selected_player_id, player_pos) = sidebar_result
    
    # Check if player is selected
    if selected_player_id:
        render_main_content(
            full_df, schedule_df, player_roster_df, 
            selected_date, selected_player_id, 
            player_pos, stat, thresh, stat_label,
            selected_matchup
        )
        
        if st.button("← Back to Matchup Overview"):
            st.session_state.selected_player_id = None
            st.rerun()
    else:
        render_matchup_overview(
            full_df, player_roster_df, schedule_df,
            selected_date, selected_matchup, player_type,
            stat, thresh, stat_label
        )

if __name__ == "__main__":
    main()
