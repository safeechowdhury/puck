# NHL Data Extraction & Dashboard

A local analytics platform for NHL player data, featuring an asynchronous data pipeline and an interactive Streamlit dashboard.

## Features

### Data Pipeline (`nhl_pipeline.py`)
- **Asynchronous Fetching**: High-performance data retrieval using `httpx` and `asyncio`.
- **Polars Processing**: Fast data manipulation and aggregation.
- **Structured Logging**: JSON-formatted logs for better observability.
- **Data Integrity**: Automated checks for TOI anomalies and shift consistency.
- **Commands**:
    - `backfill <seasons>`: Fetch full historical data.
    - `refresh [--days <N>]`: Update with recent games.
    - `validate <player_name>`: Diagnostic tool for data accuracy.

### Dashboard (`app.py`)
- **Interactive Visualizations**: Altair charts for player performance trends.
- **Advanced Filtering**:
    - **Game Filter**: Select specific scopes like "This Season", "Last Season", "H2H", or "Last 5/10/20/30 Games".
    - **Context Filters**: Filter by Location (Home/Away) and Rest Days.
- **Multi-Stat Analysis**: Analyze Goals, Assists, Points, Shots, Hits, Blocked Shots, and **Plus/Minus**.
- **Matchup Overview**: Compare player stats for a specific game matchup.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd nhl_data_extraction
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Pipeline

Run the pipeline to fetch data:

```bash
# Backfill specific seasons (e.g., 2023-2024 and 2024-2025)
python nhl_pipeline.py backfill 20232024 20242025

# Refresh data for the last 7 days
python nhl_pipeline.py refresh --days 7

# Validate data for a specific player
python nhl_pipeline.py validate "McDavid"
```

### Dashboard

Launch the dashboard:

```bash
streamlit run app.py
```

## Project Structure

- `nhl_pipeline.py`: Main data extraction script.
- `app.py`: Streamlit dashboard application.
- `data/`: Directory storing Parquet data files.
- `requirements.txt`: Python dependencies.
- `roadmap.md`: Future development plans.
