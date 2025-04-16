# WazePolice Scraper

A Python tool for extracting police reports and other alert types from Waze by directly accessing the internal API endpoints used by the Waze live map.

Version: 3.0.0 
Author: github - @zudsniper

## Features

- Extract police reports and other alert types from Waze in real-time
- Support for custom geographic boundaries
- Multiple output formats (JSON, GeoJSON, CSV)
- Continuous polling with configurable intervals
- Time-limited operation with "99d 99h 99m 99s" format
- Custom alert type filtering
- Timestamps added to output files automatically
- Full property preservation mode (--full flag)
- Session interruption handling and resume capability
- Automatic checkpointing with automatic resume from latest checkpoint

## Installation

### Requirements

- Python 3.6 or higher
- Required packages: httpx, pandas, geojson, typer, loguru, jsonschema

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/zudsniper/wazepolice.git
   cd wazepolice
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   Alternatively, install required packages directly:
   ```
   pip install httpx pandas geojson typer loguru jsonschema
   ```

3. Make the script executable (Unix/Linux/macOS):
   ```
   chmod +x wazepolice.py
   ```

## Usage

### Basic Usage

```bash
# Automatically resume from the latest checkpoint, or start new if none exists
python wazepolice.py

# Force a new scraping session (ignoring any existing checkpoints)
python wazepolice.py --new

# One-time extraction with default bounds (Atlanta to Nashville)
python wazepolice.py --interval 0

# Continuous extraction with 5 minute intervals
python wazepolice.py --interval 300

# Custom output file
python wazepolice.py --output police_data.json

# Custom geographic bounds (lat1,lon1,lat2,lon2)
python wazepolice.py --bounds 34.0522,-118.2437,34.2522,-118.0437
```

### Advanced Options

```bash
# Run for a specific duration (1 day, 2 hours, 30 minutes, 15 seconds)
python wazepolice.py --runtime "1d 2h 30m 15s"

# Filter for multiple alert types (default is POLICE only)
python wazepolice.py --filter "POLICE,ACCIDENT,HAZARD"

# Preserve all alert properties (full mode)
python wazepolice.py --full

# Specify output format (json, geojson, or csv)
python wazepolice.py --format geojson

# Save raw API response data
python wazepolice.py --raw raw_data.json

# Use custom JSON schema file
python wazepolice.py --schema my_schema.json

# Enable automatic checkpointing every 60 seconds
python wazepolice.py --checkpoint 60

# Resume from a specific session checkpoint
python wazepolice.py --resume session_checkpoint_1234567890.json
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--bounds` | Bounding box as lat1,lon1,lat2,lon2 | 33.7490,-84.3880,36.1627,-86.7816 |
| `--output` | Output file path | police.json |
| `--interval` | Polling interval in seconds (0 for one-time) | 600 |
| `--format`, `-f` | Force output format: 'json', 'geojson', or 'csv' | json |
| `--schema` | Path to JSON schema file | ./schema/wazedata.json |
| `--raw` | Save raw schema-validated data to this file | None |
| `--runtime` | Maximum runtime in format '99d 99h 99m 99s' | None |
| `--filter` | Comma-separated list of alert types to extract | POLICE |
| `--full` | Preserve all alert properties instead of specific fields | False |
| `--checkpoint` | Interval in seconds to save checkpoint data for potential resume | 300 |
| `--resume` | Resume scraping from a specific saved session file | None |
| `--new` | Force start a new session even if checkpoint files exist | False |
| `--help` | Show help information | |

## Output Format

The scraper can output data in three formats:

### JSON (default)

Standard mode extracts specific fields:
```json
[
  {
    "id": "abc123",
    "type": "POLICE",
    "subtype": "VISIBLE",
    "lat": 34.0522,
    "lon": -118.2437,
    "reported_by": "userXYZ",
    "confidence": 5,
    "reliability": 4,
    "report_rating": 3,
    "timestamp": "2023-04-01T12:34:56.789012",
    "pub_millis": 1680353696789
  }
]
```

In full mode (--full), all properties from the Waze API are preserved, and the filename includes "_full":
```
police_full_1680353696.json
```

### GeoJSON

GeoJSON-formatted data for use with mapping applications.

### CSV

Comma-separated values format for easy import into spreadsheet applications.

## File Naming

Output files are automatically appended with a Unix epoch timestamp:
```
police_1680353696.json
```

When using full mode (--full), "_full" is added to the filename:
```
police_full_1680353696.json
```

## Interruption Handling and Resume

WazePolice supports graceful handling of interruptions and will automatically resume from the latest checkpoint:

### Automatic Resume

By default, the script will:
1. Look for the most recent checkpoint file in the current directory
2. If found, resume from that checkpoint automatically
3. If no checkpoint files exist, start a new session

If you want to force a new session (ignoring any checkpoints), use:
```bash
python wazepolice.py --new
```

To resume from a specific checkpoint rather than the most recent one:
```bash
python wazepolice.py --resume session_checkpoint_1234567890.json
```

### Checkpoints

The script automatically saves checkpoints at regular intervals (default: every 5 minutes). You can control this interval with the `--checkpoint` option.

```bash
# Save checkpoints every 2 minutes
python wazepolice.py --checkpoint 120

# Disable checkpointing
python wazepolice.py --checkpoint 0
```

When the script is interrupted (e.g., by pressing Ctrl+C), it will save a final checkpoint before exiting.

### Resuming Details

When resuming from a checkpoint:
- Original bounds and alert types from the checkpoint file are used
- If the original session had a runtime limit, the remaining time is calculated automatically
- The session statistics are preserved, enabling accurate tracking of total alerts found

### Checkpoint Files

Checkpoint files are saved with these naming patterns:
- Regular checkpoints: `session_checkpoint_[timestamp].json`
- Final session stats: `session_[timestamp].json`
- Error checkpoints: `session_error_[timestamp].json`

## Alert Types

Common alert types you can use with `--filter`:

- `POLICE` - Police reports
- `ACCIDENT` - Accident reports
- `HAZARD` - Road hazards
- `JAM` - Traffic jams
- `ROAD_CLOSED` - Road closures
- `CONSTRUCTION` - Construction areas

## Examples

### Extract all alert types with all properties

```bash
python wazepolice.py --filter "POLICE,ACCIDENT,HAZARD,JAM,ROAD_CLOSED,CONSTRUCTION" --full
```

### Monitor a specific area for police reports for 24 hours

```bash
python wazepolice.py --bounds 34.0522,-118.2437,34.2522,-118.0437 --interval 300 --runtime "24h"
```

### Extract all alert types and save as GeoJSON

```bash
python wazepolice.py --filter "POLICE,ACCIDENT,HAZARD,JAM,ROAD_CLOSED" --format geojson
```

### One-time extraction to CSV

```bash
python wazepolice.py --interval 0 --format csv --output waze_alerts.csv
```

### Long running session with frequent checkpoints

```bash
python wazepolice.py --interval 300 --checkpoint 60 --runtime "1d" --filter "POLICE,ACCIDENT" --format json
```

### Force a new session ignoring existing checkpoints

```bash
python wazepolice.py --new --interval 300 --filter "POLICE,ACCIDENT,HAZARD"
```

### Resume from a specific checkpoint (not the latest)

```bash
python wazepolice.py --resume session_checkpoint_1744615221.json --format csv
```

## Troubleshooting

### API Connectivity Issues

If the scraper is having trouble connecting to the Waze API, try:

1. Checking your internet connection
2. Using smaller geographic bounds
3. Increasing the polling interval

### Schema Validation Errors

If you encounter schema validation errors, ensure you have the latest schema file in the `schema` directory.

### Session Resume Issues

If you have trouble resuming a session:

1. Make sure you have valid checkpoint files in your directory
2. Try using `--new` to start a fresh session if resuming fails
3. Use `--resume` with a specific checkpoint file if automatic resume fails
4. Check the logs for specific error messages

## License

This project is provided for educational purposes only. Usage of this tool should comply with Waze's terms of service.

## Disclaimer

This tool accesses undocumented API endpoints which may change without notice. It is not affiliated with or endorsed by Waze.
