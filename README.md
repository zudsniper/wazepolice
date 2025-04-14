# WazePolice Scraper

A Python tool for extracting police reports and other alert types from Waze by directly accessing the internal API endpoints used by the Waze live map.

Version: 1.0.2  
Author: github - @zudsniper

## Features

- Extract police reports and other alert types from Waze in real-time
- Support for custom geographic boundaries
- Multiple output formats (JSON, GeoJSON, CSV)
- Continuous polling with configurable intervals
- Time-limited operation with "99d 99h 99m 99s" format
- Custom alert type filtering
- Timestamps added to output files automatically

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
# One-time extraction with default bounds (Atlanta to Nashville)
python wazepolice.py

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

# Specify output format (json, geojson, or csv)
python wazepolice.py --format geojson

# Save raw API response data
python wazepolice.py --raw raw_data.json

# Use custom JSON schema file
python wazepolice.py --schema my_schema.json
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
| `--help` | Show help information | |

## Output Format

The scraper can output data in three formats:

### JSON (default)

```json
[
  {
    "id": "abc123",
    "type": "POLICE",
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

### GeoJSON

GeoJSON-formatted data for use with mapping applications.

### CSV

Comma-separated values format for easy import into spreadsheet applications.

## File Naming

Output files are automatically appended with a Unix epoch timestamp:
```
police_1680353696.json
```

## Examples

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

## Troubleshooting

### API Connectivity Issues

If the scraper is having trouble connecting to the Waze API, try:

1. Checking your internet connection
2. Using smaller geographic bounds
3. Increasing the polling interval

### Schema Validation Errors

If you encounter schema validation errors, ensure you have the latest schema file in the `schema` directory.

## License

This project is provided for educational purposes only. Usage of this tool should comply with Waze's terms of service.

## Disclaimer

This tool accesses undocumented API endpoints which may change without notice. It is not affiliated with or endorsed by Waze.
