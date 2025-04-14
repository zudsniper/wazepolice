#!/usr/bin/env python3
"""
Waze Police Data API Scraper

This script extracts police report coordinates from Waze by directly accessing
the internal API endpoints used by the Waze live map.

Usage:
    python wazepolice.py --bounds lat1,lon1,lat2,lon2
    python wazepolice.py --output police_data.json
    python wazepolice.py --interval 300
    python wazepolice.py --runtime "1d 2h 30m 15s"
    python wazepolice.py --filter "POLICE,ACCIDENT,HAZARD"
"""

import json
import time
import httpx
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import jsonschema

import pandas as pd
import geojson
import typer
from loguru import logger

# Version and author information
VERSION = "1.1.0"
AUTHOR = "@zudsniper"

# Default coordinates (can be modified via command-line args)
DEFAULT_BOUNDS = (33.7490, -84.3880, 36.1627, -86.7816)  # Atlanta to Nashville
DEFAULT_OUTPUT = "police.json"  # Changed default to JSON
DEFAULT_INTERVAL = 600  # 10 minutes
DEFAULT_ALERT_TYPES = ["POLICE"]  # Default alert types to filter


class WazeAPIDataScraper:
    """Extracts police report coordinates from Waze by using internal API endpoints."""

    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = DEFAULT_BOUNDS,
        schema_path: str = Path(__file__).parent / "schema" / "wazedata.json",
        alert_types: List[str] = DEFAULT_ALERT_TYPES
    ):
        """
        Initialize the scraper.

        Args:
            bounds: Tuple of (lat1, lon1, lat2, lon2) defining the bounding box.
            schema_path: Path to the JSON schema file.
            alert_types: List of alert types to extract.
        """
        self.bounds = bounds
        self.alert_types = [alert_type.upper() for alert_type in alert_types]
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.waze.com/live-map/",
            "Origin": "https://www.waze.com",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Connection": "keep-alive",
        }
        
        # Load JSON schema
        self.schema = None
        try:
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
            logger.info(f"Loaded schema from {schema_path}")
        except Exception as e:
            logger.warning(f"Unable to load schema file: {str(e)}")
            
        logger.info(f"Initialized API scraper with bounds: {bounds}, alert types: {self.alert_types}")

    def validate_data(self, data: Dict) -> bool:
        """
        Validate data against the JSON schema.
        
        Args:
            data: Data to validate
            
        Returns:
            True if validation passed, False otherwise
        """
        if not self.schema:
            logger.warning("No schema loaded, skipping validation")
            return True
            
        try:
            jsonschema.validate(instance=data, schema=self.schema)
            logger.info("Data validation passed")
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"Schema validation failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error during schema validation: {str(e)}")
            return False

    def extract_police_data(self) -> List[Dict]:
        """
        Extract alert data from Waze's API based on the specified alert types.

        Returns:
            List of dictionaries containing alert data.
        """
        results = []
        timestamp = datetime.now().isoformat()
        now = int(time.time() * 1000)  # Current time in milliseconds
        self.last_raw_data = None  # Reset last raw data

        try:
            # Extract coordinates from bounds
            lat1, lon1, lat2, lon2 = self.bounds
            
            # Primary API endpoint
            primary_endpoint = f"https://www.waze.com/live-map/api/georss?top={lat2}&bottom={lat1}&left={lon1}&right={lon2}&env=na&types=alerts"
            logger.debug(f"Using primary endpoint: {primary_endpoint}")
            
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                response = client.get(primary_endpoint, headers=self.headers)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Ensure data has the required fields according to the schema
                        if not "startTimeMillis" in data:
                            data["startTimeMillis"] = now - 3600000  # 1 hour ago
                        if not "endTimeMillis" in data:
                            data["endTimeMillis"] = now
                            
                        # Validate the data against the schema
                        is_valid = self.validate_data(data)
                        
                        # Store the raw data if valid
                        if is_valid:
                            self.last_raw_data = data
                        
                        # Process data based on the structure
                        if isinstance(data, dict) and is_valid:
                            # Check for different response formats
                            if "alerts" in data:
                                results = self._process_police_data_v1(data["alerts"], timestamp)
                                logger.success(f"Successfully extracted {len(results)} alerts using format v1")
                            elif "jams" in data:
                                # Format used in some other endpoints
                                results = self._process_police_data_v2(data, timestamp)
                                logger.success(f"Successfully extracted {len(results)} alerts using format v2")
                    except json.JSONDecodeError:
                        logger.warning(f"Response was not valid JSON from primary endpoint")
                else:
                    logger.warning(f"Failed to get data from primary endpoint: {response.status_code}")
                
                # If no results were found, try fallback methods
                if not results:
                    logger.info("Primary endpoint failed. Trying fallback methods...")
                    fallback_endpoints = [
                        # US server
                        f"https://www.waze.com/rtserver/web/TGeoRSS?bottom={lat1}&left={lon1}&ma=600&mj=100&mu=100&right={lon2}&top={lat2}&types=alerts,traffic",
                        # Alternative endpoint (might work for other regions)
                        f"https://www.waze.com/live-map/api/georss?bottom={lat1}&left={lon1}&top={lat2}&right={lon2}&ma=200&mu=100",
                        # Another format seen in some implementations
                        f"https://www.waze.com/rtserver/web/TGeoRSS?format=JSON&bottom={lat1}&left={lon1}&top={lat2}&right={lon2}&ma=500&mu=300&mj=200&types=alerts,traffic,users"
                    ]
                    
                    # Try each fallback endpoint
                    for endpoint in fallback_endpoints:
                        logger.debug(f"Trying fallback endpoint: {endpoint}")
                        response = client.get(endpoint, headers=self.headers)
                        
                        if response.status_code == 200:
                            try:
                                data = response.json()
                                
                                # Ensure data has the required fields according to the schema
                                if not "startTimeMillis" in data:
                                    data["startTimeMillis"] = now - 3600000  # 1 hour ago
                                if not "endTimeMillis" in data:
                                    data["endTimeMillis"] = now
                                
                                # Validate the data against the schema
                                is_valid = self.validate_data(data)
                                
                                # Store the raw data if valid
                                if is_valid:
                                    self.last_raw_data = data
                                
                                # Process data based on the structure
                                if isinstance(data, dict) and is_valid:
                                    # Check for different response formats
                                    if "alerts" in data:
                                        results = self._process_police_data_v1(data["alerts"], timestamp)
                                        logger.success(f"Successfully extracted {len(results)} alerts using format v1")
                                        break
                                    elif "jams" in data:
                                        # Format used in some other endpoints
                                        results = self._process_police_data_v2(data, timestamp)
                                        logger.success(f"Successfully extracted {len(results)} alerts using format v2")
                                        break
                            except json.JSONDecodeError:
                                logger.warning(f"Response was not valid JSON from {endpoint}")
                        else:
                            logger.warning(f"Failed to get data from {endpoint}: {response.status_code}")
                    
                    # If still no results, try last fallback method
                    if not results:
                        logger.info("All endpoints failed. Trying final fallback method...")
                        results = self._try_fallback_method(timestamp, client)
                
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            
        return results

    def _process_police_data_v1(self, alerts, timestamp):
        """Process alert data from alerts format."""
        results = []
        
        for alert in alerts:
            try:
                # Check if alert type matches any of the requested types
                if alert.get("type") in self.alert_types:
                    # Ensure the alert has the required fields according to the schema
                    if not "location" in alert:
                        # Create a location object if missing
                        if "y" in alert and "x" in alert:
                            alert["location"] = {"x": alert["x"], "y": alert["y"]}
                        else:
                            logger.warning(f"Alert missing required location data: {alert}")
                            continue
                            
                    # Check for missing required fields
                    if not "uuid" in alert:
                        logger.warning(f"Alert missing required uuid field: {alert}")
                        continue
                        
                    if not "pubMillis" in alert:
                        alert["pubMillis"] = int(time.time() * 1000)
                    
                    result = {
                        "id": alert.get("uuid", ""),
                        "type": alert.get("type", ""),
                        "lat": alert.get("location", {}).get("y"),
                        "lon": alert.get("location", {}).get("x"),
                        "reported_by": alert.get("reportedBy", ""),
                        "confidence": alert.get("confidence", 0),
                        "reliability": alert.get("reliability", 0),
                        "report_rating": alert.get("reportRating", 0),
                        "timestamp": timestamp,
                        "pub_millis": alert.get("pubMillis", 0),
                    }
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing alert: {str(e)}")
                
        return results

    def _process_police_data_v2(self, data, timestamp):
        """Process alert data from alternative format."""
        results = []
        
        # This format might organize alerts differently
        alerts = []
        
        # Check different possible structures
        if "alerts" in data:
            alerts = data["alerts"]
        elif "data" in data and "alerts" in data["data"]:
            alerts = data["data"]["alerts"]
            
        for alert in alerts:
            try:
                # Check if it's a requested alert type (different formats use different keys)
                alert_type = alert.get("type") or alert.get("subtype") or ""
                if alert_type in self.alert_types:
                    # Different formats store coordinates differently
                    lat = None
                    lon = None
                    
                    if "location" in alert:
                        lat = alert["location"].get("y")
                        lon = alert["location"].get("x")
                    elif "y" in alert and "x" in alert:
                        lat = alert.get("y")
                        lon = alert.get("x")
                        
                    if lat and lon:
                        # Ensure required fields are present according to schema
                        if not "uuid" in alert:
                            logger.warning(f"Alert missing required uuid field")
                            continue
                            
                        if not "pubMillis" in alert:
                            alert["pubMillis"] = int(time.time() * 1000)
                            
                        result = {
                            "id": alert.get("uuid", ""),
                            "type": alert_type,
                            "lat": lat,
                            "lon": lon,
                            "reported_by": alert.get("reportedBy", ""),
                            "confidence": alert.get("confidence", 0),
                            "reliability": alert.get("reliability", 0),
                            "report_rating": alert.get("reportRating", 0),
                            "timestamp": timestamp,
                            "pub_millis": alert.get("pubMillis", 0),
                        }
                        results.append(result)
            except Exception as e:
                logger.error(f"Error processing alert: {str(e)}")
                
        return results

    def _try_fallback_method(self, timestamp, client=None):
        """Try a different approach based on another observed API pattern."""
        results = []
        now = int(time.time() * 1000)  # Current time in milliseconds
        
        try:
            lat1, lon1, lat2, lon2 = self.bounds
            
            # This format was observed in some implementations
            url = f"https://www.waze.com/rtserver/web/GeoRSS?format=JSON&left={lon1}&right={lon2}&bottom={lat1}&top={lat2}&ma=500&mj=100&mu=100"
            
            if client is None:
                with httpx.Client(timeout=30, follow_redirects=True) as client:
                    response = client.get(url, headers=self.headers)
            else:
                response = client.get(url, headers=self.headers)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Ensure data has the required fields according to the schema
                    if not "startTimeMillis" in data:
                        data["startTimeMillis"] = now - 3600000  # 1 hour ago
                    if not "endTimeMillis" in data:
                        data["endTimeMillis"] = now
                        
                    # Validate the data against the schema
                    is_valid = self.validate_data(data)
                    
                    if is_valid:
                        alerts = data.get("alerts", [])
                        
                        for alert in alerts:
                            if alert.get("type") in self.alert_types:
                                # Ensure required fields are present according to schema
                                if not "uuid" in alert:
                                    logger.warning(f"Alert missing required uuid field")
                                    continue
                                    
                                if not "pubMillis" in alert:
                                    alert["pubMillis"] = int(time.time() * 1000)
                                    
                                result = {
                                    "id": alert.get("uuid", ""),
                                    "type": alert.get("type", ""),
                                    "lat": alert.get("y"),
                                    "lon": alert.get("x"),
                                    "reported_by": alert.get("reportedBy", ""),
                                    "confidence": alert.get("confidence", 0),
                                    "reliability": alert.get("reliability", 0),
                                    "report_rating": alert.get("reportRating", 0),
                                    "timestamp": timestamp,
                                    "pub_millis": alert.get("pubMillis", 0),
                                }
                                results.append(result)
                        
                        logger.success(f"Fallback method extracted {len(results)} alerts")
                except json.JSONDecodeError:
                    logger.warning("Response was not valid JSON from fallback method")
        except Exception as e:
            logger.error(f"Error in fallback method: {str(e)}")
            
        return results

    def save_to_raw_json(self, data: Dict, output_path: str):
        """
        Save the raw API response in the Waze schema format.
        
        Args:
            data: Complete API response data following the schema
            output_path: Path to save the JSON file
        """
        if not data:
            logger.warning("No data to save")
            return
            
        # Add Unix epoch timestamp to filename
        output_path = self._add_timestamp_to_path(output_path)
            
        # Save to file
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.success(f"Saved raw schema data to {output_path}")

    def _add_timestamp_to_path(self, path_str: str) -> Path:
        """
        Add Unix epoch timestamp to filename.
        
        Args:
            path_str: Original file path
            
        Returns:
            Path with timestamp added before the extension
        """
        path = Path(path_str)
        timestamp = int(time.time())
        return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")

    def save_to_geojson(self, data: List[Dict], output_path: str = DEFAULT_OUTPUT, append: bool = True):
        """
        Save extracted data to a GeoJSON file.

        Args:
            data: List of dictionaries containing alert data.
            output_path: Path to save the GeoJSON file.
            append: Whether to append to existing file.
        """
        if not data:
            logger.warning("No data to save")
            return
            
        # Add Unix epoch timestamp to filename
        output_path = self._add_timestamp_to_path(output_path)
        
        # Convert to GeoJSON
        features = []
        for item in data:
            try:
                # Create GeoJSON feature
                feature = geojson.Feature(
                    geometry=geojson.Point((item["lon"], item["lat"])),
                    properties={
                        "id": item["id"],
                        "type": item["type"],
                        "reported_by": item["reported_by"],
                        "confidence": item["confidence"],
                        "reliability": item["reliability"],
                        "report_rating": item["report_rating"],
                        "timestamp": item["timestamp"],
                        "pub_millis": item["pub_millis"],
                    }
                )
                features.append(feature)
            except Exception as e:
                logger.error(f"Error creating GeoJSON feature: {str(e)}")
        
        # Create FeatureCollection
        feature_collection = geojson.FeatureCollection(features)
        
        # Check if file exists and append mode is on
        if output_path.exists() and append:
            try:
                with open(output_path, "r") as f:
                    existing_data = geojson.loads(f.read())
                
                # Combine existing features with new features
                all_features = existing_data["features"] + feature_collection["features"]
                feature_collection = geojson.FeatureCollection(all_features)
                logger.info(f"Appended to existing file: {output_path}")
            except Exception as e:
                logger.error(f"Error when appending to existing file: {str(e)}")
        
        # Save to file
        with open(output_path, "w") as f:
            geojson.dump(feature_collection, f)
            
        logger.success(f"Saved {len(data)} records to {output_path}")

    def save_to_csv(self, data: List[Dict], output_path: str, append: bool = True):
        """
        Save extracted data to a CSV file.

        Args:
            data: List of dictionaries containing alert data.
            output_path: Path to save the CSV file.
            append: Whether to append to existing file.
        """
        if not data:
            logger.warning("No data to save")
            return
            
        # Add Unix epoch timestamp to filename
        output_path = self._add_timestamp_to_path(output_path)
        
        # Create new DataFrame with current data
        df = pd.DataFrame(data)
        
        # Check if file exists and append mode is on
        if output_path.exists() and append:
            try:
                existing_df = pd.read_csv(output_path)
                # Combine with new data
                df = pd.concat([existing_df, df], ignore_index=True)
                logger.info(f"Appended to existing file: {output_path}")
            except Exception as e:
                logger.error(f"Error when appending to existing file: {str(e)}")
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.success(f"Saved {len(data)} records to {output_path}")

    def save_to_json(self, data: List[Dict], output_path: str, append: bool = True):
        """
        Save extracted data to a standard JSON file.
        Preserves all original data fields without transformation.

        Args:
            data: List of dictionaries containing alert data.
            output_path: Path to save the JSON file.
            append: Whether to append to existing file.
        """
        if not data:
            logger.warning("No data to save")
            return
            
        # Add Unix epoch timestamp to filename
        output_path = self._add_timestamp_to_path(output_path)
        
        # Check if file exists and append mode is on
        if output_path.exists() and append:
            try:
                with open(output_path, "r") as f:
                    existing_data = json.load(f)
                
                # Combine existing data with new data
                if isinstance(existing_data, list):
                    data = existing_data + data
                logger.info(f"Appended to existing file: {output_path}")
            except Exception as e:
                logger.error(f"Error when appending to existing file: {str(e)}")
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.success(f"Saved {len(data)} records to {output_path}")


def parse_runtime(runtime_str: str) -> int:
    """
    Parse a runtime string in format "99d 99h 99m 99s" into seconds.
    
    Args:
        runtime_str: String in format "99d 99h 99m 99s"
        
    Returns:
        Total seconds
    """
    total_seconds = 0
    
    # Define patterns for each time unit
    patterns = {
        "d": 86400,  # days to seconds
        "h": 3600,   # hours to seconds
        "m": 60,     # minutes to seconds
        "s": 1       # seconds
    }
    
    # Extract all time components
    for unit, multiplier in patterns.items():
        match = re.search(r"(\d+)\s*" + unit, runtime_str)
        if match:
            total_seconds += int(match.group(1)) * multiplier
    
    return total_seconds


def run(
    bounds: str = typer.Option(
        f"{DEFAULT_BOUNDS[0]},{DEFAULT_BOUNDS[1]},{DEFAULT_BOUNDS[2]},{DEFAULT_BOUNDS[3]}",
        help="Bounding box as lat1,lon1,lat2,lon2",
    ),
    output: str = typer.Option(DEFAULT_OUTPUT, help="Output file path (JSON, GeoJSON, or CSV)"),
    interval: int = typer.Option(DEFAULT_INTERVAL, help="Polling interval in seconds (0 for one-time)"),
    format: str = typer.Option("json", "--format", "-f", help="Force output format: 'json', 'geojson', or 'csv'"),
    schema_path: str = typer.Option(None, "--schema", help="Path to JSON schema file (default: wazedata.json in schema dir)"),
    raw_output: str = typer.Option(None, "--raw", help="Save raw schema-validated data to this file"),
    runtime: str = typer.Option(None, "--runtime", help="Maximum runtime in format '99d 99h 99m 99s'"),
    filter: str = typer.Option("POLICE", "--filter", help="Comma-separated list of alert types to extract (e.g., 'POLICE,ACCIDENT,HAZARD')"),
):
    """Run the Waze police data API scraper."""
    try:
        # Print version information
        logger.info(f"WazePolice Scraper v{VERSION} by {AUTHOR}")
        
        # Parse bounds
        bounds_tuple = tuple(map(float, bounds.split(",")))
        if len(bounds_tuple) != 4:
            raise ValueError("Bounds must be in format: lat1,lon1,lat2,lon2")
            
        # Use provided schema path or default
        schema = schema_path if schema_path else Path(__file__).parent / "schema" / "wazedata.json"
        
        # Parse alert types
        alert_types = [alert_type.strip().upper() for alert_type in filter.split(",")]
        logger.info(f"Filtering for alert types: {alert_types}")
            
        # Initialize scraper
        scraper = WazeAPIDataScraper(bounds=bounds_tuple, schema_path=schema, alert_types=alert_types)
        
        # Parse runtime if provided
        max_runtime = None
        if runtime:
            max_runtime = parse_runtime(runtime)
            logger.info(f"Maximum runtime set to {max_runtime} seconds")
        
        # Determine output format based on file extension or format parameter
        output_path = Path(output)
        
        if format:
            # Use the specified format
            format = format.lower()
            if format not in ["json", "geojson", "csv"]:
                logger.warning(f"Unknown format '{format}', defaulting to json")
                format = "json"
            output_format = format
        else:
            # Determine format from file extension
            suffix = output_path.suffix.lower()
            if suffix == ".geojson":
                output_format = "geojson"
            elif suffix == ".csv":
                output_format = "csv"
            else:
                output_format = "json"  # Default to JSON for all other extensions
        
        # Create a long-lived client for multiple requests
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            if interval > 0:
                logger.info(f"Starting continuous scraping with interval: {interval} seconds")
                start_time = time.time()
                try:
                    while True:
                        # Check if we've exceeded the maximum runtime
                        if max_runtime and (time.time() - start_time) > max_runtime:
                            logger.info(f"Maximum runtime of {max_runtime} seconds reached. Stopping.")
                            break
                            
                        try:
                            data = scraper.extract_police_data()
                            
                            # Save based on the determined format
                            if output_format == "geojson":
                                scraper.save_to_geojson(data, output)
                            elif output_format == "csv":
                                scraper.save_to_csv(data, output)
                            else:
                                # Default to JSON
                                scraper.save_to_json(data, output)
                                
                            # Save raw data if requested
                            if raw_output and hasattr(scraper, "last_raw_data"):
                                scraper.save_to_raw_json(scraper.last_raw_data, raw_output)
                                
                            # Calculate remaining runtime if set
                            if max_runtime:
                                elapsed = time.time() - start_time
                                remaining = max_runtime - elapsed
                                if remaining <= 0:
                                    break
                                next_interval = min(interval, remaining)
                                logger.info(f"Sleeping for {next_interval:.1f} seconds... (Runtime: {elapsed:.1f}/{max_runtime} seconds)")
                                time.sleep(next_interval)
                            else:
                                logger.info(f"Sleeping for {interval} seconds...")
                                time.sleep(interval)
                        except httpx.HTTPError as e:
                            logger.error(f"HTTP error during scraping: {str(e)}")
                            time.sleep(10)  # Short delay before retrying
                        except Exception as e:
                            logger.error(f"Error during scraping: {str(e)}")
                            time.sleep(10)  # Short delay before retrying
                except KeyboardInterrupt:
                    logger.info("Scraping stopped by user")
            else:
                logger.info("Running one-time scraping")
                data = scraper.extract_police_data()
                
                # Save based on the determined format
                if output_format == "geojson":
                    scraper.save_to_geojson(data, output)
                elif output_format == "csv":
                    scraper.save_to_csv(data, output)
                else:
                    # Default to JSON
                    scraper.save_to_json(data, output)
                    
                # Save raw data if requested
                if raw_output and hasattr(scraper, "last_raw_data"):
                    scraper.save_to_raw_json(scraper.last_raw_data, raw_output)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(run)
