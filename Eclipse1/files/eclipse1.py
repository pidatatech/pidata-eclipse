"""
Eclipse timing for the 2026 August 12 total solar eclipse.

Usage:
    python eclipse.py                       # table for cities in and near the path
    python eclipse.py Bilbao                # detailed info for one city
    python eclipse.py "Palma de Mallorca"   # quote if the name has spaces
    python eclipse.py "43.26, -2.94"        # or pass raw coordinates

For any location on Earth, we compute:
    - when the eclipse starts, peaks, and ends (in local time)
    - how much of the Sun gets covered at maximum
    - whether the location is inside the path of totality
    - if it is, how long totality lasts there

The astronomy runs off Skyfield with JPL DE421 planetary ephemerides -
the same data NASA uses. Sub-second-accurate solar and lunar positions.
The percentage-covered number comes from the geometric intersection of
the Sun's and Moon's apparent disks.

Run:
    pip install skyfield skyfield-data rich geopy timezonefinder numpy
    python eclipse.py
"""

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from rich.console import Console
from rich.table import Table
from skyfield.api import load, load_file, wgs84
from skyfield_data import get_skyfield_data_path
from timezonefinder import TimezoneFinder

console = Console()


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# The eclipse: Aug 12, 2026. It starts around 15:30 UT (Siberia) and ends
# around 20:00 UT (Balearic Islands). We search a 7-hour window to be safe.
ECLIPSE_YEAR, ECLIPSE_MONTH, ECLIPSE_DAY = 2026, 8, 12
WINDOW_START_HOUR = 14   # UT
WINDOW_HOURS = 7

# Sample resolution for finding eclipse start/max/end. 5 seconds gives
# ~5s precision on totality duration and is plenty for a text report.
SAMPLE_STEP_SECONDS = 5

# Physical radii (km), needed to compute apparent angular sizes.
R_SUN_KM = 695_700.0
R_MOON_KM = 1_737.4

# Cities to display in the no-argument summary table. Chosen to tell the
# story of the path: Iceland, coast of Spain, meseta, Balearic Islands,
# plus a couple of nearby capitals for context.
CITIES = [
    ("Reykjavik",    "Iceland",  64.147, -21.940),
    ("Isafjordur",   "Iceland",  66.075, -23.126),
    ("Bilbao",       "Spain",    43.263,  -2.935),
    ("Santander",    "Spain",    43.462,  -3.809),
    ("Burgos",       "Spain",    42.343,  -3.697),
    ("Leon",         "Spain",    42.599,  -5.567),
    ("Valladolid",   "Spain",    41.652,  -4.724),
    ("Zaragoza",     "Spain",    41.649,  -0.887),
    ("Palma",        "Spain",    39.570,   2.650),
    ("Barcelona",    "Spain",    41.385,   2.173),
    ("Madrid",       "Spain",    40.417,  -3.703),
    ("Lisbon",       "Portugal", 38.722,  -9.139),
]


# ---------------------------------------------------------------------------
# EPHEMERIS (loaded once at import time)
# ---------------------------------------------------------------------------

# skyfield-data ships de421.bsp as a Python package, so this works offline
# and we don't pay a 17 MB download on first run.
_bsp_path = os.path.join(get_skyfield_data_path(), "de421.bsp")
_ts = load.timescale()
_eph = load_file(_bsp_path)
_sun = _eph["sun"]
_moon = _eph["moon"]
_earth = _eph["earth"]

_tz_finder = TimezoneFinder()


# ---------------------------------------------------------------------------
# GEOMETRY HELPERS
# ---------------------------------------------------------------------------

def circle_intersection_area(d: np.ndarray,
                             R: np.ndarray,
                             r: np.ndarray) -> np.ndarray:
    """Area of intersection of two circles with radii R and r whose centers
    are distance d apart. All inputs are arrays of the same shape.
    Works elementwise. Standard closed-form formula.
    """
    d = np.asarray(d, dtype=float)
    R = np.asarray(R, dtype=float)
    r = np.asarray(r, dtype=float)

    out = np.zeros_like(d)

    # Case 1: no overlap
    no_overlap = d >= (R + r)

    # Case 2: one disk fully contains the other
    contained = d <= np.abs(R - r)

    # Case 3: partial overlap
    partial = ~no_overlap & ~contained

    # Clip arccos arguments to [-1, 1] to avoid tiny numerical drift.
    dp = np.where(partial, d, 1.0)   # dummy 1 where not partial (avoids /0)
    a = np.clip((dp**2 + r**2 - R**2) / (2 * dp * r), -1.0, 1.0)
    b = np.clip((dp**2 + R**2 - r**2) / (2 * dp * R), -1.0, 1.0)
    prod = np.maximum(0.0,
                      (-dp + r + R) * (dp + r - R)
                      * (dp - r + R) * (dp + r + R))
    partial_area = r**2 * np.arccos(a) + R**2 * np.arccos(b) - 0.5 * np.sqrt(prod)

    out = np.where(partial, partial_area, out)
    out = np.where(contained, np.pi * np.minimum(R, r) ** 2, out)
    # no_overlap already covered by out = zeros_like
    return out


# ---------------------------------------------------------------------------
# ECLIPSE CALCULATION
# ---------------------------------------------------------------------------

def compute_eclipse(lat: float, lon: float) -> dict:
    """Return eclipse timing & coverage at (lat, lon).

    Result dict keys:
        visible         bool
        start_utc       datetime | None
        max_utc         datetime | None
        end_utc         datetime | None
        max_obscuration float   (0..1 - fraction of Sun's area covered)
        max_magnitude   float   (0..1+ - fraction of Sun's diameter covered)
        in_totality     bool
        totality_seconds float
    """
    observer = _earth + wgs84.latlon(lat, lon)

    # Build the time vector: WINDOW_HOURS long, sampled every SAMPLE_STEP_SECONDS.
    total_seconds = WINDOW_HOURS * 3600
    n = total_seconds // SAMPLE_STEP_SECONDS
    step_minutes = SAMPLE_STEP_SECONDS / 60.0
    minutes = np.arange(0, n) * step_minutes
    times = _ts.utc(ECLIPSE_YEAR, ECLIPSE_MONTH, ECLIPSE_DAY,
                    WINDOW_START_HOUR, minutes)

    # Apparent positions of Sun and Moon as seen from the observer.
    sun_app = observer.at(times).observe(_sun).apparent()
    moon_app = observer.at(times).observe(_moon).apparent()

    # Angular separation between disk centers (radians per sample).
    seps = sun_app.separation_from(moon_app).radians

    # Angular radii of each disk (radians per sample).
    sun_dist_km = sun_app.distance().km
    moon_dist_km = moon_app.distance().km
    sun_rad = np.arcsin(R_SUN_KM / sun_dist_km)
    moon_rad = np.arcsin(R_MOON_KM / moon_dist_km)

    # Sun altitude - eclipse only "visible" while the Sun is up.
    sun_alt_deg = sun_app.altaz()[0].degrees
    above_horizon = sun_alt_deg > 0

    # Obscuration = fraction of Sun's disk area covered by Moon's disk.
    sun_area = np.pi * sun_rad**2
    intersection = circle_intersection_area(seps, sun_rad, moon_rad)
    obscuration = np.where(above_horizon, intersection / sun_area, 0.0)

    if obscuration.max() <= 0:
        return {"visible": False,
                "start_utc": None, "max_utc": None, "end_utc": None,
                "max_obscuration": 0.0, "max_magnitude": 0.0,
                "in_totality": False, "totality_seconds": 0.0}

    # Find max coverage index, then start/end (first/last sample with any coverage).
    max_idx = int(np.argmax(obscuration))
    covered = obscuration > 0.001
    start_idx = int(np.argmax(covered))
    end_idx = len(covered) - 1 - int(np.argmax(covered[::-1]))

    # Magnitude: fraction of Sun's diameter covered along the center-line axis.
    # (sun_rad + moon_rad - sep) / (2 * sun_rad)  clipped to [0, ...]
    magnitude = np.maximum(
        0.0,
        (sun_rad + moon_rad - seps) / (2 * sun_rad),
    )
    magnitude = np.where(above_horizon, magnitude, 0.0)

    # Path of totality: Moon's disk fully covers Sun's disk at some moment
    # AND Sun is above horizon at that moment.
    total_mask = (seps + sun_rad <= moon_rad) & above_horizon
    in_totality = bool(total_mask.any())

    if in_totality:
        totality_start_idx = int(np.argmax(total_mask))
        totality_end_idx = len(total_mask) - 1 - int(np.argmax(total_mask[::-1]))
        totality_seconds = ((totality_end_idx - totality_start_idx)
                            * SAMPLE_STEP_SECONDS)
    else:
        totality_seconds = 0.0

    return {
        "visible":          True,
        "start_utc":        times[start_idx].utc_datetime(),
        "max_utc":          times[max_idx].utc_datetime(),
        "end_utc":          times[end_idx].utc_datetime(),
        "max_obscuration":  float(obscuration[max_idx]),
        "max_magnitude":    float(magnitude[max_idx]),
        "in_totality":      in_totality,
        "totality_seconds": float(totality_seconds),
    }


# ---------------------------------------------------------------------------
# FORMATTING
# ---------------------------------------------------------------------------

def local_and_utc(utc_dt: datetime, lat: float, lon: float) -> str:
    """Format a UTC datetime as 'HH:MM:SS TZ (HH:MM:SS UT)' at (lat, lon)."""
    tz_name = _tz_finder.timezone_at(lat=lat, lng=lon)
    if not tz_name:
        return utc_dt.strftime("%H:%M:%S UT")
    local = utc_dt.astimezone(ZoneInfo(tz_name))
    return f"{local.strftime('%H:%M:%S %Z')} ({utc_dt.strftime('%H:%M:%S UT')})"


def format_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(round(seconds - m * 60))
    if m == 0:
        return f"{s}s"
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# GEOCODING
# ---------------------------------------------------------------------------

def parse_query(query: str) -> tuple[float, float, str] | None:
    """Return (lat, lon, label) for a query string.

    Accepts either 'lat, lon' or a city name. Returns None if unparseable
    or the geocoder can't find the city.
    """
    # First: try to parse as raw coordinates like "43.26, -2.94".
    if "," in query:
        parts = [p.strip() for p in query.split(",", 1)]
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            pass  # not raw coords, treat as a place name
        else:
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon, f"({lat:.3f}, {lon:.3f})"
            # Parsed as floats but out of range -> tell the user, don't
            # fall through and waste a geocoder call on nonsense.
            console.print(f"[red]Coordinates out of range: "
                          f"lat must be -90..90, lon must be -180..180.[/red]")
            return None

    # Fall through: geocode via Nominatim (OpenStreetMap).
    try:
        geolocator = Nominatim(user_agent="eclipse-tracker-2026")
        location = geolocator.geocode(query, timeout=10)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        console.print(f"[red]Geocoder error: {e}[/red]")
        return None

    if location is None:
        return None
    return location.latitude, location.longitude, location.address


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def print_header() -> None:
    console.print()
    console.print("[bold yellow]# 2026 August 12 Total Solar Eclipse[/bold yellow]")
    console.print("[dim]First mainland-Europe totality since 1999. "
                  "Path: Greenland -> Iceland -> N Spain -> Balearic Sea.[/dim]")


def print_city_report(label: str, lat: float, lon: float) -> None:
    console.print()
    console.print(f"[bold cyan]{label}[/bold cyan]")
    console.print(f"[dim]{lat:.3f}, {lon:.3f}[/dim]")
    console.print("-" * 60)

    r = compute_eclipse(lat, lon)
    if not r["visible"]:
        console.print("[red]No eclipse visible from this location "
                      "(Sun below horizon during the whole event).[/red]")
        return

    console.print(f"Eclipse start:     [bold]{local_and_utc(r['start_utc'], lat, lon)}[/bold]")
    console.print(f"Maximum eclipse:   [bold]{local_and_utc(r['max_utc'],   lat, lon)}[/bold]")
    console.print(f"Eclipse end:       [bold]{local_and_utc(r['end_utc'],   lat, lon)}[/bold]")

    pct = r["max_obscuration"] * 100
    if pct >= 99.9:
        color = "bold green"
    elif pct >= 90:
        color = "yellow"
    else:
        color = "cyan"
    console.print(f"Sun coverage:      [{color}]{pct:.2f}%[/{color}]  "
                  f"(magnitude {r['max_magnitude']:.3f})")

    if r["in_totality"]:
        console.print(f"Path of totality:  [bold green]YES[/bold green]")
        console.print(f"Totality duration: [bold green]{format_duration(r['totality_seconds'])}[/bold green]")
    else:
        console.print(f"Path of totality:  [dim]No (partial only)[/dim]")


def print_cities_table() -> None:
    table = Table(show_lines=False,
                  header_style="bold",
                  title_style="bold yellow",
                  title="Cities along the eclipse path")
    table.add_column("City",         style="cyan")
    table.add_column("Country",      style="dim")
    table.add_column("Max coverage", justify="right")
    table.add_column("Max time (local)")
    table.add_column("Totality?")

    for name, country, lat, lon in CITIES:
        r = compute_eclipse(lat, lon)

        if not r["visible"]:
            table.add_row(name, country, "-", "[dim](Sun down)[/dim]", "[dim]-[/dim]")
            continue

        pct = r["max_obscuration"] * 100
        if pct >= 99.9:
            pct_str = f"[bold green]{pct:.2f}%[/bold green]"
        elif pct >= 90:
            pct_str = f"[yellow]{pct:.2f}%[/yellow]"
        else:
            pct_str = f"{pct:.1f}%"

        # Just HH:MM in the table to keep width down.
        tz_name = _tz_finder.timezone_at(lat=lat, lng=lon)
        if tz_name:
            local_str = r["max_utc"].astimezone(ZoneInfo(tz_name)).strftime("%H:%M %Z")
        else:
            local_str = r["max_utc"].strftime("%H:%M UT")

        if r["in_totality"]:
            path_str = f"[bold green]{format_duration(r['totality_seconds'])}[/bold green]"
        else:
            path_str = "[dim]partial[/dim]"

        table.add_row(name, country, pct_str, local_str, path_str)

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    print_header()

    if len(sys.argv) < 2:
        console.print()
        console.print("[dim]No location given. Showing summary for cities in the path.[/dim]")
        console.print("[dim]Tip: `python eclipse.py \"Palma de Mallorca\"` for one city, "
                      "or `python eclipse.py \"43.26, -2.94\"` for coords.[/dim]")
        print_cities_table()
        return

    query = " ".join(sys.argv[1:]).strip()
    console.print()
    console.print(f"[dim]Looking up '{query}'...[/dim]")

    parsed = parse_query(query)
    if parsed is None:
        console.print(f"[red]Could not find '{query}'. Try a bigger nearby city, "
                      "or pass raw coordinates as 'lat, lon'.[/red]")
        sys.exit(1)

    lat, lon, label = parsed
    print_city_report(label if len(label) < 60 else query, lat, lon)


if __name__ == "__main__":
    main()
