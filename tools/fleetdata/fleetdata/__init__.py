"""fleetdata: unified loaders for public "units x time series x event table" datasets (see core.py)."""
from .core import Fleet, Writer, open_fleet, KINDS, EVENT_COLS, UNIT_COLS  # noqa: F401
