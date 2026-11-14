"""File-based loaders for dashboard pages. Every loader degrades to None
(never raises) when a result file is missing or malformed, so pages can
render an explicit "Not available" state instead of crashing.
"""

import json
import os

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def path(*parts):
    return os.path.join(REPO_ROOT, *parts)


def load_json(*parts):
    p = path(*parts)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_csv(*parts):
    p = path(*parts)
    if not os.path.isfile(p):
        return None
    try:
        df = pd.read_csv(p)
        return df if not df.empty else df
    except (pd.errors.EmptyDataError, OSError):
        return None


def load_text(*parts):
    p = path(*parts)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def load_image_path(*parts):
    p = path(*parts)
    return p if os.path.isfile(p) else None


def file_mtime(*parts):
    p = path(*parts)
    if not os.path.isfile(p):
        return None
    try:
        import datetime

        return datetime.datetime.fromtimestamp(os.path.getmtime(p))
    except OSError:
        return None


def load_npy(*parts):
    p = path(*parts)
    if not os.path.isfile(p):
        return None
    try:
        import numpy as np

        return np.load(p)
    except (OSError, ValueError):
        return None
