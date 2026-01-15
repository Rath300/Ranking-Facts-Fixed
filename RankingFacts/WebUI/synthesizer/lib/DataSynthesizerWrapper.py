import json
import os
import numpy as np
import pandas as pd


def _read_csv(csv_file):
    try:
        return pd.read_csv(csv_file)
    except UnicodeDecodeError:
        return pd.read_csv(csv_file, encoding='latin1')


def get_histograms_of(csv_file, bins=10):
    """
    Inspect a CSV and emit basic histogram/bar-chart data used by the UI.

    Returns a dict with attribute lists and writes a <name>_plot.json file
    that contains barchart (categorical) and histogram (numeric) payloads.
    """
    df = _read_csv(csv_file)

    numeric_atts = df.select_dtypes(include=[np.number]).columns.tolist()
    cate_atts = [c for c in df.columns if c not in numeric_atts]
    all_atts = df.columns.tolist()
    drawable_atts = all_atts  # UI iterates over this for charts

    hist_json = {"barchart": {}, "histogram": {}}

    # Categorical counts
    for col in cate_atts:
        counts = df[col].fillna("NA").astype(str).value_counts()
        hist_json["barchart"][col] = {
            "bins": counts.index.tolist(),
            "counts": counts.tolist()
        }

    # Numeric histograms
    for col in numeric_atts:
        series = df[col].dropna()
        if series.empty:
            hist_json["histogram"][col] = {"bin_l": [], "counts": []}
            continue
        counts, bin_edges = np.histogram(series, bins=bins)
        bin_l = [round(float(b), 3) for b in bin_edges[:-1]]
        hist_json["histogram"][col] = {"bin_l": bin_l, "counts": counts.tolist()}

    # Persist plot json next to data
    description_file = os.path.splitext(csv_file)[0] + "_plot.json"
    with open(description_file, 'w') as outfile:
        json.dump(hist_json, outfile, indent=2)

    return {
        "cate_atts": cate_atts,
        "all_atts": all_atts,
        "numeric_atts": numeric_atts,
        "drawable_atts": drawable_atts
    }


def get_categorical_attributes_csv(csv_file):
    df = _read_csv(csv_file)
    numeric_atts = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in df.columns if c not in numeric_atts]


def get_binary_attributes_csv(csv_file):
    df = _read_csv(csv_file)
    binary_atts = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2:
            binary_atts.append(col)
    return binary_atts



