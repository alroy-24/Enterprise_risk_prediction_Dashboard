"""
Risk Rating System - Convert probabilities to credit-style ratings (AAA to D).
Industry-standard approach used by rating agencies and banks.
"""
import pandas as pd
import numpy as np


def probability_to_rating(prob: float) -> str:
    """
    Convert risk probability to credit rating.
    
    Rating Scale:
    - AAA: < 0.01 (Minimal risk)
    - AA:  0.01-0.03 (Very low risk)
    - A:   0.03-0.05 (Low risk)
    - BBB: 0.05-0.10 (Moderate risk)
    - BB:  0.10-0.20 (Elevated risk)
    - B:   0.20-0.35 (High risk)
    - CCC: 0.35-0.50 (Very high risk)
    - CC:  0.50-0.70 (Substantial risk)
    - C:   0.70-0.85 (Extreme risk)
    - D:   >= 0.85 (Default/imminent default)
    """
    if prob < 0.01:
        return "AAA"
    elif prob < 0.03:
        return "AA"
    elif prob < 0.05:
        return "A"
    elif prob < 0.10:
        return "BBB"
    elif prob < 0.20:
        return "BB"
    elif prob < 0.35:
        return "B"
    elif prob < 0.50:
        return "CCC"
    elif prob < 0.70:
        return "CC"
    elif prob < 0.85:
        return "C"
    else:
        return "D"


def rating_to_numeric(rating: str) -> int:
    """Convert rating to numeric score (lower = better)."""
    scale = {"AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5, "B": 6, "CCC": 7, "CC": 8, "C": 9, "D": 10}
    return scale.get(rating, 10)


def add_risk_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk rating column to dataframe."""
    df = df.copy()
    df["risk_rating"] = df["predicted_proba"].apply(probability_to_rating)
    df["rating_numeric"] = df["risk_rating"].apply(rating_to_numeric)
    return df


def get_rating_color(rating: str) -> str:
    """Get color for rating (green = good, red = bad)."""
    colors = {
        "AAA": "#00AA00", "AA": "#44CC44", "A": "#88DD88",
        "BBB": "#FFDD00", "BB": "#FFAA00", "B": "#FF7700",
        "CCC": "#FF5500", "CC": "#FF3300", "C": "#CC0000", "D": "#990000"
    }
    return colors.get(rating, "#666666")


