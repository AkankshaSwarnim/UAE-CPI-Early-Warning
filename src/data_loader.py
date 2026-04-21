"""
UAE CPI data loader.

Handles the 2014-base (INDEX14) / 2021-base (INDEX21) rebasing discontinuity
by chaining the two series via the officially published Jan-2021 MoM change
on the new base (CPI_MTHCHG21).

Validated: reconstructed series matches published CPI_ANNCHG to ±0.01 pp
and CPI_ANNCHG21 (2024-2025) to ±0.01 pp.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

DIVISION_MAP = {
    "ALL": "All Items",
    "FNB": "Food & Beverages",
    "TOB": "Tobacco",
    "TEX": "Textiles/Clothing",
    "HOU": "Housing/Utilities",
    "FUR": "Furniture/Household",
    "MED": "Medical Care",
    "TRN": "Transportation",
    "COM": "Communications",
    "REC": "Recreation/Culture",
    "EDU": "Education",
    "RES": "Restaurants/Hotels",
    "INS": "Insurance/Financial",
    "MIS": "Miscellaneous",
}

CPI_WEIGHTS = {
    # FCSA weights, 2021 base (percent of basket)
    "Housing/Utilities": 34.1,
    "Transportation": 14.6,
    "Food & Beverages": 13.8,
    "Restaurants/Hotels": 7.1,
    "Miscellaneous": 6.4,
    "Recreation/Culture": 4.3,
    "Education": 4.2,
    "Textiles/Clothing": 3.3,
    "Furniture/Household": 3.2,
    "Communications": 3.1,
    "Medical Care": 1.4,
    "Insurance/Financial": 2.5,
    "Tobacco": 1.0,
    "Food & Beverages ": 13.8,  # tolerate trailing space
}


def load_raw(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load the raw FCSA monthly CPI file and attach Division labels + Date."""
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "FCSA_DF_CPI_3_2_0_all.csv")
    df["Division"] = df["CPI_DIV"].map(DIVISION_MAP)
    df["Date"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m")
    return df


def build_chained_index(df: pd.DataFrame, division: str = "ALL") -> pd.DataFrame:
    """
    Chain CPI_INDEX14 (2008-2020) with CPI_INDEX21 (2021-2025) for a given division.

    The 2014 base is rescaled onto the 2021 base using the implied Dec-2020
    level: dec2020_on21base = first21 / (1 + jan21_mom_on21base/100).
    The link factor is dec2020_on21base / dec2020_on14base.

    Returns a DataFrame with columns: Date, Index, YoY, MoM.
    """
    sub = df[df["CPI_DIV"] == division]
    idx14 = sub[sub["MEASURE"] == "CPI_INDEX14"][["Date", "OBS_VALUE"]].sort_values("Date")
    idx21 = sub[sub["MEASURE"] == "CPI_INDEX21"][["Date", "OBS_VALUE"]].sort_values("Date")
    mth21 = sub[sub["MEASURE"] == "CPI_MTHCHG21"][["Date", "OBS_VALUE"]].sort_values("Date")

    if idx14.empty or idx21.empty or mth21.empty:
        raise ValueError(f"Missing base series for division {division!r}")

    last14 = idx14.loc[idx14["Date"] == "2020-12-01", "OBS_VALUE"].values
    first21 = idx21.loc[idx21["Date"] == "2021-01-01", "OBS_VALUE"].values
    jan21_mom = mth21.loc[mth21["Date"] == "2021-01-01", "OBS_VALUE"].values

    if not (len(last14) and len(first21) and len(jan21_mom)):
        raise ValueError(f"Can't find link points for division {division!r}")

    dec2020_on21base = first21[0] / (1 + jan21_mom[0] / 100)
    link = dec2020_on21base / last14[0]

    idx14_rebased = idx14.copy()
    idx14_rebased["OBS_VALUE"] = idx14_rebased["OBS_VALUE"] * link

    merged = pd.concat([idx14_rebased, idx21], ignore_index=True).sort_values("Date").reset_index(drop=True)
    merged = merged.rename(columns={"OBS_VALUE": "Index"})
    merged["YoY"] = merged["Index"].pct_change(12) * 100
    merged["MoM"] = merged["Index"].pct_change(1) * 100
    return merged


def build_sector_panel(df: pd.DataFrame, start: str = "2014-01-01") -> pd.DataFrame:
    """
    Build a wide panel of chained CPI indices for all divisions.

    Returns a DataFrame indexed by Date with one column per division (Index level).
    """
    frames = []
    for code, label in DIVISION_MAP.items():
        try:
            s = build_chained_index(df, code)[["Date", "Index"]].set_index("Date")
            s.columns = [label]
            frames.append(s)
        except ValueError:
            continue
    panel = pd.concat(frames, axis=1).sort_index()
    panel = panel[panel.index >= pd.Timestamp(start)]
    return panel


def yoy_panel(index_panel: pd.DataFrame) -> pd.DataFrame:
    """YoY (%) for each column in an index-level panel."""
    return index_panel.pct_change(12) * 100


def mom_panel(index_panel: pd.DataFrame) -> pd.DataFrame:
    """MoM (%) for each column in an index-level panel."""
    return index_panel.pct_change(1) * 100


def validate_chain(df: pd.DataFrame, chained_all: pd.DataFrame) -> dict:
    """
    Compare the chained 'All Items' YoY against the officially published
    CPI_ANNCHG (2014 base) and CPI_ANNCHG21 (2021 base).
    """
    pub14 = (
        df[(df["MEASURE"] == "CPI_ANNCHG") & (df["CPI_DIV"] == "ALL")]
        [["Date", "OBS_VALUE"]]
        .rename(columns={"OBS_VALUE": "pub"})
    )
    pub21 = (
        df[(df["MEASURE"] == "CPI_ANNCHG21") & (df["CPI_DIV"] == "ALL")]
        [["Date", "OBS_VALUE"]]
        .rename(columns={"OBS_VALUE": "pub"})
    )

    c14 = chained_all.merge(pub14, on="Date")
    c21 = chained_all.merge(pub21, on="Date")
    return {
        "n_2014_base": len(c14),
        "mean_abs_diff_2014_base": (c14["YoY"] - c14["pub"]).abs().mean(),
        "max_abs_diff_2014_base": (c14["YoY"] - c14["pub"]).abs().max(),
        "n_2021_base": len(c21),
        "mean_abs_diff_2021_base": (c21["YoY"] - c21["pub"]).abs().mean(),
        "max_abs_diff_2021_base": (c21["YoY"] - c21["pub"]).abs().max(),
    }


if __name__ == "__main__":
    df = load_raw()
    chained = build_chained_index(df, "ALL")
    chained = chained[chained["Date"] >= "2014-01-01"]
    print("Chained series:", chained["Date"].min().date(), "to", chained["Date"].max().date(), f"(n={len(chained)})")
    print("\nValidation:")
    for k, v in validate_chain(df, chained).items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    panel = build_sector_panel(df)
    print(f"\nSector panel: {panel.shape[0]} months x {panel.shape[1]} sectors")
