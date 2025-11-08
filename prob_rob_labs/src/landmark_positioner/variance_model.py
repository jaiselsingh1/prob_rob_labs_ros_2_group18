#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

from pathlib import Path
import argparse

# default path inside workspace pictures folder (non-root)
CSV_PATH = "/home/hqh/ros2_ws/src/prob_rob_labs_ros_2/pictures/landmark_data_cyan.csv"

# region where measurements are considered reliable
REASONABLE_D_MIN = 1.0
REASONABLE_D_MAX = 10.0
REASONABLE_TH_ABS = 0.6

# factor to inflate variance outside the reliable region
OUT_OF_RANGE_SCALE = 3.0


def load_error_data(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"csv file not found at {csv_path}")

    df = pd.read_csv(p)

    # normalize column names: strip whitespace and lower
    df.columns = [c.strip() for c in df.columns]

    required = {
        "measured_d",
        "measured_theta",
        "true_d",
        "true_theta",
        "err_d",
        "err_theta",
    }
    present = set(df.columns)
    missing = required - present
    if missing:
        raise ValueError(f"missing required columns in csv: {missing}; present columns: {present}")

    # coerce numeric columns and drop NaNs/infs
    for col in required | {"stamp"}:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))

    # keep only reasonable positive distances
    df = df[(df["measured_d"] > 0.0) & (df["true_d"] > 0.0)]

    if df.empty:
        raise ValueError("no valid samples after cleaning")

    return df


def compute_base_variances(df: pd.DataFrame):
    d_meas = df["measured_d"].to_numpy()
    th_meas = df["measured_theta"].to_numpy()
    d_err = df["err_d"].to_numpy()
    th_err = df["err_theta"].to_numpy()

    dist_mask = (d_meas >= REASONABLE_D_MIN) & (d_meas <= REASONABLE_D_MAX)
    theta_mask = (np.abs(th_meas) <= REASONABLE_TH_ABS)

    # Use sample variance (ddof=1) when possible; fall back to population var if n<2
    def safe_var(x):
        x = np.asarray(x)
        if x.size >= 2:
            return float(np.var(x, ddof=1))
        elif x.size == 1:
            return 0.0
        else:
            return float(np.nan)

    if dist_mask.sum() >= 10:
        var_d = safe_var(d_err[dist_mask])
    else:
        var_d = safe_var(d_err)

    if theta_mask.sum() >= 10:
        var_th = safe_var(th_err[theta_mask])
    else:
        var_th = safe_var(th_err)

    return var_d, var_th


def make_variance_model(var_d_base: float, var_th_base: float):
    def sigma_d2(d: float) -> float:
        if d < REASONABLE_D_MIN or d > REASONABLE_D_MAX:
            return OUT_OF_RANGE_SCALE * var_d_base
        return var_d_base

    def sigma_theta2(theta: float) -> float:
        if abs(theta) > REASONABLE_TH_ABS:
            return OUT_OF_RANGE_SCALE * var_th_base
        return var_th_base

    def R(d: float, theta: float):
        return np.array([
            [sigma_d2(d), 0.0],
            [0.0,         sigma_theta2(theta)],
        ])

    return sigma_d2, sigma_theta2, R


def main():
    parser = argparse.ArgumentParser(description='Build a variance model from landmark CSV data')
    parser.add_argument('--csv', '-c', default=CSV_PATH, help='path to CSV file')
    args = parser.parse_args()

    # if default path doesn't exist, try to find any landmark_data*.csv in workspace pictures
    csv_path = args.csv
    if not os.path.exists(csv_path):
        pics = Path('/home/hqh/ros2_ws/src/prob_rob_labs_ros_2/pictures')
        if pics.exists():
            matches = list(pics.glob('landmark_data*.csv'))
            if matches:
                csv_path = str(matches[0])

    df = load_error_data(csv_path)
    var_d_base, var_th_base = compute_base_variances(df)

    print(f"using csv: {csv_path}")
    print(f"N samples = {len(df)}")
    print(f"base var_d     = {var_d_base:.6e}")
    print(f"base var_theta = {var_th_base:.6e}")

    sigma_d2, sigma_theta2, R = make_variance_model(var_d_base, var_th_base)

    print("\nexample sigma_d^2(d):")
    for d in [0.5, 2.0, 5.0, 12.0]:
        print(f"  d={d:4.1f} -> {sigma_d2(d):.6e}")

    print("\nexample sigma_theta^2(theta):")
    for th in [0.0, 0.3, 0.8]:
        print(f"  theta={th:5.2f} -> {sigma_theta2(th):.6e}")

    print("\nexample R(3.0, 0.2):")
    print(R(3.0, 0.2))


if __name__ == "__main__":
    main()