"""
Dataset loading and anomaly label construction for RAG-TrajAD.
Supports Porto, T-Drive, GeoLife with deterministic synthetic anomaly labels.
"""

import os
import json
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset
import torch


# ─────────────────────────────────────────────
# Spatial tokenization helpers
# ─────────────────────────────────────────────

def latlon_to_cell(lat: float, lon: float, resolution: float = 0.005) -> int:
    """Map lat/lon to integer grid cell id."""
    lat_idx = int((lat + 90) / resolution)
    lon_idx = int((lon + 180) / resolution)
    return lat_idx * 100000 + lon_idx


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distance in metres between two lat/lon points."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def trajectory_length_m(coords: List[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(coords) - 1):
        total += haversine_m(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
    return total


def shortest_path_approx_m(coords: List[Tuple[float, float]]) -> float:
    """Approximate shortest path as straight-line OD distance."""
    return haversine_m(coords[0][0], coords[0][1], coords[-1][0], coords[-1][1])


# ─────────────────────────────────────────────
# Anomaly construction rules
# ─────────────────────────────────────────────

def is_detour_anomaly(coords, detour_threshold: float = 1.5) -> bool:
    sp = shortest_path_approx_m(coords)
    if sp < 100:  # very short trip, skip
        return False
    actual = trajectory_length_m(coords)
    return actual > detour_threshold * sp


def compute_segment_speeds(coords, timestamps) -> List[float]:
    """Compute point-to-point speeds (m/s) for a trajectory."""
    speeds = []
    for i in range(len(coords) - 1):
        dist = haversine_m(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
        dt = max(timestamps[i + 1] - timestamps[i], 1e-3)
        speeds.append(dist / dt)
    return speeds


def is_speed_anomaly(coords, timestamps, speed_median: float = None, speed_std: float = None,
                     sigma: float = 3.0) -> bool:
    """Flag if max segment speed deviates > sigma*std from dataset median."""
    if speed_median is None or speed_std is None or speed_std < 1e-3:
        return False
    speeds = compute_segment_speeds(coords, timestamps)
    if not speeds:
        return False
    max_speed = max(speeds)
    return max_speed > speed_median + sigma * speed_std


def is_loop_anomaly(coords, loop_fraction: float = 0.5, close_thresh_m: float = 200) -> bool:
    """Self-intersection: endpoint within 200m of a midpoint."""
    n = len(coords)
    if n < 10:
        return False
    mid_start = int(n * loop_fraction * 0.5)
    mid_end = int(n * loop_fraction * 1.5)
    end_lat, end_lon = coords[-1]
    for i in range(mid_start, min(mid_end, n - 1)):
        if haversine_m(end_lat, end_lon, coords[i][0], coords[i][1]) < close_thresh_m:
            return True
    return False


def is_time_warp_anomaly(duration_s: float, od_mean_s: float, od_std_s: float, sigma: float = 3.0) -> bool:
    if od_std_s < 1e-3:
        return False
    return abs(duration_s - od_mean_s) > sigma * od_std_s


# GeoLife anomaly rules

def is_mode_inconsistency(mode_label: str, speed_mps: float) -> bool:
    """Speed inconsistent with stated transport mode."""
    thresholds = {
        'walk': (0, 3.0),
        'bike': (0, 8.5),
        'bus': (0, 22.0),
        'car': (0, 40.0),
        'subway': (0, 30.0),
        'train': (0, 55.0),
    }
    if mode_label not in thresholds:
        return False
    lo, hi = thresholds[mode_label]
    return speed_mps > hi * 1.2 or (speed_mps < lo * 0.5 and speed_mps < 0.5)


def is_off_routine(coord_lat, coord_lon, user_habitual_routes: List, thresh_m: float = 500) -> bool:
    """Check if point is far from any habitual route."""
    if not user_habitual_routes:
        return False
    min_dist = min(
        haversine_m(coord_lat, coord_lon, r[0], r[1])
        for route in user_habitual_routes
        for r in route
    )
    return min_dist > thresh_m


# ─────────────────────────────────────────────
# Token encoding
# ─────────────────────────────────────────────

CELL_VOCAB_SIZE = 50000
MOTION_BINS = 32
TIME_BINS = 168  # 7 days × 24 hours


def motion_token(speed_mps: float, heading_change_deg: float, is_dwell: bool) -> int:
    """Encode motion as a discrete token 0..MOTION_BINS-1."""
    speed_bin = min(int(speed_mps / 1.5), 7)           # 0-7
    heading_bin = min(int(abs(heading_change_deg) / 22.5), 7)  # 0-7, 22.5° per bin
    dwell_bit = 1 if is_dwell else 0
    return speed_bin * 16 + heading_bin * 2 + dwell_bit  # 0..255, clip to MOTION_BINS


def time_token(unix_ts: float) -> int:
    """Hour-of-week token 0..167."""
    import datetime
    dt = datetime.datetime.utcfromtimestamp(unix_ts)
    return dt.weekday() * 24 + dt.hour


def encode_trajectory(
    coords: List[Tuple[float, float]],
    timestamps: List[float],
    max_len: int = 128,
    cell_resolution: float = 0.005,
) -> Dict[str, torch.Tensor]:
    """
    Returns dict of token tensors for one trajectory window.
    cell_tokens: (T,)  int64
    motion_tokens: (T,) int64
    time_tokens: (T,) int64
    """
    n = min(len(coords), max_len)
    coords = coords[:n]
    timestamps = timestamps[:n]

    cell_tokens = []
    motion_tokens = []
    time_tokens = []

    for i in range(n):
        lat, lon = coords[i]
        cell_id = latlon_to_cell(lat, lon, cell_resolution) % CELL_VOCAB_SIZE
        cell_tokens.append(cell_id)

        # motion
        if i == 0:
            speed, heading_chg, dwell = 0.0, 0.0, False
        else:
            dist = haversine_m(coords[i - 1][0], coords[i - 1][1], lat, lon)
            dt = max(timestamps[i] - timestamps[i - 1], 1e-3)
            speed = dist / dt
            # heading
            def bearing(c1, c2):
                dy = c2[0] - c1[0]
                dx = math.cos(math.radians(c1[0])) * (c2[1] - c1[1])
                return math.degrees(math.atan2(dx, dy)) % 360
            if i >= 2:
                b1 = bearing(coords[i - 2], coords[i - 1])
                b2 = bearing(coords[i - 1], coords[i])
                heading_chg = abs((b2 - b1 + 180) % 360 - 180)
            else:
                heading_chg = 0.0
            dwell = speed < 0.5 and dist < 10

        mt = motion_token(speed, heading_chg, dwell) % MOTION_BINS
        tt = time_token(timestamps[i]) if timestamps[i] > 0 else 0

        motion_tokens.append(mt)
        time_tokens.append(tt)

    return {
        'cell_tokens': torch.tensor(cell_tokens, dtype=torch.long),
        'motion_tokens': torch.tensor(motion_tokens, dtype=torch.long),
        'time_tokens': torch.tensor(time_tokens, dtype=torch.long),
        'length': torch.tensor(n, dtype=torch.long),
    }


# ─────────────────────────────────────────────
# Dataset classes
# ─────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Base class. Subclassed by Porto, TDrive, GeoLife.
    Each item: {tokens, label, domain_id, subtype}
    """
    DOMAIN_ID: int = 0
    NAME: str = 'base'

    def __init__(self, data_path: str, split: str = 'train', max_len: int = 128,
                 anomaly_fraction: float = 0.05, seed: int = 42,
                 detour_threshold: float = 1.5):
        self.max_len = max_len
        self.anomaly_fraction = anomaly_fraction
        self.seed = seed
        self.detour_threshold = detour_threshold
        self.split = split
        self.samples = []  # list of (coords, timestamps, label, subtype_str)
        self._load(data_path)

    def _load(self, data_path: str):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        coords, timestamps, label, subtype = self.samples[idx]
        tokens = encode_trajectory(coords, timestamps, self.max_len)
        return {
            **tokens,
            'label': torch.tensor(label, dtype=torch.long),
            'domain_id': torch.tensor(self.DOMAIN_ID, dtype=torch.long),
            'subtype': subtype,  # string, for soft-tag assignment
        }


class PortoDataset(TrajectoryDataset):
    DOMAIN_ID = 0
    NAME = 'porto'

    def _load(self, data_path: str):
        """Load Porto taxi CSV. Expected columns: TRIP_ID, POLYLINE, TIMESTAMP."""
        rng = random.Random(self.seed)
        csv_path = os.path.join(data_path, 'porto', 'train.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Porto data not found at {csv_path}")

        df = pd.read_csv(csv_path, usecols=['TRIP_ID', 'POLYLINE', 'TIMESTAMP'], nrows=120000)
        # filter empty POLYLINE
        df = df[df['POLYLINE'].notna() & (df['POLYLINE'] != '[]')]

        all_trips = []
        for _, row in df.iterrows():
            try:
                polyline = json.loads(row['POLYLINE'])
            except Exception:
                continue
            if len(polyline) < 5:
                continue
            coords = [(pt[1], pt[0]) for pt in polyline]  # (lat, lon)
            ts_start = float(row['TIMESTAMP'])
            timestamps = [ts_start + i * 15.0 for i in range(len(coords))]  # 15s sampling
            duration = timestamps[-1] - timestamps[0]
            all_trips.append((coords, timestamps, duration))

        # Compute OD pair stats for time-warp detection
        od_durations: Dict[Tuple, List[float]] = {}
        for coords, timestamps, duration in all_trips:
            od_key = (
                round(coords[0][0], 2), round(coords[0][1], 2),
                round(coords[-1][0], 2), round(coords[-1][1], 2)
            )
            od_durations.setdefault(od_key, []).append(duration)
        od_stats = {
            k: (float(np.mean(v)), float(np.std(v)))
            for k, v in od_durations.items() if len(v) >= 3
        }

        # Compute speed stats for data-adaptive speed anomaly detection
        all_max_speeds = []
        for coords, timestamps, duration in all_trips[:10000]:
            speeds = compute_segment_speeds(coords, timestamps)
            if speeds:
                all_max_speeds.append(max(speeds))
        speed_median = float(np.median(all_max_speeds)) if all_max_speeds else 0.0
        speed_std = float(np.std(all_max_speeds)) if all_max_speeds else 0.0

        # Assign labels
        normals, anomalies = [], []
        for coords, timestamps, duration in all_trips:
            detour = is_detour_anomaly(coords, self.detour_threshold)
            speed = is_speed_anomaly(coords, timestamps, speed_median, speed_std)
            loop = is_loop_anomaly(coords)
            od_key = (
                round(coords[0][0], 2), round(coords[0][1], 2),
                round(coords[-1][0], 2), round(coords[-1][1], 2)
            )
            mu, sigma = od_stats.get(od_key, (duration, 0.0))
            timewarp = is_time_warp_anomaly(duration, mu, sigma)

            subtype = 'normal'
            label = 0
            if detour:
                label, subtype = 1, 'route_deviation'
            elif speed:
                label, subtype = 1, 'kinematic'
            elif loop:
                label, subtype = 1, 'route_deviation'
            elif timewarp:
                label, subtype = 1, 'temporal'

            if label == 0:
                normals.append((coords, timestamps, 0, 'normal'))
            else:
                anomalies.append((coords, timestamps, 1, subtype))

        # Balance to anomaly_fraction
        n_anom_target = int(len(normals) * self.anomaly_fraction / (1 - self.anomaly_fraction))
        rng.shuffle(anomalies)
        anomalies = anomalies[:n_anom_target]

        all_samples = normals + anomalies
        rng.shuffle(all_samples)

        # Train/val/test split: 70/15/15
        n = len(all_samples)
        if self.split == 'train':
            self.samples = all_samples[:int(0.7 * n)]
        elif self.split == 'val':
            self.samples = all_samples[int(0.7 * n):int(0.85 * n)]
        else:
            self.samples = all_samples[int(0.85 * n):]


class TDriveDataset(TrajectoryDataset):
    DOMAIN_ID = 1
    NAME = 'tdrive'

    def _load(self, data_path: str):
        """Load T-Drive txt files. Each file: taxi_id, datetime, lon, lat"""
        rng = random.Random(self.seed)
        tdrive_dir = os.path.join(data_path, 'tdrive')
        if not os.path.exists(tdrive_dir):
            raise FileNotFoundError(f"T-Drive data not found at {tdrive_dir}")

        import glob
        from datetime import datetime

        files = sorted(glob.glob(os.path.join(tdrive_dir, '*.txt')))[:500]  # use 500 taxis

        # Segment taxi streams into individual trips (gap > 30 min = new trip)
        all_trips = []
        for fpath in files:
            try:
                df = pd.read_csv(fpath, header=None,
                                 names=['taxi_id', 'datetime', 'lon', 'lat'])
                df['ts'] = pd.to_datetime(df['datetime']).apply(lambda x: x.timestamp())
                df = df.sort_values('ts')
                coords_all = list(zip(df['lat'], df['lon']))
                ts_all = list(df['ts'])

                # segment
                seg_coords, seg_ts = [], []
                for i, (c, t) in enumerate(zip(coords_all, ts_all)):
                    if seg_ts and (t - seg_ts[-1]) > 1800:
                        if len(seg_coords) >= 5:
                            all_trips.append((seg_coords, seg_ts))
                        seg_coords, seg_ts = [], []
                    seg_coords.append(c)
                    seg_ts.append(t)
                if len(seg_coords) >= 5:
                    all_trips.append((seg_coords, seg_ts))
            except Exception:
                continue

        # OD stats for time-warp
        od_durations = {}
        for coords, timestamps in all_trips:
            dur = timestamps[-1] - timestamps[0]
            od_key = (round(coords[0][0], 2), round(coords[0][1], 2),
                      round(coords[-1][0], 2), round(coords[-1][1], 2))
            od_durations.setdefault(od_key, []).append(float(dur))
        od_stats = {k: (float(np.mean(v)), float(np.std(v)))
                    for k, v in od_durations.items() if len(v) >= 3}

        # Compute speed stats for data-adaptive speed anomaly detection
        all_max_speeds = []
        for coords, timestamps in all_trips[:10000]:
            speeds = compute_segment_speeds(coords, timestamps)
            if speeds:
                all_max_speeds.append(max(speeds))
        speed_median = float(np.median(all_max_speeds)) if all_max_speeds else 0.0
        speed_std = float(np.std(all_max_speeds)) if all_max_speeds else 0.0

        normals, anomalies = [], []
        for coords, timestamps in all_trips:
            duration = timestamps[-1] - timestamps[0]
            detour = is_detour_anomaly(coords, self.detour_threshold)
            speed = is_speed_anomaly(coords, timestamps, speed_median, speed_std)
            loop = is_loop_anomaly(coords)
            od_key = (round(coords[0][0], 2), round(coords[0][1], 2),
                      round(coords[-1][0], 2), round(coords[-1][1], 2))
            mu, sigma = od_stats.get(od_key, (duration, 0.0))
            timewarp = is_time_warp_anomaly(duration, mu, sigma)

            label, subtype = 0, 'normal'
            if detour:
                label, subtype = 1, 'route_deviation'
            elif speed:
                label, subtype = 1, 'kinematic'
            elif loop:
                label, subtype = 1, 'route_deviation'
            elif timewarp:
                label, subtype = 1, 'temporal'

            if label == 0:
                normals.append((coords, timestamps, 0, 'normal'))
            else:
                anomalies.append((coords, timestamps, 1, subtype))

        n_anom_target = int(len(normals) * self.anomaly_fraction / (1 - self.anomaly_fraction))
        rng.shuffle(anomalies)
        anomalies = anomalies[:n_anom_target]
        all_samples = normals + anomalies
        rng.shuffle(all_samples)

        n = len(all_samples)
        if self.split == 'train':
            self.samples = all_samples[:int(0.7 * n)]
        elif self.split == 'val':
            self.samples = all_samples[int(0.7 * n):int(0.85 * n)]
        else:
            self.samples = all_samples[int(0.85 * n):]


class GeoLifeDataset(TrajectoryDataset):
    DOMAIN_ID = 2
    NAME = 'geolife'

    def _load(self, data_path: str):
        """Load GeoLife .plt files with optional mode labels."""
        rng = random.Random(self.seed)
        geolife_dir = os.path.join(data_path, 'geolife', 'Data')
        if not os.path.exists(geolife_dir):
            raise FileNotFoundError(f"GeoLife data not found at {geolife_dir}")

        import glob
        from datetime import datetime

        # Build user habitual routes (top-5 most visited route centroids)
        user_routes: Dict[str, List] = {}
        all_trips = []

        user_dirs = sorted(glob.glob(os.path.join(geolife_dir, '*')))[:182]
        for user_dir in user_dirs:
            user_id = os.path.basename(user_dir)
            plt_files = sorted(glob.glob(os.path.join(user_dir, 'Trajectory', '*.plt')))

            # Load mode labels if available
            label_file = os.path.join(user_dir, 'labels.txt')
            label_df = None
            if os.path.exists(label_file):
                try:
                    label_df = pd.read_csv(label_file, sep='\t', skiprows=1,
                                           names=['start_time', 'end_time', 'mode'])
                    label_df['start_ts'] = pd.to_datetime(
                        label_df['start_time']).apply(lambda x: x.timestamp())
                    label_df['end_ts'] = pd.to_datetime(
                        label_df['end_time']).apply(lambda x: x.timestamp())
                except Exception:
                    label_df = None

            user_coords_all = []
            for plt_file in plt_files[:50]:  # limit per user
                try:
                    df = pd.read_csv(plt_file, skiprows=6, header=None,
                                     names=['lat', 'lon', '_', 'alt', 'days', 'date', 'time'])
                    df['datetime_str'] = df['date'].astype(str) + ' ' + df['time'].astype(str)
                    df['ts'] = pd.to_datetime(df['datetime_str']).apply(lambda x: x.timestamp())
                    coords = list(zip(df['lat'].astype(float), df['lon'].astype(float)))
                    timestamps = list(df['ts'].astype(float))
                    if len(coords) < 5:
                        continue

                    user_coords_all.extend(coords)

                    # Determine mode label
                    mode = 'unknown'
                    if label_df is not None and len(timestamps) > 0:
                        t_mid = (timestamps[0] + timestamps[-1]) / 2
                        mask = (label_df['start_ts'] <= t_mid) & (label_df['end_ts'] >= t_mid)
                        if mask.any():
                            mode = label_df[mask].iloc[0]['mode'].lower().strip()

                    # Compute mean speed
                    total_dist = trajectory_length_m(coords)
                    duration = max(timestamps[-1] - timestamps[0], 1)
                    mean_speed = total_dist / duration

                    all_trips.append((user_id, coords, timestamps, mode, mean_speed))
                except Exception:
                    continue

            # Build habitual routes: cluster user coords into top-5 centroid routes
            if len(user_coords_all) > 10:
                sample_coords = user_coords_all[::max(1, len(user_coords_all) // 100)]
                user_routes[user_id] = sample_coords[:100]

        # Assign labels
        normals, anomalies = [], []
        for user_id, coords, timestamps, mode, mean_speed in all_trips:
            label, subtype = 0, 'normal'
            mode_anom = (mode != 'unknown') and is_mode_inconsistency(mode, mean_speed)
            routine_anom = is_off_routine(
                coords[len(coords) // 2][0], coords[len(coords) // 2][1],
                [user_routes.get(user_id, [])]
            )
            if mode_anom:
                label, subtype = 1, 'kinematic'
            elif routine_anom:
                label, subtype = 1, 'route_deviation'

            if label == 0:
                normals.append((coords, timestamps, 0, 'normal'))
            else:
                anomalies.append((coords, timestamps, 1, subtype))

        n_anom_target = int(len(normals) * self.anomaly_fraction / (1 - self.anomaly_fraction))
        rng.shuffle(anomalies)
        anomalies = anomalies[:n_anom_target]
        all_samples = normals + anomalies
        rng.shuffle(all_samples)

        n = len(all_samples)
        if self.split == 'train':
            self.samples = all_samples[:int(0.7 * n)]
        elif self.split == 'val':
            self.samples = all_samples[int(0.7 * n):int(0.85 * n)]
        else:
            self.samples = all_samples[int(0.85 * n):]


def get_dataset(name: str, data_path: str, split: str = 'train',
                max_len: int = 128, anomaly_fraction: float = 0.05,
                seed: int = 42, detour_threshold: float = 1.5) -> TrajectoryDataset:
    cls = {'porto': PortoDataset, 'tdrive': TDriveDataset, 'geolife': GeoLifeDataset}[name]
    return cls(data_path, split=split, max_len=max_len,
               anomaly_fraction=anomaly_fraction, seed=seed,
               detour_threshold=detour_threshold)
