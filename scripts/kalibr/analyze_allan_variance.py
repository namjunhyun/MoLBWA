#!/usr/bin/env python3
"""Compute IMU noise density / random walk from a long stationary rosbag2
recording of /imu, and write them into calibration/imu.yaml in Kalibr's
format (accelerometer_noise_density, accelerometer_random_walk,
gyroscope_noise_density, gyroscope_random_walk).

Requires the same isolated venv used for bag conversion (rosbags + allantools):
    calibration/.convert_venv/bin/pip install rosbags allantools

Usage:
    calibration/.convert_venv/bin/python3 scripts/kalibr/analyze_allan_variance.py \
        calibration/bags/allan_imu_<timestamp> [--write]

Without --write, only prints the fitted values and a sanity summary (recording
duration, sample counts) -- doesn't touch imu.yaml. Use --write once the
recording is long enough (multiple hours recommended) and the fit looks sane.

Method (standard Allan-deviation IMU noise ID, same convention as
ori-drs/allan_variance_ros and most IEEE-STD-952-derived tools):
  - Overlapping Allan deviation (allantools.oadev) per axis, gyro and accel.
  - White noise / (angle|velocity) random walk coefficient N: fit slope -1/2
    line to the short-tau region, read off at tau=1s.
  - Rate random walk coefficient K: fit slope +1/2 line to the region past
    the ADEV curve's minimum (bias instability trough), read off at tau=3s
    (sigma(tau) = K * sqrt(tau/3)).
  - Per-sensor N/K are averaged across the 3 axes (x,y,z) for a single
    scalar noise_density / random_walk, matching what Kalibr's imu.yaml
    expects (it doesn't support per-axis noise).
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import yaml
import allantools
from rosbags.highlevel import AnyReader


def load_imu(bag_path):
    t, gx, gy, gz, ax, ay, az = [], [], [], [], [], [], []
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == "/imu"]
        if not conns:
            raise SystemExit(f"No /imu topic found in {bag_path}")
        for connection, timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            t.append(timestamp * 1e-9)
            gx.append(msg.angular_velocity.x)
            gy.append(msg.angular_velocity.y)
            gz.append(msg.angular_velocity.z)
            ax.append(msg.linear_acceleration.x)
            ay.append(msg.linear_acceleration.y)
            az.append(msg.linear_acceleration.z)
    t = np.array(t)
    return t, np.array([gx, gy, gz]), np.array([ax, ay, az])


def fit_white_noise(taus, adevs):
    """Fit slope -1/2 over the shortest-tau third of the curve; return N (ADEV at tau=1)."""
    n = max(3, len(taus) // 3)
    log_t = np.log10(taus[:n])
    log_a = np.log10(adevs[:n])
    slope, intercept = np.polyfit(log_t, log_a, 1)
    # sigma(tau) = N * tau^-0.5 -> log(sigma) = log(N) - 0.5*log(tau)
    # Use the actual fitted intercept at log10(tau)=0 regardless of fitted slope,
    # but report the slope too so the caller can sanity-check it's near -0.5.
    N = 10 ** intercept
    return N, slope


def fit_rate_random_walk(taus, adevs):
    """Find the ADEV minimum (bias instability trough), fit slope +1/2 just past it,
    return K such that sigma(tau) = K * sqrt(tau/3)."""
    min_idx = int(np.argmin(adevs))
    lo = min_idx
    hi = min(len(taus), min_idx + max(3, (len(taus) - min_idx) // 2))
    if hi - lo < 3:
        return None, None
    log_t = np.log10(taus[lo:hi])
    log_a = np.log10(adevs[lo:hi])
    slope, intercept = np.polyfit(log_t, log_a, 1)
    # sigma(tau) = K * sqrt(tau/3) -> log(sigma) = log(K) - 0.5*log(3) + 0.5*log(tau)
    logK = intercept + 0.5 * np.log10(3)
    K = 10 ** logK
    return K, slope


def analyze_axis(rate_hz, data):
    taus, adevs, errs, ns = allantools.oadev(data, rate=rate_hz, data_type="freq", taus="octave")
    N, white_slope = fit_white_noise(taus, adevs)
    K, rw_slope = fit_rate_random_walk(taus, adevs)
    return N, K, white_slope, rw_slope, taus, adevs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_path")
    ap.add_argument("--imu-yaml", default="calibration/imu.yaml")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    t, gyro, accel = load_imu(args.bag_path)
    duration_s = t[-1] - t[0]
    n = len(t)
    rate_hz = n / duration_s
    print(f"Loaded {n} IMU samples over {duration_s:.1f}s ({duration_s/3600:.2f}h), "
          f"measured rate {rate_hz:.2f} Hz")
    if duration_s < 3600:
        print("WARNING: recording is under 1 hour. Allan variance random-walk (K) "
              "estimates need several hours to be reliable; treat results as rough "
              "until the recording is longer.")

    gyro_N, gyro_K = [], []
    accel_N, accel_K = [], []
    for axis, label in zip(range(3), "xyz"):
        N, K, ws, rws, taus, adevs = analyze_axis(rate_hz, gyro[axis])
        print(f"gyro {label}: N={N:.6g} rad/s/sqrt(Hz) (white slope {ws:.2f}), "
              f"K={K if K is None else f'{K:.6g}'} rad/s^2/sqrt(Hz) (rw slope {rws if rws is not None else float('nan'):.2f})")
        gyro_N.append(N)
        if K is not None:
            gyro_K.append(K)

    for axis, label in zip(range(3), "xyz"):
        N, K, ws, rws, taus, adevs = analyze_axis(rate_hz, accel[axis])
        print(f"accel {label}: N={N:.6g} m/s^2/sqrt(Hz) (white slope {ws:.2f}), "
              f"K={K if K is None else f'{K:.6g}'} m/s^3/sqrt(Hz) (rw slope {rws if rws is not None else float('nan'):.2f})")
        accel_N.append(N)
        if K is not None:
            accel_K.append(K)

    result = {
        "gyroscope_noise_density": float(np.mean(gyro_N)),
        "gyroscope_random_walk": float(np.mean(gyro_K)) if gyro_K else None,
        "accelerometer_noise_density": float(np.mean(accel_N)),
        "accelerometer_random_walk": float(np.mean(accel_K)) if accel_K else None,
        "update_rate": round(rate_hz, 1),
    }
    print("\nAveraged across axes:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if args.write:
        if any(v is None for v in result.values()):
            raise SystemExit("Refusing to write: some values are None (recording likely too "
                              "short for a random-walk fit). Record longer and retry.")
        out = {
            "rostopic": "/imu",
            "update_rate": result["update_rate"],
            "accelerometer_noise_density": result["accelerometer_noise_density"],
            "accelerometer_random_walk": result["accelerometer_random_walk"],
            "gyroscope_noise_density": result["gyroscope_noise_density"],
            "gyroscope_random_walk": result["gyroscope_random_walk"],
        }
        with open(args.imu_yaml, "w") as f:
            yaml.dump(out, f, default_flow_style=False, sort_keys=False)
        print(f"\nWrote {args.imu_yaml}")
    else:
        print("\n(dry run -- pass --write to update", args.imu_yaml, ")")


if __name__ == "__main__":
    main()
