#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import argparse
import json


def write_json(contents, filename):
    file_path = os.path.join(os.getcwd(), filename)
    json_str = json.dumps(contents, indent=4)
    with open(file_path, "w") as f:
        f.write(json_str)
    return


def read_json(filename):
    file_path = os.path.join(os.getcwd(), filename)
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def read_timestamps(filename):
    file_path = os.path.join(os.getcwd(), filename)
    timestamps = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        timestamp = int(line)
        timestamps.append(timestamp)

    return timestamps


def combine(readings, timestamps):
    """Extend readings dict with timestamps
    """

    min_length = min(len(readings), len(timestamps))
    for idx in range(min_length):
        readings[idx]["time"] = timestamps[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add timestamps from separate file to sensor readings file")
    parser.add_argument(dest="sensor_file", action="store",
                        type=str, metavar="ReadingsFile",
                        help="File containing sensor readings")
    parser.add_argument(dest="timestamp_file", action="store",
                        type=str, metavar="TimestampFile",
                        help="File containing reading timestamps")
    parser.add_argument(dest="output_file", action="store",
                        type=str, metavar="OutputFile",
                        help="Output file")
    args = parser.parse_args()

    readings = read_json(args.sensor_file)
    timestamps = read_timestamps(args.timestamp_file)
    combine(readings, timestamps)
    write_json(readings, args.output_file)
