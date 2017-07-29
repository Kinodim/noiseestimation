#!/usr/bin/env python
from __future__ import print_function
import os
import argparse
import functools
import json

desired_keys = []


def process_content(content, desired_keys):
    res = {}
    for line in content.splitlines():
        key, val = line.split(":", 1)
        val = val.strip()
        if key in desired_keys:
            res[key] = val
        if not keep_entry(key, val):
            return {}
    return res


def keep_entry(key, val):
    if key != "time":
        return True
    if val == "0":
        return False
    if val == keep_entry.last_time:
        return False

    keep_entry.last_time = val
    return True
keep_entry.last_time = "0"


def iterate_dir(dirname, processor):
    """Takes relative path
    """
    path = os.path.join(os.getcwd(), args.data_folder)
    if not os.path.isdir(path):
        print("Error! Specified folder does not exist!")
        return

    contents = []
    for fn in sorted(os.listdir(path)):
        file_path = os.path.join(path, fn)
        with open(file_path, "r") as f:
            content = f.read()
        res = processor(content)
        if res:
            contents.append(res)
    return contents


def write_json(contents, filename):
    file_path = os.path.join(os.getcwd(), filename)
    json_str = json.dumps(contents, indent=4)
    with open(file_path, "w") as f:
        f.write(json_str)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge car sensor data into single file")
    parser.add_argument(dest="data_folder", action="store", type=str,
                        help="Folder containing data files")
    parser.add_argument(dest="output_file", action="store", type=str,
                        help="Output file name")
    parser.add_argument(dest="fields", action="store", nargs="+", type=str,
                        help="Data fields to store")
    args = parser.parse_args()

    processor = functools.partial(process_content, desired_keys=args.fields)
    contents = iterate_dir(args.data_folder, processor)
    print("Entries: ", len(contents))
    write_json(contents, args.output_file)
