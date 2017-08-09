#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import argparse
import json


def process_content(content):
    desired_keys = args.fields
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
    # filter out entries with no timestamp or those with same
    # timestamp as previous entry
    if key != "time":
        return True
    if val == "0":
        return False
    if val == keep_entry.last_time:
        return False

    keep_entry.last_time = val
    return True
keep_entry.last_time = "0"


def iterate_dir(dirname):
    """Takes relative path
    """
    path = os.path.join(os.getcwd(), dirname)
    if not os.path.isdir(path):
        print("Error: Specified folder %s does not exist" % dirname)
        sys.exit(-1)

    contents = []
    for fn in sorted(os.listdir(path)):
        file_path = os.path.join(path, fn)
        with open(file_path, "r") as f:
            content = f.read()
        res = process_content(content)
        if res:
            contents.append(res)
    return contents


def read_dirs(dirnames):
    res = iterate_dir(dirnames[0])

    for dirname in dirnames[1:]:
        contents = iterate_dir(dirname)
        for idx, entry in enumerate(res):
            if idx >= len(contents):
                print("Error: No matching file for index %d in directory %s" %
                      (idx, dirname))
                break
            entry.update(contents[idx])

    return res


def write_json(contents, filename):
    file_path = os.path.join(os.getcwd(), filename)
    json_str = json.dumps(contents, indent=4)
    with open(file_path, "w") as f:
        f.write(json_str)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge car sensor data into single file")
    parser.add_argument("-d", dest="data_folders", action="store", nargs="+",
                        type=str, required=True, metavar="DataFolder",
                        help="Folder(s) containing data files")
    parser.add_argument("-o", dest="output_file", action="store", type=str,
                        required=True, metavar="OutputFile",
                        help="Output file name")
    parser.add_argument(dest="fields", action="store", nargs="+",
                        type=str,
                        metavar="Field", help="Data fields to store")
    args = parser.parse_args()

    contents = read_dirs(args.data_folders)
    print("Entries: ", len(contents))
    write_json(contents, args.output_file)
