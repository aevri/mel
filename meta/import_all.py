#! /usr/bin/env python
# encoding: utf-8
"""Imports all the modules under the specified path.

This can be useful as a basic static analysis test, assuming that the imports
do not have side-effects.

"""

from __future__ import print_function

import argparse
import importlib
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "PATH",
        help="path to the package to import from")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true")
    args = parser.parse_args()

    parent_dir, package = os.path.split(args.PATH)

    # Python looks in sys.path to find modules to import, if we don't do this
    # then it probably won't find any of the modules under parent_dir.
    sys.path.append(os.path.abspath(parent_dir))

    os.chdir(parent_dir)

    for root, _, files in os.walk(package):
        for f in files:
            if not f.endswith('.py'):
                continue
            module_name = f[:-3]
            module_list = root.split('/')

            if not module_name == '__init__':
                module_list.append(module_name)

            module_ref = '.'.join(module_list)
            if args.verbose:
                print(module_ref)
            importlib.import_module(module_ref)


if __name__ == "__main__":
    sys.exit(main())
