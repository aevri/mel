"""Mel - a command-line utility to help with mole management."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    args = parser.parse_args()
