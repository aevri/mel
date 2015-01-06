"""Mel - a command-line utility to help with mole management."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import mel.cmd.addcluster
import mel.cmd.addsingle


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    subparsers = parser.add_subparsers()

    _setup_parser_for_module(subparsers, mel.cmd.addcluster, 'add-cluster')
    _setup_parser_for_module(subparsers, mel.cmd.addsingle, 'add-single')

    args = parser.parse_args()
    return args.func(args)


def _setup_parser_for_module(subparsers, module, name):
    doc = module.__doc__
    doc_subject = doc.splitlines()[0]
    doc_epilog = '\n'.join(doc.splitlines()[1:])
    parser = subparsers.add_parser(
        name,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help=doc_subject,
        description=doc_subject,
        epilog=doc_epilog)
    module.setup_parser(parser)
    parser.set_defaults(func=module.process_args)
