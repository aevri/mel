"""Mel - a command-line utility to help with mole management."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import mel.cmd.addcluster
import mel.cmd.addsingle
import mel.cmd.list
import mel.cmd.microadd
import mel.cmd.microview
import mel.cmd.rotomapedit
import mel.cmd.rotomapmolepicker
import mel.cmd.rotomaprelate
import mel.cmd.rotomapshow
import mel.cmd.rotomapuuid


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    subparsers = parser.add_subparsers()

    _setup_parser_for_module(subparsers, mel.cmd.addcluster, 'add-cluster')
    _setup_parser_for_module(subparsers, mel.cmd.addsingle, 'add-single')
    _setup_parser_for_module(subparsers, mel.cmd.list, 'list')
    _setup_parser_for_module(subparsers, mel.cmd.microadd, 'micro-add')
    _setup_parser_for_module(subparsers, mel.cmd.microview, 'micro-view')
    _setup_parser_for_module(subparsers, mel.cmd.rotomapedit, 'rotomap-edit')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapmolepicker, 'rotomap-molepicker')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomaprelate, 'rotomap-relate')
    _setup_parser_for_module(subparsers, mel.cmd.rotomapshow, 'rotomap-show')
    _setup_parser_for_module(subparsers, mel.cmd.rotomapuuid, 'rotomap-uuid')

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
