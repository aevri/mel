"""Mel Debug - for debugging the internals of the 'mel' command."""


import argparse

import mel.cmddebug.benchrelate
import mel.cmddebug.rendervaluefield
import mel.cmddebug.triangulate


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    subparsers = parser.add_subparsers()

    # Work around a bug in argparse with subparsers no longer being required:
    # http://bugs.python.org/issue9253#msg186387
    subparsers.required = True
    subparsers.dest = 'command'

    # vulture will report these as unused unless we do this
    #
    # pylint: disable=pointless-statement
    subparsers.required
    subparsers.dest
    # pylint: enable=pointless-statement

    _setup_parser_for_module(
        subparsers, mel.cmddebug.benchrelate, 'bench-relate')
    _setup_parser_for_module(
        subparsers, mel.cmddebug.rendervaluefield, 'render-valuefield')
    _setup_parser_for_module(
        subparsers, mel.cmddebug.triangulate, 'triangulate')

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
