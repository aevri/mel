"""Mel - a command-line utility to help with mole management."""


import argparse
import sys


OLD_PYTHON_MESSAGE="""You are running mel on an old Python interpreter

Unfortunately Frobulator 6.0 and above are not compatible with Python < 3.6
anymore, and you still ended up with this version installed on your system.
Make sure you have pip >= 9.0 to avoid this kind of issues, as well as
setuptools >= 24.2:

 $ pip install pip setuptools --upgrade

Upgrade your system to use Python 3.6 or later to run this version of mel.

"""

if sys.version_info < (3,6):
    raise ImportError(OLD_PYTHON_MESSAGE)


import mel.cmd.addcluster
import mel.cmd.addsingle
import mel.cmd.error
import mel.cmd.list
import mel.cmd.microadd
import mel.cmd.microcompare
import mel.cmd.microview
import mel.cmd.rotomapautomark
import mel.cmd.rotomapautomask
import mel.cmd.rotomapautomasksvm
import mel.cmd.rotomapcompare
import mel.cmd.rotomapconfirm
import mel.cmd.rotomapdiff
import mel.cmd.rotomapedit
import mel.cmd.rotomapfsck
import mel.cmd.rotomapidentify
import mel.cmd.rotomaplist
import mel.cmd.rotomapmontagesingle
import mel.cmd.rotomaporganise
import mel.cmd.rotomapoverview
import mel.cmd.rotomaprelate
import mel.cmd.rotomapshow
import mel.cmd.rotomapudiff
import mel.cmd.rotomapuuid


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

    _setup_parser_for_module(subparsers, mel.cmd.addcluster, 'add-cluster')
    _setup_parser_for_module(subparsers, mel.cmd.addsingle, 'add-single')
    _setup_parser_for_module(subparsers, mel.cmd.list, 'list')
    _setup_parser_for_module(subparsers, mel.cmd.microadd, 'micro-add')
    _setup_parser_for_module(subparsers, mel.cmd.microcompare, 'micro-compare')
    _setup_parser_for_module(subparsers, mel.cmd.microview, 'micro-view')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapautomark, 'rotomap-automark')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapautomask, 'rotomap-automask')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapautomasksvm, 'rotomap-automask-svm')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapcompare, 'rotomap-compare')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapconfirm, 'rotomap-confirm')
    _setup_parser_for_module(subparsers, mel.cmd.rotomapdiff, 'rotomap-diff')
    _setup_parser_for_module(subparsers, mel.cmd.rotomapedit, 'rotomap-edit')
    _setup_parser_for_module(subparsers, mel.cmd.rotomapfsck, 'rotomap-fsck')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapidentify, 'rotomap-identify')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomaplist, 'rotomap-list')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapmontagesingle, 'rotomap-montage-single')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomaporganise, 'rotomap-organise')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapoverview, 'rotomap-overview')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomaprelate, 'rotomap-relate')
    _setup_parser_for_module(subparsers, mel.cmd.rotomapshow, 'rotomap-show')
    _setup_parser_for_module(subparsers, mel.cmd.rotomapudiff, 'rotomap-udiff')
    _setup_parser_for_module(subparsers, mel.cmd.rotomapuuid, 'rotomap-uuid')

    args = parser.parse_args()
    try:
        return args.func(args)
    except mel.cmd.error.UsageError as e:
        print('Usage error:', e, file=sys.stderr)
        return 2
    except BrokenPipeError:
        # Silently exit on broken pipes, e.g. when our output is piped to head.

        # Explicitly close stderr before exiting, to avoid an additional
        # message from Python on stderr about the pipe break being ignored.
        # http://bugs.python.org/issue11380,#msg153320
        sys.stderr.close()
    except mel.lib.ui.AbortKeyInterruptError:
        # Using this return code may also break us out of an outer loop, e.g.
        # 'xargs' will stop processing if the program it calls exists with 255.
        return 255


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


if __name__ == '__main__':
    sys.exit(main())
