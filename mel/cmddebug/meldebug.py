"""Mel Debug - for debugging the internals of the 'mel' command."""


import argparse
import sys

import mel.lib.ui

import mel.cmd.error

import mel.cmddebug.benchautomark
import mel.cmddebug.rendervaluefield


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

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
        subparsers, mel.cmddebug.benchautomark, 'bench-automark'
    )
    _setup_parser_for_module(
        subparsers, mel.cmddebug.rendervaluefield, 'render-valuefield'
    )

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
        epilog=doc_epilog,
    )
    module.setup_parser(parser)
    parser.set_defaults(func=module.process_args)


if __name__ == '__main__':
    sys.exit(main())


# -----------------------------------------------------------------------------
# Copyright (C) 2016-2018 Angelos Evripiotis.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------ END-OF-FILE ----------------------------------
