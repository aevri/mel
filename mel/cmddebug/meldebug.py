"""Mel Debug - for debugging the internals of the 'mel' command."""


import argparse
import sys

import mel.cmddebug.benchautomark
import mel.cmddebug.benchautopaste
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
        subparsers, mel.cmddebug.benchautopaste, 'bench-autopaste'
    )
    _setup_parser_for_module(
        subparsers, mel.cmddebug.rendervaluefield, 'render-valuefield'
    )

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
        epilog=doc_epilog,
    )
    module.setup_parser(parser)
    parser.set_defaults(func=module.process_args)


if __name__ == '__main__':
    sys.exit(main())
# -----------------------------------------------------------------------------
# Copyright (C) 2016-2017 Angelos Evripiotis.
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
