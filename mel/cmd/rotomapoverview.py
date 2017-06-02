"""Display an overview of all the rotomaps at the paths."""

import os
import re


def setup_parser(parser):
    parser.add_argument(
        'PATH',
        nargs='*',
        default=['.'],
        help="Path to look for rotomaps.")


def process_args(args):
    rotomap_re = re.compile('\d{8}T\d{4}')

    for path in args.PATH:
        for root, dirs, files in os.walk(path):
            if dirs:
                pass
            if not files:
                pass
            if rotomap_re.fullmatch(os.path.basename(root)):
                overview = make_overview(root, files)
                print(root, overview)


def make_overview(path, files):
    angles = {f: None for f in files if f.lower().endswith('.jpg')}

    for f in files:
        if f.lower().endswith('.json'):
            jpg_name = f[:-5]
            if jpg_name in angles:
                angles[jpg_name] = f

    results = []
    for a in sorted(angles):
        data_path = angles[a]
        if data_path is None:
            results.append('-')
        else:
            results.append('+')

    return(''.join(results))
