"""Display an overview of all the rotomaps at the paths."""

import os


def setup_parser(parser):
    parser.add_argument(
        'PATH',
        nargs='+',
        help="Path to the rotomap.")


def process_args(args):
    for path in args.PATH:
        files = os.listdir(path)

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

        print(path, ''.join(results))
