"""Pick a particular mole from a rotomap and print it's details."""


import cv2

import mel.lib.common
import mel.lib.image
import mel.lib.math
import mel.lib.ui

import mel.rotomap.display


def setup_parser(parser):
    parser.add_argument(
        '--images',
        nargs='+',
        action='append',
        required=True,
        help="A list of paths to images, specify multiple times for multiple "
             "sets.")
    parser.add_argument(
        '--display-width',
        type=int,
        default=None,
        help="Width of the preview display window.")
    parser.add_argument(
        '--display-height',
        type=int,
        default=None,
        help="Width of the preview display window.")
    parser.add_argument(
        '--loop',
        action='store_true',
        help='Do not exit after picking the first mole, loop til user quits.')
    parser.add_argument(
        '--copy-to-clipboard',
        action='store_true',
        help='Copy ids to the clipboard, as well as printing. Mac OSX only.')


def process_args(args):
    editor = mel.rotomap.display.Editor(
        args.images, args.display_width, args.display_height)

    mel.lib.ui.bring_python_to_front()

    mole_uuid = None
    is_finished = False

    def mouse_callback(event, x, y, flags, _param):
        del _param
        nonlocal mole_uuid
        nonlocal is_finished
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                editor.show_zoomed(x, y)
            else:
                mole_uuid = editor.get_mole_uuid(x, y)
                is_finished = True

    editor.display.set_mouse_callback(mouse_callback)

    while True:
        is_finished = False
        while not is_finished:
            key = cv2.waitKey(50)
            if key != -1:
                if key == mel.lib.ui.WAITKEY_LEFT_ARROW:
                    editor.show_prev()
                elif key == mel.lib.ui.WAITKEY_RIGHT_ARROW:
                    editor.show_next()
                elif key == ord(' '):
                    editor.show_fitted()
                elif key == 13:
                    editor.toggle_markers()
                else:
                    is_finished = True

        if mole_uuid is None:
            return 1

        print(mole_uuid)
        if args.copy_to_clipboard:
            mel.lib.ui.set_clipboard_contents(mole_uuid)

        if not args.loop:
            return 0
