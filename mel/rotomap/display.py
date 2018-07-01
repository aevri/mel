"""Display a rotomap."""

import enum
import functools

import collections
import cv2
import numpy

import mel.lib.common
import mel.lib.image
import mel.lib.ui
import mel.rotomap.detectmoles
import mel.rotomap.mask
import mel.rotomap.moles
import mel.rotomap.relate
import mel.rotomap.tricolour


DEFAULT_MASKER_RADIUS = 200


def draw_mole(image, x, y, colours):

    radius = 16
    for index in range(3):
        cv2.circle(image, (x, y), radius, colours[index], -1)
        radius -= 4


def draw_non_canonical_mole(image, x, y, colours):
    radius = 16
    thickness = 4
    for index in range(3):
        top_left = (x - radius, y - radius)
        bottom_right = (x + radius, y + radius)
        cv2.rectangle(image, top_left, bottom_right, colours[index], thickness)
        radius -= thickness


def draw_crosshair(image, x, y):
    inner_radius = 16
    outer_radius = 24
    white = (255, 255, 255)
    black = (0, 0, 0)

    directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # Right, down, left, up

    size_color_list = [(3, white), (2, black)]

    for size, color in size_color_list:
        for d in directions:
            cv2.line(
                image,
                (x + (inner_radius * d[0]), y + (inner_radius * d[1])),
                (x + (outer_radius * d[0]), y + (outer_radius * d[1])),
                color,
                size)


class Display:

    def __init__(self, width, height):
        self._name = str(id(self))

        if width is None or height is None:
            full_width_height = mel.lib.ui.guess_fullscreen_width_height()
            if width is None:
                width = full_width_height[0]
            if height is None:
                height = full_width_height[1]

        self._rect = numpy.array((width, height))

        cv2.namedWindow(self._name)
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._name, *self._rect)

        self._transform = None

        self._zoom_pos = None
        self._is_zoomed = False

    def show_current(self, image, overlay):

        if not self._is_zoomed:
            self._transform = FittedImageTransform(
                image, self._rect)
        else:
            self._transform = ZoomedImageTransform(
                image, self._zoom_pos, self._rect)

        image = self._transform.render()
        if overlay is not None:
            image = overlay(image, self._transform)

        cv2.imshow(self._name, image)

    def set_fitted(self):
        self._is_zoomed = False

    def set_zoomed(self, x, y):
        self._zoom_pos = numpy.array((x, y))
        self._is_zoomed = True

    def is_zoomed(self):
        return self._is_zoomed

    def get_zoom_pos(self):
        if not self._is_zoomed:
            raise Exception('Not zoomed')
        return self._zoom_pos

    def set_mouse_callback(self, callback):
        cv2.setMouseCallback(self._name, callback)

    def clear_mouse_callback(self):
        cv2.setMouseCallback(
            self._name,
            mel.lib.common.make_null_mouse_callback())

    def windowxy_to_imagexy(self, window_x, window_y):
        return self._transform.transformedxy_to_imagexy(window_x, window_y)

    def set_title(self, title):
        cv2.setWindowTitle(self._name, title)


def make_composite_overlay(*overlays):
    """Return an overlay, which will composite the supplied overlays in turn.

    :*overlays: The overlay callables to composite.
    :returns: A function which will composite *overlays and return the image.
    """
    def do_overlay(image, transform):
        for o in overlays:
            image = o(image, transform)
        return image

    return do_overlay


class StatusOverlay():

    def __init__(self):
        self.text = ""

    def __call__(self, image, transform):

        if self.text:
            text_image = mel.lib.image.render_text_as_image(self.text)
            mel.lib.common.copy_image_into_image(
                text_image, image, 0, 0)

        return image


class MoleMarkerOverlay():

    def __init__(self, uuid_to_tricolour):
        self._is_showing_markers = True
        self._is_faded_markers = True
        self._highlight_uuid = None

        self._uuid_to_tricolour = uuid_to_tricolour
        if self._uuid_to_tricolour is None:
            self._uuid_to_tricolour = (
                mel.rotomap.tricolour.uuid_to_tricolour_first_digits)

        self.moles = None

    def toggle_markers(self):
        self._is_showing_markers = not self._is_showing_markers

    def set_highlight_uuid(self, highlight_uuid):
        self._highlight_uuid = highlight_uuid

    def toggle_faded_markers(self):
        self._is_faded_markers = not self._is_faded_markers

    def __call__(self, image, transform):

        if not self._is_showing_markers:
            return image

        highlight_mole = None
        if self._highlight_uuid is not None:
            for m in self.moles:
                if m['uuid'] == self._highlight_uuid:
                    highlight_mole = m
                    break

        marker_image = image
        if self._is_faded_markers:
            marker_image = image.copy()
        for mole in self.moles:
            x, y = transform.imagexy_to_transformedxy(
                mole['x'], mole['y'])
            if mole is highlight_mole:
                draw_crosshair(marker_image, x, y)
            colours = self._uuid_to_tricolour(mole['uuid'])
            if mole[mel.rotomap.moles.KEY_IS_CONFIRMED]:
                draw_mole(marker_image, x, y, colours)
            else:
                draw_non_canonical_mole(marker_image, x, y, colours)
        if self._is_faded_markers:
            image = cv2.addWeighted(image, 0.75, marker_image, 0.25, 0.0)

        return image


class MarkedMoleOverlay():
    """An overlay to make marked moles obvious, for checking mark positions."""

    def __init__(self):
        self.moles = None
        self.is_accentuate_marked_mode = False

    def __call__(self, image, transform):
        if self.is_accentuate_marked_mode:
            return self._draw_accentuated(image, transform)
        else:
            return self._draw_markers(image, transform)

    def _draw_accentuated(self, image, transform):
        # Reveal the moles that have been marked, whilst still showing
        # markers. This is good for verifying that markers are actually
        # positioned on moles.

        mask_radius = 50

        image = image.copy() // 2
        mask = numpy.zeros((*image.shape[:2], 1), numpy.uint8)

        for mole in self.moles:
            x, y = transform.imagexy_to_transformedxy(mole['x'], mole['y'])
            draw_mole(mask, x, y, [[255] * 3] * 3)
            cv2.circle(mask, (x, y), mask_radius, 255, -1)

        masked_faded = cv2.bitwise_and(image, image, mask=mask)
        image = cv2.add(masked_faded, image)

        for mole in self.moles:
            x, y = transform.imagexy_to_transformedxy(mole['x'], mole['y'])
            draw_crosshair(image, x, y)

        return image

    def _draw_markers(self, image, transform):
        # Hide the moles that have been marked, showing markers
        # distinctly from moles. This is good for marking moles that
        # haven't been marked, without worrying about the ones that
        # have been marked.

        for mole in self.moles:
            x, y = transform.imagexy_to_transformedxy(mole['x'], mole['y'])
            draw_mole(
                image,
                x, y,
                [[255, 0, 0], [255, 128, 128], [255, 0, 0]])

        return image


class ImageRelateOverlay():
    """An overlay to make image-to-image mappings obvious."""

    def __init__(self):
        self.prev_moles = None
        self.moles = None
        self.is_target_mode = False

    def __call__(self, image, transform):

        from_moles = self.prev_moles
        to_moles = self.moles

        default_theory_uuids = set()
        best_theory = []
        from_duplicates = set()
        to_duplicates = set()

        if self.prev_moles:
            make_info = mel.rotomap.relate.mole_list_overlap_info
            _, _, _, _, default_theory_uuids = make_info(
                self.prev_moles, self.moles)

            from_counts = collections.Counter(x['uuid'] for x in from_moles)
            to_counts = collections.Counter(x['uuid'] for x in to_moles)
            from_duplicates = {u for u, c in from_counts.items() if c > 1}
            to_duplicates = {u for u, c in to_counts.items() if c > 1}

            if self.moles:
                best_theory = mel.rotomap.relate.best_theory(
                    self.prev_moles, self.moles, iterate=False)

        green = [[0, 255, 0], [128, 255, 128], [0, 255, 0]]
        red = [[0, 0, 255], [128, 128, 255], [0, 0, 255]]

        if self.is_target_mode:
            image = self._draw_accentuated(
                image, transform, default_theory_uuids)

        if self.is_target_mode:
            saved_image = image.copy()

        if self.prev_moles:
            self._draw_moles(
                image,
                transform,
                self.prev_moles,
                red,
                default_theory_uuids,
                from_duplicates)

        if not self.is_target_mode:
            image //= 2
            self._draw_moles(
                image,
                transform,
                self.moles,
                green,
                default_theory_uuids,
                to_duplicates)

        if self.prev_moles:
            from_uuid_points = mel.rotomap.moles.to_uuid_points(
                self.prev_moles)
            to_uuid_points = mel.rotomap.moles.to_uuid_points(self.moles)
            for from_uuid, to_uuid in best_theory:
                if from_uuid and to_uuid:
                    from_pos = from_uuid_points[from_uuid]
                    to_pos = to_uuid_points[to_uuid]

                    from_pos = transform.imagexy_to_transformedxy(*from_pos)
                    to_pos = transform.imagexy_to_transformedxy(*to_pos)

                    radius = 25

                    if self.is_target_mode:
                        offset = to_pos - from_pos
                        magnitude = numpy.linalg.norm(offset)
                        radius = min(radius, magnitude / 2)
                        if radius < magnitude:
                            direction = offset / magnitude
                            adjustment = (direction * radius).astype(int)
                            to_pos -= adjustment

                    colour = (255, 255, 255)
                    if from_uuid == to_uuid:
                        colour = (255, 0, 0)

                    self._arrow(image, from_pos, to_pos, colour)

        if self.is_target_mode:
            image = cv2.addWeighted(
                image, 0.25, saved_image, 0.75, 0.0)

        return image

    def _draw_moles(
            self, image, transform, moles, colour, defaults, duplicates):
        blue = [[255, 0, 0], [255, 128, 128], [255, 0, 0]]
        alert = [[0, 255, 255], [0, 0, 255], [0, 255, 255]]
        for m in moles:
            x, y = transform.imagexy_to_transformedxy(m['x'], m['y'])

            c = colour
            if m['uuid'] in defaults:
                c = blue
            if m['uuid'] in duplicates:
                c = alert

            draw_mole(image, x, y, c)

    def _arrow(self, image, from_pos, to_pos, colour):
        cv2.arrowedLine(
            image,
            tuple(from_pos),
            tuple(to_pos),
            colour,
            10,
            cv2.LINE_AA,
            tipLength=0.25)

        cv2.arrowedLine(
            image,
            tuple(from_pos),
            tuple(to_pos),
            (0, 0, 0),
            3,
            cv2.LINE_AA,
            tipLength=0.25)

    def _draw_accentuated(self, image, transform, in_both):
        # Reveal the moles that have been marked, whilst still showing
        # markers. This is good for verifying that markers are actually
        # positioned on moles.

        mask_radius = 50

        image = image.copy() // 2
        mask = numpy.zeros((*image.shape[:2], 1), numpy.uint8)

        for mole in self.moles:
            x, y = transform.imagexy_to_transformedxy(mole['x'], mole['y'])
            draw_mole(mask, x, y, [[255] * 3] * 3)
            cv2.circle(mask, (x, y), mask_radius, 255, -1)

        masked_faded = cv2.bitwise_and(image, image, mask=mask)
        image = cv2.add(masked_faded, image)

        for mole in self.moles:
            if mole['uuid'] not in in_both:
                x, y = transform.imagexy_to_transformedxy(mole['x'], mole['y'])
                draw_crosshair(image, x, y)

        return image


class BoundingAreaOverlay():
    """An overlay to show the bounding area, if any."""

    def __init__(self):
        self.bounding_box = None

    def __call__(self, image, transform):

        image //= 2
        if self.bounding_box is not None:
            color = (0, 0, 255)
            size = 2
            space = mel.lib.ellipsespace.Transform(self.bounding_box)

            def toimage(point):
                point = space.from_space((point))
                point = transform.imagexy_to_transformedxy(*point)
                return point

            border = [
                toimage((-1, -1)),
                toimage((1, -1)),
                toimage((1, 1)),
                toimage((-1, 1)),
                toimage((-1, -1)),
            ]
            border = numpy.array(border)
            centre = [
                toimage((0, 0.1)),
                toimage((0, -0.1)),
                toimage((0.05, 0)),
                toimage((0.1, 0)),
                toimage((-0.1, 0)),
                toimage((0, 0)),
                toimage((0, 0.1)),
            ]
            centre = numpy.array(centre)

            cv2.drawContours(image, [border, centre], -1, color, size)

        return image


class ZoomedImageTransform():

    def __init__(self, image, pos, rect):
        self._pos = pos
        self._rect = rect
        self._offset = mel.lib.image.calc_centering_offset(pos, rect)

        self._image = image

    def render(self):

        return mel.lib.image.centered_at(
            self._image,
            self._pos,
            self._rect)

    def imagexy_to_transformedxy(self, x, y):
        return numpy.array((x, y)) + self._offset

    def transformedxy_to_imagexy(self, x, y):
        return numpy.array((x, y)) - self._offset


class FittedImageTransform():

    def __init__(self, image, fit_rect):
        self._fit_rect = fit_rect
        image_rect = mel.lib.image.get_image_rect(image)

        letterbox = mel.lib.image.calc_letterbox(
            *image_rect, *self._fit_rect)

        self._offset = numpy.array(letterbox[:2])
        self._scale = image.shape[1] / letterbox[2]

        self._image = image

    def render(self):
        return mel.lib.image.letterbox(
            self._image, *self._fit_rect)

    def imagexy_to_transformedxy(self, x, y):
        return (numpy.array((x, y)) / self._scale + self._offset).astype(int)

    def transformedxy_to_imagexy(self, x, y):
        return ((numpy.array((x, y)) - self._offset) * self._scale).astype(int)


class EditorMode(enum.Enum):
    edit_mole = 1
    edit_mask = 2
    mole_mark = 3
    image_relate = 4
    bounding_area = 5
    debug_automole = 0
    debug_autorelate = 9


class Editor:

    def __init__(self, directory_list, width, height):
        self._uuid_to_tricolour = mel.rotomap.tricolour.UuidTriColourPicker()
        self.display = Display(width, height)
        self.moledata_list = [MoleData(x.image_paths) for x in directory_list]

        self._mode = EditorMode.edit_mole

        self.moledata_index = 0
        self.moledata = self.moledata_list[self.moledata_index]
        self._follow = None
        self._mole_overlay = MoleMarkerOverlay(self._uuid_to_tricolour)
        self.marked_mole_overlay = MarkedMoleOverlay()
        self.image_relate_overlay = ImageRelateOverlay()
        self.bounding_area_overlay = BoundingAreaOverlay()
        self._status_overlay = StatusOverlay()
        self.show_current()

        self.masker_radius = DEFAULT_MASKER_RADIUS

        self.from_moles = None

    def set_smaller_masker(self):
        self.masker_radius //= 2

    def set_larger_masker(self):
        self.masker_radius *= 2

    def set_default_masker(self):
        self.masker_radius = DEFAULT_MASKER_RADIUS

    def set_automoledebug_mode(self):
        self._mode = EditorMode.debug_automole
        self.show_current()

    def set_autorelatedebug_mode(self):
        self._mode = EditorMode.debug_autorelate
        self.show_current()

    def set_editmole_mode(self):
        self._mode = EditorMode.edit_mole
        self.show_current()

    def set_editmask_mode(self):
        self._mode = EditorMode.edit_mask
        self.show_current()

    def set_molemark_mode(self):
        self._mode = EditorMode.mole_mark
        self.show_current()

    def set_imagerelate_mode(self):
        self._mode = EditorMode.image_relate
        self.show_current()

    def set_boundingarea_mode(self):
        self._mode = EditorMode.bounding_area
        self.show_current()

    def set_status(self, text):
        self._status_overlay.text = text

    def set_moles(self, moles):
        self.moledata.moles = moles
        self.show_current()

    def follow(self, uuid_to_follow):
        self._follow = uuid_to_follow
        self._mole_overlay.set_highlight_uuid(self._follow)

        follow_mole = None
        for m in self.moledata.moles:
            if m['uuid'] == self._follow:
                follow_mole = m
                break

        if follow_mole is not None:
            self.show_zoomed_display(follow_mole['x'], follow_mole['y'])

    def skip_to_mole(self, uuid_to_skip_to):
        original_index = self.moledata.index()
        done = False
        while not done:
            for m in self.moledata.moles:
                if m['uuid'] == uuid_to_skip_to:
                    return
            self.moledata.increment()
            self.moledata.get_image()
            if self.moledata.index() == original_index:
                return

    def toggle_markers(self):
        self._mole_overlay.toggle_markers()
        self.show_current()

    def toggle_faded_markers(self):
        self._mole_overlay.toggle_faded_markers()
        self.show_current()

    def set_from_moles(self, moles):
        self.from_moles = moles

    def set_mask(self, mouse_x, mouse_y, enable):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        value = 255 if enable else 0
        radius = self.masker_radius
        cv2.circle(self.moledata.mask, (image_x, image_y), radius, value, -1)
        self.moledata.save_mask()
        self.show_current()

    def show_current(self):
        self.display.set_title(self.moledata.current_image_path())
        image = self.moledata.get_image()

        if self._mode is EditorMode.debug_automole:
            image = image[:]
            image = mel.rotomap.detectmoles.draw_debug(
                image, self.moledata.mask)
            self.display.show_current(image, None)
        elif self._mode is EditorMode.debug_autorelate:
            image = numpy.copy(image)
            image = mel.rotomap.relate.draw_debug(
                image, self.moledata.moles, self.from_moles)
            self.display.show_current(image, None)
        elif self._mode is EditorMode.edit_mask:
            mask = self.moledata.mask
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            gray_image[:, :, 2] = mask
            self.display.show_current(gray_image, None)
        elif self._mode is EditorMode.mole_mark:
            self.marked_mole_overlay.moles = self.moledata.moles
            self.display.show_current(
                image,
                self.marked_mole_overlay)
        elif self._mode is EditorMode.image_relate:
            self.image_relate_overlay.moles = self.moledata.moles
            self.image_relate_overlay.prev_moles = self.from_moles
            self.display.show_current(
                image,
                self.image_relate_overlay)
        elif self._mode is EditorMode.bounding_area:
            box = self.moledata.metadata.get('ellipse', None)
            self.bounding_area_overlay.bounding_box = box
            self.display.show_current(
                image, self.bounding_area_overlay)
        else:
            self._mole_overlay.moles = self.moledata.moles
            self.display.show_current(
                image,
                make_composite_overlay(
                    self._mole_overlay,
                    self._status_overlay))

    def show_fitted(self):
        self.display.set_fitted()
        self.show_current()

    def show_zoomed(self, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        self.display.set_zoomed(image_x, image_y)
        self.show_current()

    def show_zoomed_display(self, image_x, image_y):
        self.display.set_zoomed(image_x, image_y)
        self.show_current()

    def show_prev_map(self):
        def transition():
            self.moledata_index -= 1
            self.moledata_index %= len(self.moledata_list)
            self.moledata = self.moledata_list[self.moledata_index]
        self._adjusted_transition(transition)
        self.show_current()

    def show_next_map(self):
        def transition():
            self.moledata_index += 1
            self.moledata_index %= len(self.moledata_list)
            self.moledata = self.moledata_list[self.moledata_index]
        self._adjusted_transition(transition)
        self.show_current()

    def show_prev(self):
        self._adjusted_transition(self.moledata.decrement)
        self.show_current()

    def show_next(self):
        self._adjusted_transition(self.moledata.increment)
        self.show_current()

    def _adjusted_transition(self, transition_func):
        if self.display.is_zoomed() and 'ellipse' in self.moledata.metadata:
            pos = self.display.get_zoom_pos()
            ellipse = self.moledata.metadata['ellipse']
            pos = mel.lib.ellipsespace.Transform(ellipse).to_space(pos)

            transition_func()
            self.moledata.ensure_loaded()

            if 'ellipse' in self.moledata.metadata:
                ellipse = self.moledata.metadata['ellipse']
                pos = mel.lib.ellipsespace.Transform(ellipse).from_space(pos)
                self.display.set_zoomed(pos[0], pos[1])
        else:
            transition_func()

    def show_next_n(self, number_to_advance):
        for i in range(number_to_advance):
            self.moledata.increment()
            self.moledata.get_image()
        self.show_current()

    def add_mole(self, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        mel.rotomap.moles.add_mole(self.moledata.moles, image_x, image_y)
        self.moledata.save_moles()
        self.show_current()

    def add_mole_display(self, image_x, image_y, mole_uuid=None):
        mel.rotomap.moles.add_mole(
            self.moledata.moles, image_x, image_y, mole_uuid)
        self.moledata.save_moles()
        self.show_current()

    def confirm_mole(self, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        mole_uuid = mel.rotomap.moles.get_nearest_mole_uuid(
            self.moledata.moles, image_x, image_y)
        mel.rotomap.moles.set_nearest_mole_uuid(
            self.moledata.moles,
            image_x,
            image_y,
            mole_uuid,
            is_canonical=True)
        self.moledata.save_moles()
        self.show_current()

    def set_mole_uuid(self, mouse_x, mouse_y, mole_uuid, is_canonical=True):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        mel.rotomap.moles.set_nearest_mole_uuid(
            self.moledata.moles, image_x, image_y, mole_uuid, is_canonical)
        self.moledata.save_moles()
        self.show_current()

    def get_mole_uuid(self, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        return mel.rotomap.moles.get_nearest_mole_uuid(
            self.moledata.moles, image_x, image_y)

    def move_nearest_mole(self, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        mel.rotomap.moles.move_nearest_mole(
            self.moledata.moles, image_x, image_y)
        self.moledata.save_moles()
        self.show_current()

    def remove_mole(self, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        mel.rotomap.moles.remove_nearest_mole(
            self.moledata.moles, image_x, image_y)
        self.moledata.save_moles()
        self.show_current()

    def crud_mole(self, mole_uuid, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)

        i = mel.rotomap.moles.uuid_mole_index(self.moledata.moles, mole_uuid)
        if i is not None:
            self.moledata.moles[i]['x'] = image_x
            self.moledata.moles[i]['y'] = image_y
        else:
            mel.rotomap.moles.add_mole(
                self.moledata.moles, image_x, image_y, mole_uuid)

        self.moledata.save_moles()
        self.show_current()

    def remap_uuid(self, from_uuid, to_uuid):
        print(f'Remap globally {from_uuid} to {to_uuid}.')
        self.moledata.remap_uuid(from_uuid, to_uuid)
        self.show_current()


class MoleData:

    def __init__(self, path_list):

        # Make an instance-specific cache of images. Note that this means that
        # mel will need to be re-run in order to pick up changes to mole
        # images. This seems to be fine for use-cases to date, only the mole
        # data seems to change from underneath really.
        @functools.lru_cache()
        def load_image(image_path):
            return mel.lib.image.load_image(image_path)
        self._load_image = load_image

        self.moles = []
        self.metadata = {}
        self.image = None
        self.mask = None
        self._mask_path = None
        self._path_list = path_list
        self._list_index = 0
        self._num_images = len(self._path_list)
        self._loaded_index = None
        self.ensure_loaded()

    def get_image(self):
        self.ensure_loaded()
        return self.image

    def ensure_loaded(self):

        if self._loaded_index == self._list_index:
            return

        image_path = self._path_list[self._list_index]
        self.image = self._load_image(image_path)

        self.moles = mel.rotomap.moles.load_image_moles(image_path)
        self.metadata = mel.rotomap.moles.load_image_metadata(image_path)

        height, width = self.image.shape[:2]
        self._mask_path = mel.rotomap.mask.path(image_path)
        self.mask = mel.rotomap.mask.load_or_none(image_path)
        if self.mask is None:
            self.mask = numpy.zeros((height, width), numpy.uint8)

        self._loaded_index = self._list_index

    def remap_uuid(self, from_uuid, to_uuid):
        for image_path in self._path_list:
            moles = mel.rotomap.moles.load_image_moles(image_path)
            for m in moles:
                if m['uuid'] == from_uuid:
                    m['uuid'] = to_uuid
                    m[mel.rotomap.moles.KEY_IS_CONFIRMED] = True
            mel.rotomap.moles.save_image_moles(moles, image_path)

        image_path = self._path_list[self._list_index]
        self.moles = mel.rotomap.moles.load_image_moles(image_path)

    def decrement(self):
        new_index = self._list_index + self._num_images - 1
        self._list_index = new_index % self._num_images

    def increment(self):
        self._list_index = (self._list_index + 1) % self._num_images

    def index(self):
        return self._list_index

    def save_mask(self):
        mel.lib.common.write_image(self._mask_path, self.mask)

    def save_moles(self):
        image_path = self._path_list[self._list_index]
        mel.rotomap.moles.normalise_moles(self.moles)
        mel.rotomap.moles.save_image_moles(self.moles, image_path)

    def current_image_path(self):
        return self._path_list[self._list_index]
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
