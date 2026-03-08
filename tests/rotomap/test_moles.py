"""Test suite for mel.rotomap.moles."""

# Copyright 2016-2026 Angelos Evripiotis.
# Generated with assistance from Claude Code.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import math
import pathlib

import numpy
import pytest

import mel.rotomap.moles as moles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mole(x, y, uuid_=None, is_confirmed=True):
    """Create a mole dict with required fields."""
    return {
        "x": x,
        "y": y,
        "uuid": uuid_ or moles.make_new_uuid(),
        moles.KEY_IS_CONFIRMED: is_confirmed,
    }


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, separators=(",", ": "), sort_keys=True)
        print(file=f)


def _create_jpeg_stub(path):
    """Create a minimal file that qualifies as a JPEG by name."""
    pathlib.Path(path).write_bytes(b"\xff\xd8\xff")


# ---------------------------------------------------------------------------
# save_json / load_json round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_round_trip_list(self, tmp_path):
        data = [{"a": 1, "b": 2}, {"c": 3}]
        p = tmp_path / "data.json"
        moles.save_json(p, data)
        assert moles.load_json(p) == data

    def test_round_trip_dict(self, tmp_path):
        data = {"key": "value", "nested": {"x": 10}}
        p = tmp_path / "data.json"
        moles.save_json(p, data)
        assert moles.load_json(p) == data

    def test_round_trip_empty_list(self, tmp_path):
        p = tmp_path / "empty.json"
        moles.save_json(p, [])
        assert moles.load_json(p) == []

    def test_save_json_trailing_newline(self, tmp_path):
        p = tmp_path / "nl.json"
        moles.save_json(p, {"a": 1})
        text = p.read_text()
        assert text.endswith("\n")

    def test_save_json_sorted_keys(self, tmp_path):
        p = tmp_path / "sorted.json"
        moles.save_json(p, {"z": 1, "a": 2})
        text = p.read_text()
        assert text.index('"a"') < text.index('"z"')


# ---------------------------------------------------------------------------
# save_image_moles / load_image_moles round-trip
# ---------------------------------------------------------------------------


class TestImageMolesRoundTrip:
    def test_round_trip(self, tmp_path):
        img = tmp_path / "photo.jpg"
        _create_jpeg_stub(img)

        m = [_make_mole(10, 20, "aaa"), _make_mole(30, 40, "bbb")]
        moles.save_image_moles(m, img)
        loaded = moles.load_image_moles(img)
        assert len(loaded) == 2
        assert loaded[0]["uuid"] == "aaa"
        assert loaded[1]["x"] == 30

    def test_round_trip_extra_stem(self, tmp_path):
        img = tmp_path / "photo.jpg"
        _create_jpeg_stub(img)

        m = [_make_mole(5, 6, "ccc")]
        moles.save_image_moles(m, img, extra_stem="automark")
        loaded = moles.load_image_moles(img, extra_stem="automark")
        assert loaded[0]["uuid"] == "ccc"

        # Default stem should return empty (no file)
        assert moles.load_image_moles(img) == []

    def test_load_missing_image_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            moles.load_image_moles(tmp_path / "nope.jpg")

    def test_load_no_mole_file_returns_empty(self, tmp_path):
        img = tmp_path / "empty.jpg"
        _create_jpeg_stub(img)
        assert moles.load_image_moles(img) == []

    def test_load_coerces_xy_to_int(self, tmp_path):
        img = tmp_path / "coerce.jpg"
        _create_jpeg_stub(img)
        data = [
            {
                "x": 1.7,
                "y": 2.3,
                "uuid": "abc",
                moles.KEY_IS_CONFIRMED: True,
            }
        ]
        _write_json(pathlib.Path(str(img) + ".json"), data)
        loaded = moles.load_image_moles(img)
        assert loaded[0]["x"] == 1
        assert loaded[0]["y"] == 2
        assert isinstance(loaded[0]["x"], int)

    def test_load_raises_when_missing_confirmed_key(self, tmp_path):
        img = tmp_path / "bad.jpg"
        _create_jpeg_stub(img)
        data = [{"x": 0, "y": 0, "uuid": "u1"}]
        _write_json(pathlib.Path(str(img) + ".json"), data)
        with pytest.raises(Exception, match=moles.KEY_IS_CONFIRMED):
            moles.load_image_moles(img)

    def test_load_raises_when_uuid_is_none(self, tmp_path):
        img = tmp_path / "null_uuid.jpg"
        _create_jpeg_stub(img)
        data = [
            {
                "x": 0,
                "y": 0,
                "uuid": None,
                moles.KEY_IS_CONFIRMED: True,
            }
        ]
        _write_json(pathlib.Path(str(img) + ".json"), data)
        with pytest.raises(Exception, match="UUID cannot be None"):
            moles.load_image_moles(img)


# ---------------------------------------------------------------------------
# save_image_metadata / load_image_metadata
# ---------------------------------------------------------------------------


class TestImageMetadata:
    def test_round_trip(self, tmp_path):
        img = tmp_path / "meta.jpg"
        _create_jpeg_stub(img)
        meta = {"ellipse": ((100, 100), (200, 300), 0)}
        moles.save_image_metadata(meta, str(img))
        loaded = moles.load_image_metadata(img)
        assert loaded["ellipse"] == [[100, 100], [200, 300], 0]

    def test_load_no_metadata_returns_empty_dict(self, tmp_path):
        img = tmp_path / "no_meta.jpg"
        _create_jpeg_stub(img)
        assert moles.load_image_metadata(img) == {}

    def test_load_invalid_ellipse_raises(self, tmp_path):
        img = tmp_path / "bad_meta.jpg"
        _create_jpeg_stub(img)
        meta = {"ellipse": [[0, 0], [0, 0], 0]}
        _write_json(pathlib.Path(str(img) + ".meta.json"), meta)
        with pytest.raises(ValueError, match="Bad data"):
            moles.load_image_metadata(img)


# ---------------------------------------------------------------------------
# make_new_uuid
# ---------------------------------------------------------------------------


class TestMakeNewUuid:
    def test_returns_hex_string(self):
        u = moles.make_new_uuid()
        assert isinstance(u, str)
        int(u, 16)  # must be valid hex

    def test_unique(self):
        uuids = {moles.make_new_uuid() for _ in range(100)}
        assert len(uuids) == 100


# ---------------------------------------------------------------------------
# add_mole
# ---------------------------------------------------------------------------


class TestAddMole:
    def test_add_mole_auto_uuid(self):
        m = []
        moles.add_mole(m, 10, 20)
        assert len(m) == 1
        assert m[0]["x"] == 10
        assert m[0]["y"] == 20
        assert m[0][moles.KEY_IS_CONFIRMED] is False

    def test_add_mole_explicit_uuid(self):
        m = []
        moles.add_mole(m, 5, 6, mole_uuid="my-uuid")
        assert m[0]["uuid"] == "my-uuid"
        assert m[0][moles.KEY_IS_CONFIRMED] is True


# ---------------------------------------------------------------------------
# nearest_mole_index / nearest_mole_index_distance
# ---------------------------------------------------------------------------


class TestNearestMole:
    def test_single_mole(self):
        m = [_make_mole(10, 10, "a")]
        assert moles.nearest_mole_index(m, 0, 0) == 0

    def test_selects_closest(self):
        m = [
            _make_mole(0, 0, "a"),
            _make_mole(100, 100, "b"),
        ]
        assert moles.nearest_mole_index(m, 99, 99) == 1

    def test_empty_list(self):
        idx, dist = moles.nearest_mole_index_distance([], 0, 0)
        assert idx is None
        assert dist is None

    def test_distance_value(self):
        m = [_make_mole(3, 4, "a")]
        _, dist = moles.nearest_mole_index_distance(m, 0, 0)
        assert dist == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# uuid_mole_index
# ---------------------------------------------------------------------------


class TestUuidMoleIndex:
    def test_found(self):
        m = [_make_mole(0, 0, "a"), _make_mole(1, 1, "b")]
        assert moles.uuid_mole_index(m, "b") == 1

    def test_not_found(self):
        m = [_make_mole(0, 0, "a")]
        assert moles.uuid_mole_index(m, "zzz") is None

    def test_empty_list(self):
        assert moles.uuid_mole_index([], "x") is None


# ---------------------------------------------------------------------------
# set_nearest_mole_uuid / get_nearest_mole_uuid
# ---------------------------------------------------------------------------


class TestSetGetNearestMoleUuid:
    def test_set_uuid(self):
        m = [_make_mole(10, 10, "old")]
        moles.set_nearest_mole_uuid(m, 10, 10, "new")
        assert m[0]["uuid"] == "new"
        assert m[0][moles.KEY_IS_CONFIRMED] is True

    def test_set_uuid_not_canonical(self):
        m = [_make_mole(10, 10, "old")]
        moles.set_nearest_mole_uuid(m, 10, 10, "new", is_canonical=False)
        assert m[0][moles.KEY_IS_CONFIRMED] is False

    def test_get_uuid(self):
        m = [_make_mole(5, 5, "u1"), _make_mole(50, 50, "u2")]
        assert moles.get_nearest_mole_uuid(m, 6, 6) == "u1"

    def test_get_uuid_empty(self):
        assert moles.get_nearest_mole_uuid([], 0, 0) is None

    def test_set_empty_noop(self):
        m = []
        moles.set_nearest_mole_uuid(m, 0, 0, "u")
        assert m == []


# ---------------------------------------------------------------------------
# move_nearest_mole / remove_nearest_mole
# ---------------------------------------------------------------------------


class TestMoveRemoveMole:
    def test_move(self):
        m = [_make_mole(0, 0, "a")]
        moles.move_nearest_mole(m, 99, 88)
        assert m[0]["x"] == 99
        assert m[0]["y"] == 88

    def test_move_empty_noop(self):
        m = []
        moles.move_nearest_mole(m, 1, 1)
        assert m == []

    def test_remove(self):
        m = [_make_mole(0, 0, "a"), _make_mole(100, 100, "b")]
        moles.remove_nearest_mole(m, 1, 1)
        assert len(m) == 1
        assert m[0]["uuid"] == "b"

    def test_remove_empty_noop(self):
        m = []
        moles.remove_nearest_mole(m, 0, 0)
        assert m == []


# ---------------------------------------------------------------------------
# normalise_moles
# ---------------------------------------------------------------------------


class TestNormaliseMoles:
    def test_converts_to_int(self):
        m = [{"x": 1.9, "y": 2.1}]
        moles.normalise_moles(m)
        assert m[0]["x"] == 1
        assert m[0]["y"] == 2
        assert isinstance(m[0]["x"], int)

    def test_empty_list(self):
        m = []
        moles.normalise_moles(m)
        assert m == []


# ---------------------------------------------------------------------------
# MoleData
# ---------------------------------------------------------------------------


class TestMoleData:
    def test_basic_properties(self):
        m = [_make_mole(10, 20, "u1"), _make_mole(30, 40, "u2")]
        md = moles.MoleData(m)
        assert md.uuids == frozenset({"u1", "u2"})
        assert len(md.moles) == 2

    def test_uuid_points(self):
        m = [_make_mole(5, 10, "u1")]
        md = moles.MoleData(m)
        assert "u1" in md.uuid_points
        numpy.testing.assert_array_equal(md.uuid_points["u1"], [5, 10])

    def test_empty(self):
        md = moles.MoleData([])
        assert md.uuids == frozenset()
        assert md.moles == ()
        assert md.uuid_points == {}


# ---------------------------------------------------------------------------
# MoleListDiff
# ---------------------------------------------------------------------------


class TestMoleListDiff:
    def test_new_and_missing(self):
        old = {"a", "b", "c"}
        new = {"b", "c", "d"}
        diff = moles.MoleListDiff(old, new, set(), set())
        assert diff.new == {"d"}
        assert diff.missing == {"a"}

    def test_ignore_new(self):
        diff = moles.MoleListDiff({"a"}, {"a", "b"}, ignore_new={"b"}, ignore_missing=set())
        assert diff.new == set()

    def test_ignore_missing(self):
        diff = moles.MoleListDiff(
            {"a", "b"}, {"a"}, ignore_new=set(), ignore_missing={"b"}
        )
        assert diff.missing == set()

    def test_all_same(self):
        diff = moles.MoleListDiff({"a"}, {"a"}, set(), set())
        assert diff.new == set()
        assert diff.missing == set()

    def test_empty_sets(self):
        diff = moles.MoleListDiff(set(), set(), set(), set())
        assert diff.new == set()
        assert diff.missing == set()


# ---------------------------------------------------------------------------
# mole_to_point / mole_list_to_pointvec / to_uuid_points
# ---------------------------------------------------------------------------


class TestPointConversions:
    def test_mole_to_point(self):
        m = _make_mole(7, 13, "a")
        pt = moles.mole_to_point(m)
        numpy.testing.assert_array_equal(pt, [7, 13])
        assert numpy.issubdtype(pt.dtype, numpy.integer)

    def test_mole_list_to_pointvec(self):
        m = [_make_mole(1, 2, "a"), _make_mole(3, 4, "b")]
        pv = moles.mole_list_to_pointvec(m)
        assert pv.shape == (2, 2)
        numpy.testing.assert_array_equal(pv[1], [3, 4])

    def test_to_uuid_points(self):
        m = [_make_mole(10, 20, "u1")]
        up = moles.to_uuid_points(m)
        assert "u1" in up
        numpy.testing.assert_array_equal(up["u1"], [10, 20])

    def test_mole_list_to_pointvec_empty(self):
        pv = moles.mole_list_to_pointvec([])
        assert pv.shape == (0,)


# ---------------------------------------------------------------------------
# normalised_ellipse_mask
# ---------------------------------------------------------------------------


class TestNormalisedEllipseMask:
    def test_90_degree_rotation(self):
        assert moles.normalised_ellipse_mask(((1, 2), (100, 200), 90)) == (
            (1, 2),
            (200, 100),
            0,
        )

    def test_no_rotation(self):
        assert moles.normalised_ellipse_mask(((0, 0), (50, 50), 0)) == (
            (0, 0),
            (50, 50),
            0,
        )

    def test_negative_rotation(self):
        result = moles.normalised_ellipse_mask(((0, 0), (100, 200), -90))
        assert result[2] == 0

    def test_small_rotation_unchanged(self):
        result = moles.normalised_ellipse_mask(((5, 5), (10, 20), 30))
        assert result[2] == 30


# ---------------------------------------------------------------------------
# validate_ellipse_mask
# ---------------------------------------------------------------------------


class TestValidateEllipseMask:
    def test_valid(self):
        moles.validate_ellipse_mask(((100, 100), (200, 200), 0))

    def test_too_small(self):
        with pytest.raises(ValueError, match="too small"):
            moles.validate_ellipse_mask(((10, 10), (0, 100), 0))

    def test_too_big(self):
        with pytest.raises(ValueError, match="too big"):
            moles.validate_ellipse_mask(((10, 10), (999999, 100), 0))

    def test_out_of_bounds_negative(self):
        with pytest.raises(ValueError, match="out of bounds"):
            moles.validate_ellipse_mask(((-1, 10), (100, 100), 0))

    def test_out_of_bounds_exceeds_max(self):
        with pytest.raises(ValueError, match="out of bounds"):
            moles.validate_ellipse_mask(((10001, 10), (100, 100), 0))


# ---------------------------------------------------------------------------
# load_potential_set_file
# ---------------------------------------------------------------------------


class TestLoadPotentialSetFile:
    def test_loads_lines(self, tmp_path):
        f = tmp_path / "ignore"
        f.write_text("aaa\nbbb\n")
        result = moles.load_potential_set_file(tmp_path, "ignore")
        assert result == {"aaa", "bbb"}

    def test_ignores_comments_and_blanks(self, tmp_path):
        f = tmp_path / "ignore"
        f.write_text("# comment\n\n  \nvalid\n")
        result = moles.load_potential_set_file(tmp_path, "ignore")
        assert result == {"valid"}

    def test_missing_file_returns_empty(self, tmp_path):
        result = moles.load_potential_set_file(tmp_path, "nonexistent")
        assert result == set()


# ---------------------------------------------------------------------------
# RotomapDirectory
# ---------------------------------------------------------------------------


class TestRotomapDirectory:
    def _make_rotomap(self, tmp_path, image_names=("a.jpg",)):
        for name in image_names:
            img = tmp_path / name
            _create_jpeg_stub(img)
            _write_json(
                pathlib.Path(str(img) + ".json"),
                [_make_mole(0, 0, name)],
            )
        return tmp_path

    def test_construction(self, tmp_path):
        self._make_rotomap(tmp_path)
        rd = moles.RotomapDirectory(tmp_path)
        assert len(rd.image_paths) == 1

    def test_sorted_image_paths(self, tmp_path):
        self._make_rotomap(tmp_path, ("c.jpg", "a.jpg", "b.jpg"))
        rd = moles.RotomapDirectory(tmp_path)
        basenames = [pathlib.Path(p).name for p in rd.image_paths]
        assert basenames == ["a.jpg", "b.jpg", "c.jpg"]

    def test_yield_mole_lists(self, tmp_path):
        self._make_rotomap(tmp_path, ("x.jpg",))
        rd = moles.RotomapDirectory(tmp_path)
        pairs = list(rd.yield_mole_lists())
        assert len(pairs) == 1
        assert len(pairs[0][1]) == 1

    def test_yield_frames(self, tmp_path):
        self._make_rotomap(tmp_path)
        rd = moles.RotomapDirectory(tmp_path)
        frames = list(rd.yield_frames())
        assert len(frames) == 1
        assert isinstance(frames[0], moles.RotomapFrame)

    def test_calc_uuids(self, tmp_path):
        self._make_rotomap(tmp_path, ("a.jpg", "b.jpg"))
        rd = moles.RotomapDirectory(tmp_path)
        uuids = rd.calc_uuids()
        assert "a.jpg" in uuids
        assert "b.jpg" in uuids

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(ValueError, match="not a directory"):
            moles.RotomapDirectory(f)

    def test_no_images(self, tmp_path):
        with pytest.raises(ValueError, match="no images"):
            moles.RotomapDirectory(tmp_path)

    def test_repr(self, tmp_path):
        self._make_rotomap(tmp_path)
        rd = moles.RotomapDirectory(tmp_path)
        assert "RotomapDirectory" in repr(rd)

    def test_lesions_loaded_empty(self, tmp_path):
        self._make_rotomap(tmp_path)
        rd = moles.RotomapDirectory(tmp_path)
        assert rd.lesions == []


# ---------------------------------------------------------------------------
# RotomapFrame
# ---------------------------------------------------------------------------


class TestRotomapFrame:
    def test_construction(self, tmp_path):
        img = tmp_path / "frame.jpg"
        _create_jpeg_stub(img)
        rf = moles.RotomapFrame(img)
        assert rf.moles == []
        assert isinstance(rf.moledata, moles.MoleData)

    def test_has_mole_file_false(self, tmp_path):
        img = tmp_path / "frame.jpg"
        _create_jpeg_stub(img)
        rf = moles.RotomapFrame(img)
        assert rf.has_mole_file() is False

    def test_has_mole_file_true(self, tmp_path):
        img = tmp_path / "frame.jpg"
        _create_jpeg_stub(img)
        _write_json(
            pathlib.Path(str(img) + ".json"),
            [_make_mole(0, 0, "u1")],
        )
        rf = moles.RotomapFrame(img)
        assert rf.has_mole_file() is True

    def test_has_mole_file_extra_stem(self, tmp_path):
        img = tmp_path / "frame.jpg"
        _create_jpeg_stub(img)
        _write_json(
            pathlib.Path(str(img) + ".extra.json"),
            [_make_mole(0, 0, "u1")],
        )
        rf = moles.RotomapFrame(img, extra_stem="extra")
        assert rf.has_mole_file() is True

    def test_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not directory"):
            moles.RotomapFrame(tmp_path)

    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            moles.RotomapFrame(tmp_path / "nope.jpg")

    def test_non_jpeg_raises(self, tmp_path):
        f = tmp_path / "file.png"
        f.write_bytes(b"x")
        with pytest.raises(ValueError, match="Unrecognised suffix"):
            moles.RotomapFrame(f)

    def test_repr(self, tmp_path):
        img = tmp_path / "r.jpg"
        _create_jpeg_stub(img)
        rf = moles.RotomapFrame(img)
        assert "RotomapFrame" in repr(rf)


# ---------------------------------------------------------------------------
# load_rotomap_dir_lesions_file / save_rotomap_dir_lesions_file
# ---------------------------------------------------------------------------


class TestLesionsFile:
    def test_round_trip(self, tmp_path):
        lesions = [
            {"uuid": "l1", moles.KEY_IS_UNCHANGED: True},
            {"uuid": "l2", moles.KEY_IS_UNCHANGED: False},
        ]
        moles.save_rotomap_dir_lesions_file(tmp_path, lesions)
        loaded = moles.load_rotomap_dir_lesions_file(tmp_path)
        assert len(loaded) == 2
        assert loaded[0]["uuid"] == "l1"

    def test_empty_lesions(self, tmp_path):
        assert moles.load_rotomap_dir_lesions_file(tmp_path) == []

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            moles.load_rotomap_dir_lesions_file(tmp_path / "nope")

    def test_missing_is_unchanged_raises(self, tmp_path):
        _write_json(
            tmp_path / moles.ROTOMAP_DIR_LESIONS_FILENAME,
            [{"uuid": "l1"}],
        )
        with pytest.raises(Exception, match=moles.KEY_IS_UNCHANGED):
            moles.load_rotomap_dir_lesions_file(tmp_path)

    def test_none_uuid_raises(self, tmp_path):
        _write_json(
            tmp_path / moles.ROTOMAP_DIR_LESIONS_FILENAME,
            [{"uuid": None, moles.KEY_IS_UNCHANGED: True}],
        )
        with pytest.raises(Exception, match="UUID cannot be None"):
            moles.load_rotomap_dir_lesions_file(tmp_path)


# ---------------------------------------------------------------------------
# make_argparse_rotomap_directory
# ---------------------------------------------------------------------------


class TestMakeArgparseRotomapDirectory:
    def test_valid(self, tmp_path):
        img = tmp_path / "img.jpg"
        _create_jpeg_stub(img)
        rd = moles.make_argparse_rotomap_directory(str(tmp_path))
        assert isinstance(rd, moles.RotomapDirectory)

    def test_invalid_raises_argtype(self, tmp_path):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            moles.make_argparse_rotomap_directory(str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# Copyright (C) 2016-2026 Angelos Evripiotis.
# Generated with assistance from Claude Code.
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
