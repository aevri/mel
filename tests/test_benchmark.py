#! /usr/bin/env python3
"""Benchmark test for mel rotomap guess-missing and guess-refine commands."""

import contextlib
import json
import math
import os
import pathlib
import subprocess
import tarfile
import tempfile
import urllib.request


def test_benchmark_guess_moles():
    """Test mole guessing performance using benchmark dataset."""
    with chtempdir_context():
        # Download and extract benchmark dataset
        dataset_url = (
            "https://github.com/aevri/mel-datasets/archive/refs/tags/v0.1.0.tar.gz"
        )
        dataset_path = download_and_extract_dataset(dataset_url)

        # Set up paths to test images
        m1_path = dataset_path / "mel-datasets-0.1.0" / "m1"
        source_image = (
            m1_path / "rotomaps" / "parts" / "Trunk" / "Back" / "2025_06_12" / "0.jpg"
        )
        target_image = (
            m1_path / "rotomaps" / "parts" / "Trunk" / "Back" / "2025_06_13" / "0.jpg"
        )
        target_json = target_image.with_suffix(".jpg.json")

        # Verify files exist
        assert source_image.exists(), f"Source image not found: {source_image}"
        assert target_image.exists(), f"Target image not found: {target_image}"
        assert target_json.exists(), f"Target JSON not found: {target_json}"

        # Read original moles and remove first 3 for benchmarking
        original_moles = read_moles(target_json)
        assert len(original_moles) >= 3, (
            f"Need at least 3 moles for benchmarking, found {len(original_moles)}"
        )

        removed_moles = original_moles[:3]
        remaining_moles = original_moles[3:]

        # Save modified JSON with removed moles
        save_moles(target_json, remaining_moles)

        try:
            # Run mel rotomap guess-missing
            expect_ok(
                "mel", "rotomap", "guess-missing", str(source_image), str(target_image)
            )

            # Run mel rotomap guess-refine
            expect_ok(
                "mel", "rotomap", "guess-refine", str(source_image), str(target_image)
            )

            # Read results and measure performance
            result_moles = read_moles(target_json)
            performance_metrics = calculate_performance_metrics(
                removed_moles, result_moles
            )

            # Print detailed per-mole results
            print(f"Per-mole results:")
            for mole_result in performance_metrics["mole_results"]:
                status = mole_result["status"]
                uuid_short = mole_result["uuid"][:8]
                if status == "matched":
                    print(
                        f"  ✓ Mole {uuid_short}: MATCHED at distance {mole_result['distance']:.1f} pixels"
                    )
                elif status == "found_far":
                    print(
                        f"  ✗ Mole {uuid_short}: FOUND but distance {mole_result['distance']:.1f} pixels > 50 pixel threshold"
                    )
                else:  # not_found
                    print(f"  ✗ Mole {uuid_short}: NOT FOUND")

            # Print performance summary
            print(f"\nBenchmark Results:")
            print(f"  Original moles removed: {len(removed_moles)}")
            print(
                f"  Moles found by guess commands: {performance_metrics['moles_found']}"
            )
            print(f"  Match rate: {performance_metrics['match_rate']:.2%}")
            if performance_metrics["avg_distance"] != float("inf"):
                print(
                    f"  Average distance to canonical: {performance_metrics['avg_distance']:.2f} pixels"
                )
                print(
                    f"  Max distance to canonical: {performance_metrics['max_distance']:.2f} pixels"
                )

            # Print actual results for easy copy-paste updating
            actual_results = {
                "moles_found": performance_metrics["moles_found"],
                "matched_count": performance_metrics["matched_count"],
                "avg_distance": round(performance_metrics["avg_distance"], 2)
                if performance_metrics["avg_distance"] != float("inf")
                else None,
                "max_distance": round(performance_metrics["max_distance"], 2)
                if performance_metrics["max_distance"] != float("inf")
                else None,
            }
            print(f"\nActual results (copy to update expected_performance_baseline):")
            print(f"    'moles_found': {actual_results['moles_found']},")
            print(f"    'matched_count': {actual_results['matched_count']},")
            if actual_results["avg_distance"] is not None:
                print(f"    'avg_distance': {actual_results['avg_distance']},")
            else:
                print(f"    'avg_distance': None,")
            if actual_results["max_distance"] is not None:
                print(f"    'max_distance': {actual_results['max_distance']},")
            else:
                print(f"    'max_distance': None,")

            # Performance regression checks - this should be "this good or better"
            expected_performance_baseline = {
                "moles_found": 3,
                "matched_count": 2,
                "avg_distance": 178.33,
                "max_distance": 512.52,
            }

            assert (
                performance_metrics["moles_found"]
                >= expected_performance_baseline["moles_found"]
            ), (
                f"Performance regression: found {performance_metrics['moles_found']} moles, expected >= {expected_performance_baseline['moles_found']}"
            )
            assert (
                performance_metrics["matched_count"]
                >= expected_performance_baseline["matched_count"]
            ), (
                f"Performance regression: matched {performance_metrics['matched_count']} moles, expected >= {expected_performance_baseline['matched_count']}"
            )

        finally:
            # Restore original moles
            save_moles(target_json, original_moles)


def download_and_extract_dataset(url: str) -> pathlib.Path:
    """Download and extract the benchmark dataset."""
    dataset_archive = pathlib.Path("benchmark_dataset.tar.gz")

    print(f"Downloading benchmark dataset from {url}...")
    urllib.request.urlretrieve(url, dataset_archive)

    extract_path = pathlib.Path(".")
    print(f"Extracting dataset to {extract_path}...")
    with tarfile.open(dataset_archive, "r:gz") as tar:
        tar.extractall(extract_path)

    return extract_path


def read_moles(json_path: pathlib.Path) -> list:
    """Read moles from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def save_moles(json_path: pathlib.Path, moles: list) -> None:
    """Save moles to JSON file."""
    with open(json_path, "w") as f:
        json.dump(moles, f, indent=2)


def calculate_performance_metrics(removed_moles: list, result_moles: list) -> dict:
    """Calculate performance metrics by matching moles by UUID."""
    # Create UUID lookup for result moles
    result_moles_by_uuid = {mole["uuid"]: mole for mole in result_moles}

    # Calculate distances and matches for each removed mole
    distances = []
    matched_count = 0
    found_count = 0
    mole_results = []

    for removed_mole in removed_moles:
        removed_uuid = removed_mole["uuid"]
        canonical_x, canonical_y = removed_mole["x"], removed_mole["y"]

        # Check if this UUID was found in the results
        if removed_uuid in result_moles_by_uuid:
            found_mole = result_moles_by_uuid[removed_uuid]
            distance = math.sqrt(
                (found_mole["x"] - canonical_x) ** 2
                + (found_mole["y"] - canonical_y) ** 2
            )
            distances.append(distance)
            found_count += 1

            # Consider it a match if within 50 pixels
            if distance <= 50:
                matched_count += 1
                mole_results.append(
                    {"uuid": removed_uuid, "status": "matched", "distance": distance}
                )
            else:
                mole_results.append(
                    {"uuid": removed_uuid, "status": "found_far", "distance": distance}
                )
        else:
            mole_results.append(
                {"uuid": removed_uuid, "status": "not_found", "distance": None}
            )
            distances.append(float("inf"))

    return {
        "moles_found": found_count,
        "matched_count": matched_count,
        "match_rate": matched_count / len(removed_moles) if removed_moles else 0,
        "avg_distance": sum(d for d in distances if d != float("inf"))
        / len([d for d in distances if d != float("inf")])
        if any(d != float("inf") for d in distances)
        else float("inf"),
        "max_distance": max(d for d in distances if d != float("inf"))
        if any(d != float("inf") for d in distances)
        else float("inf"),
        "distances": distances,
        "mole_results": mole_results,
    }


@contextlib.contextmanager
def chtempdir_context():
    """Context manager for working in a temporary directory."""
    with tempfile.TemporaryDirectory() as tempdir:
        saved_path = os.getcwd()
        os.chdir(tempdir)
        try:
            yield
        finally:
            os.chdir(saved_path)


def test_benchmark_automark3():
    """Test automark3 performance using benchmark dataset."""
    with chtempdir_context():
        # Download and extract benchmark dataset
        dataset_url = (
            "https://github.com/aevri/mel-datasets/archive/refs/tags/v0.1.0.tar.gz"
        )
        dataset_path = download_and_extract_dataset(dataset_url)

        # Set up paths to test images
        m1_path = dataset_path / "mel-datasets-0.1.0" / "m1"
        source_image = (
            m1_path / "rotomaps" / "parts" / "Trunk" / "Back" / "2025_06_12" / "0.jpg"
        )
        target_image = (
            m1_path / "rotomaps" / "parts" / "Trunk" / "Back" / "2025_06_13" / "0.jpg"
        )
        target_json = target_image.with_suffix(".jpg.json")

        # Verify files exist
        assert source_image.exists(), f"Source image not found: {source_image}"
        assert target_image.exists(), f"Target image not found: {target_image}"
        assert target_json.exists(), f"Target JSON not found: {target_json}"

        # Read original moles and remove first 3 for benchmarking
        original_moles = read_moles(target_json)
        assert len(original_moles) >= 3, (
            f"Need at least 3 moles for benchmarking, found {len(original_moles)}"
        )

        removed_moles = original_moles[:3]
        remaining_moles = original_moles[3:]

        # Save modified JSON with removed moles
        save_moles(target_json, remaining_moles)

        try:
            # Run mel rotomap automark3
            expect_ok(
                "mel",
                "rotomap",
                "automark3",
                "--reference",
                str(source_image),
                "--target",
                str(target_image),
                "--dino-size",
                "small",
            )

            # Read results and measure performance
            result_moles = read_moles(target_json)
            performance_metrics = calculate_performance_metrics(
                removed_moles, result_moles
            )

            # Print detailed per-mole results
            print(f"Per-mole results (automark3):")
            for mole_result in performance_metrics["mole_results"]:
                status = mole_result["status"]
                uuid_short = mole_result["uuid"][:8]
                if status == "matched":
                    print(
                        f"  ✓ Mole {uuid_short}: MATCHED at distance {mole_result['distance']:.1f} pixels"
                    )
                elif status == "found_far":
                    print(
                        f"  ✗ Mole {uuid_short}: FOUND but distance {mole_result['distance']:.1f} pixels > 50 pixel threshold"
                    )
                else:  # not_found
                    print(f"  ✗ Mole {uuid_short}: NOT FOUND")

            # Print performance summary
            print(f"\nBenchmark Results (automark3):")
            print(f"  Original moles removed: {len(removed_moles)}")
            print(
                f"  Moles found by automark3: {performance_metrics['moles_found']}"
            )
            print(f"  Match rate: {performance_metrics['match_rate']:.2%}")
            if performance_metrics["avg_distance"] != float("inf"):
                print(
                    f"  Average distance to canonical: {performance_metrics['avg_distance']:.2f} pixels"
                )
                print(
                    f"  Max distance to canonical: {performance_metrics['max_distance']:.2f} pixels"
                )

            # Print actual results for easy copy-paste updating
            actual_results = {
                "moles_found": performance_metrics["moles_found"],
                "matched_count": performance_metrics["matched_count"],
                "avg_distance": round(performance_metrics["avg_distance"], 2)
                if performance_metrics["avg_distance"] != float("inf")
                else None,
                "max_distance": round(performance_metrics["max_distance"], 2)
                if performance_metrics["max_distance"] != float("inf")
                else None,
            }
            print(f"\nActual results (copy to update expected_performance_baseline):")
            print(f"    'moles_found': {actual_results['moles_found']},")
            print(f"    'matched_count': {actual_results['matched_count']},")
            if actual_results["avg_distance"] is not None:
                print(f"    'avg_distance': {actual_results['avg_distance']},")
            else:
                print(f"    'avg_distance': None,")
            if actual_results["max_distance"] is not None:
                print(f"    'max_distance': {actual_results['max_distance']},")
            else:
                print(f"    'max_distance': None,")

            # Performance regression checks - this should be "this good or better"
            # Using relaxed baseline for initial implementation
            expected_performance_baseline = {
                "moles_found": 2,
                "matched_count": 1,
                "avg_distance": None,  # Will set after first run
                "max_distance": None,  # Will set after first run
            }

            assert (
                performance_metrics["moles_found"]
                >= expected_performance_baseline["moles_found"]
            ), (
                f"Performance regression: found {performance_metrics['moles_found']} moles, expected >= {expected_performance_baseline['moles_found']}"
            )
            assert (
                performance_metrics["matched_count"]
                >= expected_performance_baseline["matched_count"]
            ), (
                f"Performance regression: matched {performance_metrics['matched_count']} moles, expected >= {expected_performance_baseline['matched_count']}"
            )

        finally:
            # Restore original moles
            save_moles(target_json, original_moles)


def expect_ok(*args):
    """Run command and expect it to succeed."""
    subprocess.check_call(args)


# -----------------------------------------------------------------------------
# Copyright (C) 2015-2025 Angelos Evripiotis.
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
