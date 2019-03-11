"""Kernel Density Estimate."""

import numpy


class Kde:
    def __init__(self, training_data, box_radius):

        # These imports take quite a long time. At the time of writing this is
        # the only place we need them, so avoid paying the cost if we can by
        # moving them into the only method that uses them.
        import scipy.linalg
        import scipy.stats

        self._box_radius = box_radius * 1
        self._lower = numpy.array((self._box_radius, self._box_radius))
        self._upper = numpy.array((-self._box_radius, -self._box_radius))
        self._kde = None
        self._mid = None

        _len = training_data.shape[-1]
        if _len < 1:
            return
        elif _len < 3:
            self._mid = numpy.array([training_data[0][0], training_data[1][0]])
            return

        try:
            self._kde = scipy.stats.gaussian_kde(training_data)
        except scipy.linalg.LinAlgError as e:
            print(e)
            print(training_data)
            raise

    def __call__(self, pos):
        if self._kde is not None:
            return self._kde.integrate_box(pos + self._lower, pos + self.upper)
        elif self._mid is not None:
            adj_pos = pos - self._mid
            mag = numpy.linalg.norm(adj_pos) / self._box_radius
            print(mag)
            if mag <= 1:
                mag *= mag
                mag = 1 - mag
                print('mag', mag)
                return mag
            return 0
        else:
            return 0


# -----------------------------------------------------------------------------
# Copyright (C) 2018 Angelos Evripiotis.
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
