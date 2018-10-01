"""Kernel Density Estimate."""

import numpy


class Kde:
    def __init__(self, training_data):

        # These imports take quite a long time. At the time of writing this is
        # the only place we need them, so avoid paying the cost if we can by
        # moving them into the only method that uses them.
        import scipy.linalg
        import scipy.stats

        self.len = training_data.shape[-1]

        if self.len < 3:
            self.attenuation = 0.0
            self.kde = lambda x: numpy.array((0.0,))
            return
        else:
            self.attenuation = 1

        try:
            self.kde = scipy.stats.gaussian_kde(training_data)
        except scipy.linalg.LinAlgError as e:
            print(e)
            print(training_data)
            raise

    def __call__(self, lower, upper):
        if self.attenuation:
            return self.kde.integrate_box(lower, upper)
        else:
            return 0


# -----------------------------------------------------------------------------
# Copyright (C) 2017 Angelos Evripiotis.
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
