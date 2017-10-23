"""Kernel Density Estimate."""

import scipy.linalg
import scipy.stats


class Kde():

    def __init__(self, training_data):
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
