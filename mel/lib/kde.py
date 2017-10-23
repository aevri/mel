"""Kernel Density Estimate."""

class Kde():

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
