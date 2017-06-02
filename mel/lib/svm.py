"""Convenience wrappers for the OpenCV SVM classifier.

As a beginner user of SVM, it seems to be wise to follow the advice given in
this paper:

    http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf

In summary:

    - For category input data, prefer a binary vector (0, 0, 1) instead of a
      single value in the range of [0, 2].
    - Scale input values to the range of [-1, +1] or [0, 1].
    - If in doubt, use the RBF kernel to start with.
    - Do a grid search for the best (C, gamma) parameter pair for your data.
    - Use cross-validation to determine the fitness of each pair.

"""

import cv2
import numpy


class NamedClassifier(object):

    def __init__(self, classifier=None):
        super(NamedClassifier, self).__init__()

        self._classifier = classifier
        if self._classifier is None:
            self._classifier = Classifier()

        self._name_to_number = {}
        self._number_to_name = {}
        self._next_class_num = 0

    def add_sample(self, data, class_name):

        if class_name not in self._name_to_number:
            class_num = self._next_class_num
            self._name_to_number[class_name] = class_num
            self._number_to_name[class_num] = class_name
            self._next_class_num += 1
        else:
            class_num = self._name_to_number[class_name]

        self._classifier.add_sample(data, class_num)

    def train(self):
        self._classifier.train()

    def predict(self, data):
        class_num = self._classifier.predict(data)
        return self._number_to_name[class_num]


class Classifier(object):

    def __init__(self, c=None, gamma=None):
        super(Classifier, self).__init__()

        if c is None:
            c = 1.0
        elif c <= 0:
            raise ValueError("'c' must be more than zero", c)

        if gamma is None:
            gamma = 1.0
        elif gamma <= 0:
            raise ValueError("'gamma' must be more than zero", gamma)

        self._training_data = []
        self._responses = []

        self._svm = cv2.ml.SVM_create()
        self._svm.setC(c)
        self._svm.setGamma(gamma)
        self._svm.setKernel(cv2.ml.SVM_RBF)
        self._svm.setType(cv2.ml.SVM_C_SVC)

    def add_sample(self, data, class_num):
        self._training_data.append(data)
        self._responses.append(class_num)

    def train(self):

        training_data = numpy.array(
            [
                numpy.array(x, dtype=numpy.float32)
                for x in self._training_data
            ],
            dtype=numpy.float32)

        responses = numpy.array(self._responses, dtype=numpy.int32)

        self._svm.train(
            training_data,
            cv2.ml.ROW_SAMPLE,
            responses)

    def predict(self, sample):
        data = numpy.array([sample], dtype=numpy.float32)
        return int(self._svm.predict(data)[1].ravel())
