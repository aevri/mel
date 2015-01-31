"""Analyse statistics recorded about moles."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy

import mel.lib.moleimaging


def setup_parser(parser):
    parser.add_argument(
        'path',
        type=str,
        nargs='+',
        help="Paths to images to analyse.")


def process_args(args):

    name_stats = []
    for filename in args.path:
        image = cv2.imread(filename)
        _, stats = mel.lib.moleimaging.find_mole(image)
        name = filename.split('_')[0]
        if stats is not None:
            name_stats.append((name, stats))

    # Train on all of the data except one element, then try to predict the name
    # of that element.
    #
    # Loop and do this for each element in the list.
    #
    successes = 0
    for i in xrange(len(name_stats)):
        classifier = NamedClassifier()

        training_samples = name_stats[:i] + name_stats[i + 1:]
        for name, stats in training_samples:
            classifier.add_sample(stats, name)

        classifier.train()

        name = name_stats[i][0]
        stats = name_stats[i][1]
        predicted_name = classifier.predict(stats)
        if predicted_name != name:
            print('mistake:', name, '->', predicted_name)
        else:
            successes += 1

    print('accuracy:', (successes / len(name_stats)) * 100, '%')


class NamedClassifier(object):

    def __init__(self):
        super(NamedClassifier, self).__init__()
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
        return self._number_to_name[int(class_num)]


class Classifier(object):

    def __init__(self):
        super(Classifier, self).__init__()
        self._training_data = []
        self._responses = []
        self._svm = cv2.SVM()

    def add_sample(self, data, class_num):
        self._training_data.append(data)
        self._responses.append(class_num)

    def train(self):

        svm_params = {
            "kernel_type": cv2.SVM_LINEAR,
            "svm_type": cv2.SVM_C_SVC,
            "C": 1
        }

        training_data = numpy.array(
            [
                numpy.array(x, dtype=numpy.float32)
                for x in self._training_data
            ],
            dtype=numpy.float32)

        responses = numpy.array(self._responses, dtype=numpy.float32)

        self._svm.train(
            training_data,
            responses,
            params=svm_params)

    def predict(self, data):
        return self._svm.predict(
            numpy.array(data, dtype=numpy.float32))


def _test_named_classifier():

    classifier = NamedClassifier()

    classifier.add_sample([0.0, 1.0], "badger")
    classifier.add_sample([1.0, 1.0], "orange")
    classifier.add_sample([1.0, 1.0], "orange")
    classifier.add_sample([0.0, 1.0], "badger")
    classifier.add_sample([1.0, 1.0], "orange")
    classifier.add_sample([0.0, 1.0], "badger")
    classifier.add_sample([0.5, 0.5], "tea")
    classifier.add_sample([0.5, 0.5], "tea")
    classifier.add_sample([0.5, 0.5], "tea")

    classifier.train()

    print(classifier.predict([1.0, 1.0]))
    print(classifier.predict([0.0, 1.0]))
    print(classifier.predict([0.5, 0.5]))


def _test_classifier():

    classifier = Classifier()

    classifier.add_sample([0.0, 1.0], 2.0)
    classifier.add_sample([1.0, 1.0], 0.0)
    classifier.add_sample([1.0, 1.0], 0.0)
    classifier.add_sample([0.0, 1.0], 2.0)
    classifier.add_sample([1.0, 1.0], 0.0)
    classifier.add_sample([0.0, 1.0], 2.0)
    classifier.add_sample([0.5, 0.5], 1.0)
    classifier.add_sample([0.5, 0.5], 1.0)
    classifier.add_sample([0.5, 0.5], 1.0)

    classifier.train()

    print(classifier.predict([1.0, 1.0]))
    print(classifier.predict([0.0, 1.0]))
    print(classifier.predict([0.5, 0.5]))
