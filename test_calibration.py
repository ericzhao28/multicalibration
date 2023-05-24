import pytest
import numpy as np

from calibration import (
    calculate_average_labels,
    discretize_values,
    calculate_calibration_errors,
)


def test_calculate_average_labels():
    labels = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    bins = np.array([0, 1, 0, 1])
    groups = [[0, 2], [1, 3]]
    n_bins = 2
    expected_average_labels = np.array(
        [[[0.3, 0.4], [0, 0]], [[0, 0], [0.5, 0.6]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    )
    expected_counts = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]])

    avg_labels, counts = calculate_average_labels(labels, bins, groups, n_bins)

    np.testing.assert_array_almost_equal(avg_labels, expected_average_labels)
    np.testing.assert_array_almost_equal(counts, expected_counts)


def test_discretize_values():
    values = np.array([[0.1, 0.2], [0.7, 0.8]])
    n_bins = 2
    expected_result = np.array([0, 3])

    result = discretize_values(values, n_bins)

    np.testing.assert_array_equal(result, expected_result)


def test_calculate_calibration_errors():
    average_labels = np.array([[[0.3, 0.4], [0.5, 0.6]], [[0.3, 0.4], [0.7, 0.8]]])
    predictions = np.array([[0.2, 0.3], [0.6, 0.7], [0.4, 0.5], [0.8, 0.9]])
    bins = np.array([0, 1, 0, 1])
    groups = [[0, 2], [1, 3]]
    expected_result = np.array(
        [[[[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]
    )

    result = calculate_calibration_errors(average_labels, predictions, bins, groups)

    np.testing.assert_array_almost_equal(result, expected_result)
