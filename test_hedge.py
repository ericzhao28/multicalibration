import pytest
import numpy as np
from hedge import OnlineGradientDescent, Hedge, MLProd, OptimisticHedge


@pytest.fixture
def test_setup():
    setup_data = {
        "n_actions": 3,
        "learning_rate": 0.5,
        "loss_vector": np.array([0.1, 0.2, 0.3]),
    }
    return setup_data


def test_hedge(test_setup):
    hedge = Hedge(test_setup["n_actions"], test_setup["learning_rate"])
    assert hedge.n_actions == test_setup["n_actions"]
    assert hedge.learning_rate == test_setup["learning_rate"]
    assert np.allclose(
        hedge.weights, np.ones(test_setup["n_actions"]) / test_setup["n_actions"]
    )
    action = hedge.predict()
    assert action in range(test_setup["n_actions"])
    hedge.update(test_setup["loss_vector"])
    assert np.all(hedge.weights <= 1)


def test_optimistic_hedge(test_setup):
    hedge = OptimisticHedge(test_setup["n_actions"], test_setup["learning_rate"])
    assert hedge.n_actions == test_setup["n_actions"]
    assert hedge.learning_rate == test_setup["learning_rate"]
    assert np.allclose(
        hedge.weights, np.ones(test_setup["n_actions"]) / test_setup["n_actions"]
    )
    action = hedge.predict()
    assert action in range(test_setup["n_actions"])
    hedge.update(test_setup["loss_vector"])
    assert np.all(hedge.weights <= 1)


def test_mlprod(test_setup):
    mlprod = MLProd(test_setup["n_actions"], test_setup["learning_rate"])
    assert mlprod.n_actions == test_setup["n_actions"]
    assert np.allclose(mlprod.learning_rate, test_setup["learning_rate"])
    assert np.allclose(
        mlprod.w, np.ones(test_setup["n_actions"]) / test_setup["n_actions"]
    )
    action = mlprod.predict()
    assert action in range(test_setup["n_actions"])
    mlprod.update(test_setup["loss_vector"])
    assert np.all(mlprod.w <= 1)


def test_online_gradient_descent(test_setup):
    ogd = OnlineGradientDescent(test_setup["n_actions"], test_setup["learning_rate"])
    assert ogd.n_actions == test_setup["n_actions"]
    assert ogd.learning_rate == test_setup["learning_rate"]
    assert np.allclose(
        ogd.weights, np.ones(test_setup["n_actions"]) / test_setup["n_actions"]
    )
    action = ogd.predict()
    assert action in range(test_setup["n_actions"])
    ogd.update(test_setup["loss_vector"])
    assert np.all(ogd.weights >= 0)
    assert np.all(ogd.weights <= 1)


def test_algorithms_convergence():
    n_actions = 3
    learning_rate = 0.1
    iterations = 1000
    convergence_threshold = 1e-4

    algorithms = [
        Hedge(n_actions, learning_rate),
        MLProd(n_actions, learning_rate),
        OnlineGradientDescent(n_actions, learning_rate),
        OptimisticHedge(n_actions, learning_rate),
    ]

    # Loss vector for our toy problem
    loss_vector = np.array([0.1, 0.2, 0.3])

    for algo in algorithms:
        # Run the algorithm for some iterations
        for _ in range(iterations):
            old_weights = algo.weights.copy()
            algo.update(loss_vector)
            new_weights = algo.weights

        # Check convergence
        assert np.allclose(
            old_weights, new_weights, atol=convergence_threshold
        ), f"{algo.__class__.__name__} did not converge"

        # Check if the algorithm gives more weight to the action with the smallest loss
        assert np.argmax(algo.weights) == np.argmin(
            loss_vector
        ), f"{algo.__class__.__name__} did not assign the smallest weight to the action with the smallest loss"

        assert np.allclose(
            new_weights, [1, 0, 0], atol=convergence_threshold
        ), f"{algo.__class__.__name__} did not solve"
