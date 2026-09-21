"""Train one-vs-all logistic regression using manual batch gradient descent."""

import csv
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from describe import is_numerical_feature, read_csv


FEATURES = [
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Ancient Runes",
]
HOUSES = [
    "Gryffindor",
    "Hufflepuff",
    "Ravenclaw",
    "Slytherin",
]
LEARNING_RATE = 0.01
ITERATIONS = 10000
THETA_PATH = Path("thetas.csv")


def validate_arguments() -> Path:
    if len(sys.argv) != 2:
        raise ValueError("usage: python3 src/logreg_train.py <csv_file>")
    return Path(sys.argv[1])


def validate_training_columns(
    headers: Sequence[str], rows: Sequence[Sequence[str]]
) -> tuple[int, list[int]]:
    if "Hogwarts House" not in headers:
        raise ValueError("CSV is missing the Hogwarts House column")
    house_index = headers.index("Hogwarts House")
    feature_columns = []
    for feature_name in FEATURES:
        if feature_name not in headers:
            raise ValueError(f"CSV is missing the {feature_name} column")
        feature_index = headers.index(feature_name)
        if not is_numerical_feature(rows, feature_index):
            raise ValueError(f"{feature_name} must contain finite numerical values")
        feature_columns.append(feature_index)
    return house_index, feature_columns


def organize_training_data(
    rows: Sequence[Sequence[str]], house_index: int, feature_columns: Sequence[int]
) -> tuple[list[list[float]], list[str]]:
    X = []
    houses = []
    for row in rows:
        house = row[house_index].strip()
        if house not in HOUSES:
            continue
        feature_values = []
        for feature_index in feature_columns:
            cell = row[feature_index].strip()
            if not cell:
                break
            feature_values.append(float(cell))
        if len(feature_values) != len(feature_columns):
            continue
        X.append(feature_values)
        houses.append(house)
    if not X:
        raise ValueError("CSV contains no valid training rows")
    for house in HOUSES:
        if house not in houses:
            raise ValueError(f"CSV contains no valid training rows for {house}")
    return X, houses


def normalize_feature(values: Sequence[float]) -> tuple[list[float], float, float]:
    minimum = min(values)
    maximum = max(values)
    value_range = maximum - minimum
    if value_range == 0:
        raise ValueError("cannot normalize a constant training feature")
    if not math.isfinite(value_range):
        raise ValueError("training feature range is not finite")
    normalized_values = [(value - minimum) / value_range for value in values]
    return normalized_values, minimum, maximum


def normalize_features(
    X: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[float], list[float]]:
    normalized_X = [[0.0] * len(FEATURES) for _ in X]
    minimums = []
    maximums = []
    for feature_index in range(len(FEATURES)):
        values = [sample[feature_index] for sample in X]
        normalized_values, minimum, maximum = normalize_feature(values)
        for sample_index, value in enumerate(normalized_values):
            normalized_X[sample_index][feature_index] = value
        minimums.append(minimum)
        maximums.append(maximum)
    return normalized_X, minimums, maximums


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    exponential = math.exp(z)
    return exponential / (1 + exponential)


def calculate_probabilities(
    X: Sequence[Sequence[float]], weights: Sequence[float], bias: float
) -> list[float]:
    probabilities = []
    for sample in X:
        z = bias
        for feature_index in range(len(FEATURES)):
            z += weights[feature_index] * sample[feature_index]
        probabilities.append(sigmoid(z))
    return probabilities


def calculate_gradients(
    X: Sequence[Sequence[float]], y: Sequence[int], probabilities: Sequence[float]
) -> tuple[list[float], float]:
    m = len(X)
    weight_gradients = [0.0] * len(FEATURES)
    bias_gradient = 0.0
    for sample_index, sample in enumerate(X):
        error = probabilities[sample_index] - y[sample_index]
        bias_gradient += error
        for feature_index in range(len(FEATURES)):
            weight_gradients[feature_index] += error * sample[feature_index]
    weight_gradients = [gradient / m for gradient in weight_gradients]
    bias_gradient /= m
    if not math.isfinite(bias_gradient) or any(
        not math.isfinite(gradient) for gradient in weight_gradients
    ):
        raise ValueError("training gradients are not finite")
    return weight_gradients, bias_gradient


def gradient_descent(
    X: Sequence[Sequence[float]], y: Sequence[int]
) -> tuple[float, list[float]]:
    weights = [0.0] * len(FEATURES)
    bias = 0.0
    for _ in range(ITERATIONS):
        probabilities = calculate_probabilities(X, weights, bias)
        weight_gradients, bias_gradient = calculate_gradients(X, y, probabilities)
        for feature_index in range(len(FEATURES)):
            weights[feature_index] -= LEARNING_RATE * weight_gradients[feature_index]
        bias -= LEARNING_RATE * bias_gradient
        if not math.isfinite(bias) or any(
            not math.isfinite(weight) for weight in weights
        ):
            raise ValueError("trained parameters are not finite")
    return bias, weights


def train_one_vs_all(
    X: Sequence[Sequence[float]],
    houses: Sequence[str],
    optimizer: Callable[
        [Sequence[Sequence[float]], Sequence[int]], tuple[float, list[float]]
    ],
) -> dict[str, tuple[float, list[float]]]:
    models = {}
    for target_house in HOUSES:
        y = [1 if house == target_house else 0 for house in houses]
        models[target_house] = optimizer(X, y)
    return models


def convert_thetas(
    bias: float,
    weights: Sequence[float],
    minimums: Sequence[float],
    maximums: Sequence[float],
) -> tuple[float, list[float]]:
    converted_bias = bias
    converted_weights = []
    for feature_index in range(len(FEATURES)):
        feature_range = maximums[feature_index] - minimums[feature_index]
        converted_weight = weights[feature_index] / feature_range
        converted_bias -= converted_weight * minimums[feature_index]
        converted_weights.append(converted_weight)
    if not math.isfinite(converted_bias) or any(
        not math.isfinite(weight) for weight in converted_weights
    ):
        raise ValueError("converted parameters are not finite")
    return converted_bias, converted_weights


def convert_models(
    models: dict[str, tuple[float, list[float]]],
    minimums: Sequence[float],
    maximums: Sequence[float],
) -> dict[str, tuple[float, list[float]]]:
    converted_models = {}
    for house in HOUSES:
        bias, weights = models[house]
        converted_models[house] = convert_thetas(bias, weights, minimums, maximums)
    return converted_models


def save_thetas(models: dict[str, tuple[float, list[float]]], theta_path: Path) -> None:
    try:
        with open(theta_path, "w", encoding="utf-8", newline="") as theta_file:
            writer = csv.writer(theta_file)
            writer.writerow(["House", "Bias", *FEATURES])
            for house in HOUSES:
                bias, weights = models[house]
                writer.writerow([house, bias, *weights])
    except (OSError, csv.Error) as error:
        raise ValueError(f"Could not save thetas to '{theta_path}': {error}") from error


def main() -> int:
    try:
        dataset_path = validate_arguments()
        headers, rows = read_csv(dataset_path)
        house_index, feature_columns = validate_training_columns(headers, rows)
        X, houses = organize_training_data(rows, house_index, feature_columns)
        normalized_X, minimums, maximums = normalize_features(X)
        normalized_models = train_one_vs_all(normalized_X, houses, gradient_descent)
        models = convert_models(normalized_models, minimums, maximums)
        save_thetas(models, THETA_PATH)
        print(f"Training complete. Thetas saved to '{THETA_PATH.name}'.")
        return 0
    except (OSError, UnicodeError, ValueError, ArithmeticError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
