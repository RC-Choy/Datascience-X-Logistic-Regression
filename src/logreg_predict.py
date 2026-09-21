"""Predict Hogwarts houses using saved one-vs-all logistic regression models."""

import csv
import sys
from collections.abc import Sequence
from pathlib import Path

from describe import is_numerical_feature, read_csv
from logreg_train import HOUSES, sigmoid


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "houses.csv"


def validate_arguments() -> tuple[Path, Path]:
    if len(sys.argv) != 3:
        raise ValueError("usage: python3 src/logreg_predict.py <dataset_test.csv> <model_parameters.csv>")
    return (Path(sys.argv[1]), Path(sys.argv[2]))


def load_model_parameters(
    model_path: Path,
) -> tuple[list[str], dict[str, float], dict[str, tuple[float, list[float]]]]:
    headers, rows = read_csv(model_path)

    if headers[:2] != ["House", "Bias"] or len(headers) < 3:
        raise ValueError(
            "Model CSV must start with House, Bias, followed by feature columns"
        )
    if not rows:
        raise ValueError("Model CSV is missing median and house model rows")

    features = headers[2:]
    median_row = rows[0]
    if median_row[0] != "Median":
        raise ValueError("First model CSV data row must be labelled Median")

    medians = {}
    for column, feature in enumerate(features, start=2):
        if not is_numerical_feature([median_row], column):
            raise ValueError(f"Median for {feature} must be a finite number")
        medians[feature] = float(median_row[column])

    models = {}
    for row in rows[1:]:
        house = row[0]
        if house not in HOUSES:
            raise ValueError(f"Model CSV contains unknown house: {house!r}")
        if house in models:
            raise ValueError(f"Model CSV contains duplicate house: {house}")

        for column in range(1, len(headers)):
            if not is_numerical_feature([row], column):
                raise ValueError(
                    f"{house}: {headers[column]} must be a finite number"
                )

        bias = float(row[1])
        weights = []
        for column in range(2, len(headers)):
            weights.append(float(row[column]))
        models[house] = (bias, weights)

    missing_houses = [house for house in HOUSES if house not in models]
    if missing_houses:
        raise ValueError(
            f"Model CSV is missing houses: {', '.join(missing_houses)}"
        )
    return features, medians, models


def validate_test_columns(
    headers: Sequence[str], rows: Sequence[Sequence[str]], features: Sequence[str]
) -> tuple[int, list[int]]:
    if "Index" not in headers:
        raise ValueError("CSV is missing the Index column")
    index_column = headers.index("Index")
    feature_columns = []
    for feature in features:
        if feature not in headers:
            raise ValueError(f"CSV is missing the {feature} column")
        column = headers.index(feature)
        if not is_numerical_feature(rows, column):
            raise ValueError(f"{feature} must contain finite numerical values")
        feature_columns.append(column)
    return index_column, feature_columns


def organize_test_data(
    rows: Sequence[Sequence[str]],
    index_column: int,
    feature_columns: Sequence[int],
    features: Sequence[str],
    medians: dict[str, float],
) -> tuple[list[str], list[list[float]]]:
    indices = []
    X = []
    for row in rows:
        indices.append(row[index_column])
        sample = []
        for feature, column in zip(features, feature_columns):
            cell = row[column].strip()
            value = float(cell) if cell else medians[feature]
            sample.append(value)
        X.append(sample)
    return indices, X


def calculate_probability(
    sample: Sequence[float], bias: float, weights: Sequence[float]
) -> float:
    z = bias
    for value, weight in zip(sample, weights):
        z += weight * value
    return sigmoid(z)


def predict_houses(
    X: Sequence[Sequence[float]], models: dict[str, tuple[float, list[float]]]
) -> list[str]:
    predictions = []
    for sample in X:
        probabilities = {}
        for house, (bias, weights) in models.items():
            probabilities[house] = calculate_probability(sample, bias, weights)
        predicted_house = max(probabilities, key=probabilities.get)
        predictions.append(predicted_house)
    return predictions


def save_predictions(
    indices: Sequence[str], predictions: Sequence[str], output_path: Path
) -> None:
    try:
        with open(output_path, "w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["Index", "Hogwarts House"])
            for index, house in zip(indices, predictions):
                writer.writerow([index, house])
    except (OSError, csv.Error) as error:
        raise ValueError(
            f"Could not save predictions to '{output_path}': {error}"
        ) from error


def main() -> int:
    try:
        dataset_path, model_path = validate_arguments()
        features, medians, models = load_model_parameters(model_path)
        headers, rows = read_csv(dataset_path)
        index_column, feature_columns = validate_test_columns(headers, rows, features)
        indices, X = organize_test_data(
            rows, index_column, feature_columns, features, medians
        )
        predictions = predict_houses(X, models)
        save_predictions(indices, predictions, OUTPUT_PATH)
        print(f"Prediction complete. Houses saved to '{OUTPUT_PATH.name}'.")
        return 0
    except (OSError, UnicodeError, ValueError, ArithmeticError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
