"""Compare numerical course score distributions across Hogwarts houses."""

import math
import sys
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pathlib import Path

from describe import identify_numerical_features, read_csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "dataset_train.csv"
HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
HOUSE_COLORS = {
    "Gryffindor": "red",
    "Hufflepuff": "yellow",
    "Ravenclaw": "blue",
    "Slytherin": "green",
}


def organize_scores_by_house(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    numerical_features: Sequence[tuple[int, str]],
) -> dict[str, dict[str, list[float]]]:
    if "Hogwarts House" not in header:
        raise ValueError("CSV is missing the Hogwarts House column")
    house_index = header.index("Hogwarts House")
    course_scores = {
        feature: {house: [] for house in HOUSES}
        for _, feature in numerical_features
    }
    for row in rows:
        house = row[house_index]
        if house not in HOUSES:
            continue
        for feature_index, feature in numerical_features:
            raw_value = row[feature_index].strip()
            if raw_value:
                course_scores[feature][house].append(float(raw_value))
    return course_scores


def plot_course_histogram(
    axis: Axes,
    course_name: str,
    house_scores: dict[str, list[float]],
) -> None:
    scores = [score for house in HOUSES for score in house_scores[house]]
    shared_bin_edges = np.histogram_bin_edges(scores, bins="auto")
    for house in HOUSES:
        axis.hist(
            house_scores[house],
            bins=shared_bin_edges,
            color=HOUSE_COLORS[house],
            alpha=0.5,
            label=house,
            histtype="bar",
        )
    axis.set_title(course_name)
    axis.set_xlabel("Score")
    axis.set_ylabel("Frequency")
    axis.legend(loc="upper right")
    axis.grid(False)


def plot_histograms(course_scores: dict[str, dict[str, list[float]]]) -> None:
    number_of_courses = len(course_scores)
    columns = 4
    rows = math.ceil(number_of_courses / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(16, 3.5 * rows), constrained_layout=True, squeeze=False
    )
    axes = axes.flatten()
    for axis, (course, house_scores) in zip(axes, course_scores.items()):
        plot_course_histogram(axis, course, house_scores)
    for axis in axes[number_of_courses:]:
        axis.set_axis_off()
    figure.suptitle("Hogwarts Course Score Distributions by House")
    plt.show()


def main() -> int:
    try:
        header, rows = read_csv(DATASET_PATH)
        numerical_features = identify_numerical_features(header, rows)
        course_scores = organize_scores_by_house(header, rows, numerical_features)
        plot_histograms(course_scores)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
