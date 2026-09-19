"""Display numerical course distributions and pairs by Hogwarts house."""

import sys
import textwrap
from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from describe import identify_numerical_features, read_csv
from histogram import DATASET_PATH, HOUSE_COLORS, organize_scores_by_house
from scatter_plot import organize_pairs


def organize_pairs_by_house(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    numerical_features: Sequence[tuple[int, str]],
) -> dict[str, dict[str, dict[str, list[tuple[float, float]]]]]:
    """Group validated features by house, keeping only complete row pairs."""
    if "Hogwarts House" not in headers:
        raise ValueError("CSV is missing the Hogwarts House column")
    house_index = headers.index("Hogwarts House")
    pairs = {
        x_feature: {
            y_feature: {house: [] for house in HOUSE_COLORS}
            for _, y_feature in numerical_features
            if x_feature != y_feature
        }
        for _, x_feature in numerical_features
    }
    house_rows = {house: [] for house in HOUSE_COLORS}
    for row in rows:
        house = row[house_index]
        if house in house_rows:
            house_rows[house].append(row)

    for house, rows_for_house in house_rows.items():
        house_pairs = organize_pairs(rows_for_house, numerical_features)
        for x_feature, comparisons in house_pairs.items():
            for y_feature, points in comparisons.items():
                pairs[x_feature][y_feature][house] = points
                pairs[y_feature][x_feature][house] = [(y, x) for x, y in points]
    return pairs


def plot_histogram(
    ax: Axes,
    feature: str,
    histogram_data: dict[str, dict[str, list[float]]],
) -> None:
    for house, color in HOUSE_COLORS.items():
        ax.hist(
            histogram_data[feature][house],
            color=color,
            alpha=0.5,
            label=house,
            histtype="bar",
        )


def plot_scatter(
    ax: Axes,
    x_feature: str,
    y_feature: str,
    scatter_data: dict[str, dict[str, dict[str, list[tuple[float, float]]]]],
) -> None:
    for house, color in HOUSE_COLORS.items():
        points = scatter_data[x_feature][y_feature][house]
        x_values = [x for x, _ in points]
        y_values = [y for _, y in points]
        ax.scatter(x_values, y_values, color=color, s=10, alpha=0.5, label=house)


def configure_axes(
    ax: Axes,
    row: int,
    column: int,
    feature_count: int,
    x_feature: str,
    y_feature: str,
) -> None:
    if column != 0:
        ax.tick_params(axis="y", left=False, right=False, labelleft=False)
        ax.yaxis.get_offset_text().set_visible(False)
    else:
        ax.set_ylabel(
            textwrap.fill(y_feature, width=16),
            fontsize=8,
            rotation=30,
            ha="right",
            va="bottom",
        )
    if row != feature_count - 1:
        ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False)
        ax.xaxis.get_offset_text().set_visible(False)
    else:
        ax.set_xlabel(
            textwrap.fill(x_feature, width=16),
            fontsize=8,
        )


def plot_pair_matrix(
    numerical_features: Sequence[tuple[int, str]],
    histogram_data: dict[str, dict[str, list[float]]],
    scatter_data: dict[str, dict[str, dict[str, list[tuple[float, float]]]]],
) -> None:
    feature_count = len(numerical_features)
    figure, axes = plt.subplots(
        nrows=feature_count,
        ncols=feature_count,
        figsize=(18, 18),
        constrained_layout=True,
        squeeze=False,
    )
    for row, (_, y_feature) in enumerate(numerical_features):
        for column, (_, x_feature) in enumerate(numerical_features):
            ax = axes[row][column]
            if row == column:
                plot_histogram(ax, x_feature, histogram_data)
            else:
                plot_scatter(ax, x_feature, y_feature, scatter_data)
            configure_axes(
                ax, row, column, feature_count, x_feature, y_feature
            )

    handles = [
        Patch(facecolor=color, alpha=0.5, label=house)
        for house, color in HOUSE_COLORS.items()
    ]
    figure.align_ylabels(axes[:, 0])
    figure.legend(handles=handles, loc="outside right upper")
    plt.show()


def main() -> int:
    try:
        headers, rows = read_csv(DATASET_PATH)
        numerical_features = identify_numerical_features(headers, rows)
        histogram_data = organize_scores_by_house(headers, rows, numerical_features)
        scatter_data = organize_pairs_by_house(headers, rows, numerical_features)
        plot_pair_matrix(numerical_features, histogram_data, scatter_data)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
