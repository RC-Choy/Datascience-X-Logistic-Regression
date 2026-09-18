"""Visually compare every unique pair of numerical course features."""

import math
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button

from describe import identify_numerical_features, read_csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "dataset_train.csv"


def organize_pairs(
    rows: Sequence[Sequence[str]], features: Sequence[tuple[int, str]]
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Keep row pairs intact; features must already be numerically validated."""
    pairs = {}
    for i, (x_index, x_name) in enumerate(features[:-1]):
        pairs[x_name] = {}
        for y_index, y_name in features[i + 1:]:
            points = []
            for row in rows:
                x_cell = row[x_index].strip()
                y_cell = row[y_index].strip()
                if not x_cell or not y_cell:
                    continue
                points.append((float(x_cell), float(y_cell)))
            pairs[x_name][y_name] = points
    return pairs


def plot_scatter(
    axis: Axes, x_name: str, y_name: str, points: Sequence[tuple[float, float]]
) -> None:
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    axis.scatter(x_values, y_values, s=10, alpha=0.5)
    axis.set_title(f"vs {y_name}")
    axis.set_xlabel(x_name)
    axis.set_ylabel(y_name)


def create_pages(
    figure: Figure,
    features: Sequence[tuple[int, str]],
    pairs: dict[str, dict[str, list[tuple[float, float]]]],
) -> list[list[Axes]]:
    pages = []
    columns = 4
    # The first page has the most comparisons; every page shares its grid.
    rows = math.ceil((len(features) - 1) / columns)
    figure.set_size_inches(16, 3.5 * rows)
    grid = figure.add_gridspec(rows, columns)
    for _, base_feature in features[:-1]:
        page_axes = [
            figure.add_subplot(grid[position]) for position in range(rows * columns)
        ]
        comparisons = pairs[base_feature]
        for axis, (compared_feature, points) in zip(page_axes, comparisons.items()):
            plot_scatter(axis, base_feature, compared_feature, points)
        for axis in page_axes[len(comparisons):]:
            axis.set_visible(False)
        pages.append(page_axes)
    return pages


def set_page_visibility(page_axes: Sequence[Axes], visible: bool) -> None:
    for axis in page_axes:
        # Even an empty scatter plot has a collection; unused positions do not.
        if axis.collections:
            axis.set_visible(visible)


def main() -> int:
    try:
        headers, rows = read_csv(DATASET_PATH)
        features = identify_numerical_features(headers, rows)
        if len(features) < 2:
            raise ValueError("CSV must contain at least two numerical course features")
        pairs = organize_pairs(rows, features)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    figure = plt.figure(constrained_layout=True)
    figure.get_layout_engine().set(rect=(0, 0.10, 1, 0.90))
    pages = create_pages(figure, features, pairs)
    title = figure.suptitle(
        f"{features[0][1]} comparisons — Page 1 / {len(pages)}"
    )

    previous_button = Button(figure.add_axes((0.39, 0.025, 0.10, 0.04)), "Previous")
    next_button = Button(figure.add_axes((0.51, 0.025, 0.10, 0.04)), "Next")

    current_page = 0

    def show_page(page_index: int) -> None:
        for index, page_axes in enumerate(pages):
            set_page_visibility(page_axes, index == page_index)
        title.set_text(
            f"{features[page_index][1]} comparisons — "
            f"Page {page_index + 1} / {len(pages)}"
        )
        figure.canvas.draw_idle()

    def previous_page(event: object) -> None:
        nonlocal current_page
        if current_page > 0:
            current_page -= 1
            show_page(current_page)

    def next_page(event: object) -> None:
        nonlocal current_page
        if current_page < len(pages) - 1:
            current_page += 1
            show_page(current_page)

    previous_button.on_clicked(previous_page)
    next_button.on_clicked(next_page)
    show_page(current_page)
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
