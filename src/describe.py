"""Describe numerical CSV features using manually calculated statistics."""

import csv
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from os import PathLike


INVALID = None
STATISTIC_NAMES = (
    "Count", "Missing Count", "Mean", "Std", "Min",
    "25%", "50%", "75%", "Max", "IQR",
)


def validate_arguments() -> str:
    if len(sys.argv) != 2:
        raise ValueError("usage: python3 src/describe.py <csv_file>")
    return sys.argv[1]


def validate_header(headers: Sequence[str] | None) -> None:
    if not headers or any(not name.strip() for name in headers):
        raise ValueError("CSV header is missing or contains an empty column name")
    seen: set[str] = set()
    for name in headers:
        column_name = name.strip()
        if column_name in seen:
            raise ValueError(f"CSV header contains duplicate column name: {column_name!r}")
        seen.add(column_name)


def validate_row(row: Sequence[str], header_count: int, row_number: int) -> bool:
    """Return False for blank rows; reject malformed non-blank rows."""
    if not any(cell.strip() for cell in row):
        return False
    if len(row) != header_count:
        raise ValueError(
            f"row {row_number}: expected {header_count} columns, got {len(row)}"
        )
    return True


def read_csv(file_path: str | PathLike[str]) -> tuple[list[str], list[list[str]]]:
    rows = []
    with open(file_path, encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file, strict=True)
        try:
            headers = next(reader, None)
            if headers is None:
                raise ValueError("CSV file is empty")
            validate_header(headers)
            while True:
                row_number = reader.line_num + 1
                row = next(reader, None)
                if row is None:
                    break
                if validate_row(row, len(headers), row_number):
                    rows.append(row)
        except csv.Error as error:
            raise ValueError(f"CSV row {reader.line_num}: {error}") from error
    return headers, rows


def is_numerical_feature(rows: Iterable[Sequence[str]], column_index: int) -> bool:
    found_value = False
    for row in rows:
        cell = row[column_index].strip()
        if not cell:
            continue
        try:
            value = float(cell)
        except (ValueError, OverflowError):
            return False
        if not math.isfinite(value):
            return False
        found_value = True
    return found_value


def identify_numerical_features(
    headers: Sequence[str], rows: Sequence[Sequence[str]]
) -> list[tuple[int, str]]:
    features = []
    for column_index, feature_name in enumerate(headers):
        if feature_name.strip() == "Index":
            continue
        if is_numerical_feature(rows, column_index):
            features.append((column_index, feature_name))
    if not features:
        raise ValueError("CSV contains no numerical features")
    return features


def extract_values(rows: Iterable[Sequence[str]], column_index: int) -> list[float]:
    values = []
    for row in rows:
        cell = row[column_index].strip()
        if cell:
            values.append(float(cell))
    return sorted(values)


def calculate_count(values: Iterable[float]) -> int:
    count = 0
    for _ in values:
        count += 1
    return count


def calculate_missing_count(total_rows: int, count: int) -> int:
    return total_rows - count


def calculate_mean(values: Iterable[float], count: int) -> float:
    total = 0.0
    for value in values:
        total += value
    return total / count


def calculate_std(values: Iterable[float], mean: float | None, count: int) -> float:
    if mean is INVALID:
        raise ValueError("depends on invalid Mean")
    squared_deviations = 0.0
    for value in values:
        squared_deviations += (value - mean) ** 2
    return math.sqrt(squared_deviations / count)


def calculate_min(sorted_values: Sequence[float]) -> float:
    return sorted_values[0]


def calculate_quartile(sorted_values: Sequence[float], p: float) -> float:
    """Use exclusive ranks p * (n + 1), with one-based statistical ranks."""
    count = len(sorted_values)
    if count < 2:
        raise ValueError("exclusive quartiles require at least two valid values")
    if p not in (0.25, 0.50, 0.75):
        raise ValueError("quartile probability must be 0.25, 0.50, or 0.75")
    position = p * (count + 1)
    lower_rank = int(position)
    # Keep two neighboring values available. For n=2, Q1 and Q3 extrapolate.
    if lower_rank < 1:
        lower_rank = 1
    elif lower_rank >= count:
        lower_rank = count - 1
    fraction = position - lower_rank
    lower = sorted_values[lower_rank - 1]
    upper = sorted_values[lower_rank]
    if fraction == 0:
        return lower
    if fraction == 1:
        return upper
    return (1 - fraction) * lower + fraction * upper


def calculate_max(sorted_values: Sequence[float]) -> float:
    return sorted_values[-1]


def calculate_iqr(q1: float | None, q3: float | None) -> float:
    if q1 is INVALID and q3 is INVALID:
        raise ValueError("depends on invalid 25% and 75%")
    if q1 is INVALID:
        raise ValueError("depends on invalid 25%")
    if q3 is INVALID:
        raise ValueError("depends on invalid 75%")
    return q3 - q1


def safe_calculate(
    feature_name: str,
    statistic_name: str,
    errors: list[str],
    calculation: Callable[..., float],
    *args: object,
) -> float | None:
    try:
        result = calculation(*args)
        if not math.isfinite(result):
            raise ValueError("result is not finite")
        return result
    except (ArithmeticError, ValueError, TypeError, IndexError) as error:
        errors.append(f"{feature_name}: {statistic_name}: {error}")
        return INVALID


def calculate_feature_statistics(
    feature_name: str,
    sorted_values: Sequence[float],
    total_rows: int,
    errors: list[str],
) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    calculation: Callable[..., float]
    args: tuple[object, ...]
    # Calculations run in display order so dependencies are already available.
    for name in STATISTIC_NAMES:
        match name:
            case "Count":
                calculation, args = calculate_count, (sorted_values,)
            case "Missing Count":
                calculation = calculate_missing_count
                args = (total_rows, results["Count"])
            case "Mean":
                calculation, args = calculate_mean, (sorted_values, results["Count"])
            case "Std":
                calculation = calculate_std
                args = (sorted_values, results["Mean"], results["Count"])
            case "Min":
                calculation, args = calculate_min, (sorted_values,)
            case "25%":
                calculation, args = calculate_quartile, (sorted_values, 0.25)
            case "50%":
                calculation, args = calculate_quartile, (sorted_values, 0.50)
            case "75%":
                calculation, args = calculate_quartile, (sorted_values, 0.75)
            case "Max":
                calculation, args = calculate_max, (sorted_values,)
            case _:
                calculation, args = calculate_iqr, (results["25%"], results["75%"])
        results[name] = safe_calculate(
            feature_name, name, errors, calculation, *args
        )
    return results


def display_results(
    feature_results: Iterable[tuple[str, Mapping[str, float | None]]],
) -> None:
    label_width = max(len(name) for name in STATISTIC_NAMES)
    columns = []
    for feature_name, results in feature_results:
        formatted = []
        for name in STATISTIC_NAMES:
            value = results[name]
            formatted.append("invalid" if value is INVALID else f"{value:.6f}")
        width = max(len(feature_name), max(len(value) for value in formatted))
        columns.append((feature_name, formatted, width))
    header = [" " * label_width]
    for feature_name, _, width in columns:
        header.append(f"{feature_name:>{width}}")
    print("  ".join(header))
    for index, name in enumerate(STATISTIC_NAMES):
        row = [f"{name:<{label_width}}"]
        for _, formatted, width in columns:
            row.append(f"{formatted[index]:>{width}}")
        print("  ".join(row))


def display_errors(errors: Sequence[str]) -> None:
    if errors:
        print("\nErrors:")
        for error in errors:
            print(error)


def main() -> int:
    try:
        file_path = validate_arguments()
        headers, rows = read_csv(file_path)
        features = identify_numerical_features(headers, rows)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    total_rows = len(rows)
    errors: list[str] = []
    feature_results = []
    for column_index, feature_name in features:
        sorted_values = extract_values(rows, column_index)
        results = calculate_feature_statistics(
            feature_name, sorted_values, total_rows, errors
        )
        feature_results.append((feature_name, results))
    display_results(feature_results)
    display_errors(errors)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
