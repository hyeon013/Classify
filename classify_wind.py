import csv
import math
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path


DATA_FILE = Path(__file__).with_name("wind_dataset.csv")
OUTPUT_FILE = Path(__file__).with_name("classification_predictions.csv")
SCORES_FILE = Path(__file__).with_name("classification_scores.csv")
THRESHOLD_MS = 4.0
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TRAIN_START = datetime(2023, 1, 1)
EVALUATION_START = datetime(2024, 12, 1)
EVALUATION_END = datetime(2025, 1, 1)
FEATURE_COLUMNS = (
    "wind_speed_ms",
    "temperature_C",
    "humidity_pct",
    "wind_speed_ms_1h_before",
    "wind_speed_ms_2h_before",
)
PREDICTION_COLUMNS = (
    "forecast_time",
    "target_time",
    "actual_wind_speed_ms",
    "actual_label",
    "wind_speed_ms",
    "always_zero_prediction",
    "current_wind_threshold_prediction",
    "strong_wind_probability",
    "logistic_regression_prediction",
    "logistic_regression_prediction_0_20",
)
LOGISTIC_THRESHOLD = 0.50
LOWER_LOGISTIC_THRESHOLD = 0.20
METHODS = (
    ("항상 0", "always_zero_prediction"),
    ("현재 풍속 4.0 m/s 기준", "current_wind_threshold_prediction"),
    ("로지스틱 회귀 (0.50)", "logistic_regression_prediction"),
    ("로지스틱 회귀 (0.20)", "logistic_regression_prediction_0_20"),
)


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def solve_linear_system(matrix, vector):
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ArithmeticError("로지스틱 회귀의 헤세 행렬을 풀 수 없습니다.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor

        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]

    return [augmented[row][size] for row in range(size)]


def fit_logistic_regression(training_features, training_labels):
    feature_count = len(training_features[0])
    means = [
        statistics.fmean(row[index] for row in training_features)
        for index in range(feature_count)
    ]
    scales = [
        statistics.pstdev(row[index] for row in training_features)
        for index in range(feature_count)
    ]
    scales = [scale if scale else 1.0 for scale in scales]
    standardized = [
        [1.0] + [(value - means[index]) / scales[index] for index, value in enumerate(row)]
        for row in training_features
    ]
    sample_count = len(standardized)
    coefficients = [0.0] * (feature_count + 1)
    regularization = 1.0 / sample_count

    for _ in range(100):
        gradient = [0.0] * len(coefficients)
        hessian = [[0.0] * len(coefficients) for _ in coefficients]

        for features, label in zip(standardized, training_labels):
            probability = sigmoid(sum(weight * value for weight, value in zip(coefficients, features)))
            residual = probability - label
            curvature = probability * (1 - probability)
            for row in range(len(coefficients)):
                gradient[row] += residual * features[row]
                for column in range(len(coefficients)):
                    hessian[row][column] += curvature * features[row] * features[column]

        for index in range(len(coefficients)):
            gradient[index] /= sample_count
            hessian[index] = [value / sample_count for value in hessian[index]]
        for index in range(1, len(coefficients)):
            gradient[index] += regularization * coefficients[index]
            hessian[index][index] += regularization

        adjustment = solve_linear_system(hessian, gradient)
        coefficients = [
            coefficient - change
            for coefficient, change in zip(coefficients, adjustment)
        ]
        if max(abs(change) for change in adjustment) < 1e-8:
            break
    else:
        raise ArithmeticError("로지스틱 회귀가 100회 반복 후에도 수렴하지 않았습니다.")

    return means, scales, coefficients


def predict_probability(features, means, scales, coefficients):
    standardized = [
        (value - means[index]) / scales[index]
        for index, value in enumerate(features)
    ]
    logit = coefficients[0] + sum(
        weight * value for weight, value in zip(coefficients[1:], standardized)
    )
    return sigmoid(logit)


def score_predictions(rows, prediction_column):
    counts = {"hit": 0, "miss": 0, "false_alarm": 0, "correct_rejection": 0}
    for row in rows:
        actual = int(row["actual_label"])
        predicted = int(row[prediction_column])
        if actual and predicted:
            counts["hit"] += 1
        elif actual and not predicted:
            counts["miss"] += 1
        elif predicted:
            counts["false_alarm"] += 1
        else:
            counts["correct_rejection"] += 1

    total = len(rows)
    predicted_positive = counts["hit"] + counts["false_alarm"]
    actual_positive = counts["hit"] + counts["miss"]
    return {
        **counts,
        "rows": total,
        "accuracy": (counts["hit"] + counts["correct_rejection"]) / total,
        "precision": counts["hit"] / predicted_positive if predicted_positive else None,
        "recall": counts["hit"] / actual_positive if actual_positive else None,
        "predicted_positive": predicted_positive,
    }


def main():
    periods = {
        "학습": {"rows": 0, "zeros": 0, "ones": 0},
        "평가": {"rows": 0, "zeros": 0, "ones": 0},
    }
    training_features = []
    training_labels = []
    evaluation_rows = []
    previous_target_time = None
    row_count = 0

    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as dataset:
        reader = csv.DictReader(dataset)
        required_columns = {
            "forecast_time",
            "target_time",
            "actual_wind_speed_ms",
            *FEATURE_COLUMNS,
        }
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"CSV에 필요한 열이 없습니다: {', '.join(sorted(missing_columns))}")

        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            forecast_time = datetime.strptime(row["forecast_time"], TIME_FORMAT)
            target_time = datetime.strptime(row["target_time"], TIME_FORMAT)
            if target_time - forecast_time != timedelta(hours=1):
                raise ValueError(
                    f"{row_number}행의 forecast_time과 target_time이 정확히 1시간 차이가 아닙니다."
                )
            if previous_target_time is not None and target_time <= previous_target_time:
                raise ValueError(
                    f"{row_number}행의 target_time이 원본의 시간순으로 증가하지 않습니다."
                )
            previous_target_time = target_time

            if TRAIN_START <= target_time < EVALUATION_START:
                period = periods["학습"]
            elif EVALUATION_START <= target_time < EVALUATION_END:
                period = periods["평가"]
            else:
                continue

            try:
                actual_wind_speed = float(row["actual_wind_speed_ms"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{row_number}행의 actual_wind_speed_ms 값이 숫자가 아닙니다."
                ) from error
            try:
                features = [float(row[column]) for column in FEATURE_COLUMNS]
            except (TypeError, ValueError) as error:
                raise ValueError(f"{row_number}행의 입력 특성 값이 숫자가 아닙니다.") from error

            label = int(actual_wind_speed >= THRESHOLD_MS)
            period["rows"] += 1
            period["ones" if label else "zeros"] += 1

            if period is periods["학습"]:
                training_features.append(features)
                training_labels.append(label)
            else:
                evaluation_rows.append(
                    {
                        "forecast_time": row["forecast_time"],
                        "target_time": row["target_time"],
                        "actual_wind_speed_ms": actual_wind_speed,
                        "actual_label": label,
                        "wind_speed_ms": features[0],
                        "always_zero_prediction": 0,
                        "current_wind_threshold_prediction": int(
                            features[0] >= THRESHOLD_MS
                        ),
                        "_features": features,
                    }
                )

    if not training_features:
        raise ValueError("학습 구간에 사용할 행이 없습니다.")
    if not evaluation_rows:
        raise ValueError("2024년 12월 평가 구간에 사용할 행이 없습니다.")

    means, scales, coefficients = fit_logistic_regression(
        training_features, training_labels
    )

    for row in evaluation_rows:
        features = row.pop("_features")
        probability = predict_probability(features, means, scales, coefficients)
        row["strong_wind_probability"] = probability
        row["logistic_regression_prediction"] = int(
            probability >= LOGISTIC_THRESHOLD
        )
        row["logistic_regression_prediction_0_20"] = int(
            probability >= LOWER_LOGISTIC_THRESHOLD
        )

    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as predictions_file:
        writer = csv.DictWriter(predictions_file, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(evaluation_rows)

    actual_strong_wind_count = sum(row["actual_label"] for row in evaluation_rows)
    score_rows = []
    for method, prediction_column in METHODS:
        scores = score_predictions(evaluation_rows, prediction_column)
        score_rows.append({"method": method, **scores})

    score_columns = (
        "method",
        "rows",
        "hit",
        "miss",
        "false_alarm",
        "correct_rejection",
        "accuracy",
        "precision",
        "recall",
    )
    with SCORES_FILE.open("w", encoding="utf-8-sig", newline="") as scores_file:
        writer = csv.DictWriter(scores_file, fieldnames=score_columns)
        writer.writeheader()
        for row in score_rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "rows": row["rows"],
                    "hit": row["hit"],
                    "miss": row["miss"],
                    "false_alarm": row["false_alarm"],
                    "correct_rejection": row["correct_rejection"],
                    "accuracy": f"{row['accuracy']:.6f}",
                    "precision": (
                        f"{row['precision']:.6f}"
                        if row["precision"] is not None
                        else "해당 없음"
                    ),
                    "recall": f"{row['recall']:.6f}" if row["recall"] is not None else "해당 없음",
                }
            )

    print(f"시간 검증: 전체 {row_count:,}행, 모든 forecast_time과 target_time 차이가 정확히 1시간")
    print("시간순 검증: target_time이 원본 순서대로 증가")
    for name in ("학습", "평가"):
        counts = periods[name]
        print(
            f"{name}: 전체 {counts['rows']:,}행, "
            f"0={counts['zeros']:,}개, 1={counts['ones']:,}개"
        )

    print(f"예측 저장: {OUTPUT_FILE} ({len(evaluation_rows):,}행)")
    print(
        f"비교 평가 행: 세 방법 모두 동일한 {len(evaluation_rows):,}행 "
        f"(고유 정답 시각 {len({row['target_time'] for row in evaluation_rows}):,}개)"
    )
    print(f"실제 강풍 행: {actual_strong_wind_count:,}개")
    print(f"비교 결과 저장: {SCORES_FILE}")
    print("로지스틱 회귀 확률 기준 비교 (동일 모델·동일 평가 행)")
    print("기준 | 적중 | 놓침 | 오경보 | 정밀도 | 재현율")
    for row in score_rows[2:]:
        precision = (
            f"{row['precision']:.2%}"
            if row["precision"] is not None
            else "해당 없음"
        )
        print(
            f"{row['method']} | {row['hit']} | {row['miss']} | "
            f"{row['false_alarm']} | {precision} | {row['recall']:.2%}"
        )
    print("전체 방법 비교")
    print(
        "방법 | 적중 | 놓침 | 오경보 | 정상판정 | 정확도 | 강풍 정밀도 | 강풍 재현율"
    )
    for row in score_rows:
        precision = (
            f"{row['precision']:.2%}"
            if row["precision"] is not None
            else "해당 없음"
        )
        print(
            f"{row['method']} | {row['hit']} | {row['miss']} | "
            f"{row['false_alarm']} | {row['correct_rejection']} | "
            f"{row['accuracy']:.2%} | {precision} | {row['recall']:.2%}"
        )
    print("예측 CSV 미리보기 (헤더 + 첫 5행):")
    with OUTPUT_FILE.open("r", encoding="utf-8-sig", newline="") as predictions_file:
        preview = []
        for _, row in zip(range(6), csv.reader(predictions_file)):
            preview.append(row)
    csv.writer(sys.stdout, lineterminator="\n").writerows(preview)


if __name__ == "__main__":
    main()
