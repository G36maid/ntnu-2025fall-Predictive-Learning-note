import csv
from collections import Counter


def read_data(year):
    data = []
    with open(f"obesity_election_{year}.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            state, obesity, party, _ = row
            obesity_rate = float(obesity.strip("%"))
            data.append((obesity_rate, 1 if party == "R" else 0))  # 1 for R, 0 for D
    return data


def euclidean_distance(point1, point2):
    return abs(point1[0] - point2[0])


def knn_predict(train_data, test_point, k):
    distances = []
    for train_point in train_data:
        distance = euclidean_distance(test_point, train_point)
        distances.append((train_point, distance))

    distances.sort(key=lambda x: x[1])

    neighbors = [item[0] for item in distances[:k]]

    output_values = [neighbor[1] for neighbor in neighbors]
    prediction = Counter(output_values).most_common(1)[0][0]
    return prediction


def leave_one_out_cv(data):
    k_values = range(1, len(data))
    best_k = -1
    min_error = float("inf")

    for k in k_values:
        errors = 0
        for i in range(len(data)):
            train_data = data[:i] + data[i + 1 :]
            test_point = data[i]
            prediction = knn_predict(train_data, test_point, k)
            if prediction != test_point[1]:
                errors += 1

        error_rate = errors / len(data)
        if error_rate < min_error:
            min_error = error_rate
            best_k = k

    return best_k, min_error


if __name__ == "__main__":
    train_data_2004 = read_data(2004)
    test_data_2000 = read_data(2000)

    optimal_k, resampling_error = leave_one_out_cv(train_data_2004)

    test_errors = 0
    for test_point in test_data_2000:
        prediction = knn_predict(train_data_2004, test_point, optimal_k)
        if prediction != test_point[1]:
            test_errors += 1

    test_error_rate = test_errors / len(test_data_2000)

    print(f"Optimal k: {optimal_k}")
    print(f"Resampling error (Leave-one-out CV): {resampling_error:.4f}")
    print(f"Test error: {test_error_rate:.4f}")
