import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def read_data(year):
    """Read election and obesity data for a given year"""
    data = []
    with open(f"obesity_election_{year}.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            state, obesity, party, _ = row
            obesity_rate = float(obesity.strip("%"))
            data.append(
                (obesity_rate, 1 if party == "R" else 0, state)
            )  # Added state name
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


def leave_one_out_cv_detailed(data):
    """Return detailed CV results for all k values"""
    k_values = range(1, len(data))
    cv_results = {}

    for k in k_values:
        errors = 0
        for i in range(len(data)):
            train_data = data[:i] + data[i + 1 :]
            test_point = data[i]
            prediction = knn_predict(train_data, test_point, k)
            if prediction != test_point[1]:
                errors += 1

        error_rate = errors / len(data)
        cv_results[k] = error_rate

    best_k = min(cv_results, key=cv_results.get)
    return cv_results, best_k


def generate_data(n=10):
    np.random.seed(0)
    x = np.random.rand(n, 1)
    noise = np.random.normal(0, 0.5, (n, 1))
    y = x**2 + 0.1 * x + noise
    return x, y


def schwartz_criterion(n, mse, d):
    return n * np.log(mse) + d * np.log(n)


def detailed_polynomial_analysis(x, y, estimator_type="trigonometric"):
    """Return detailed analysis for polynomial estimators"""
    results = {}

    for m in range(1, 10):
        X = np.ones((len(x), m + 1))

        if estimator_type == "trigonometric":
            for i in range(1, m + 1):
                X[:, i] = np.cos(2 * np.pi * i * x).ravel()
        else:  # algebraic
            for i in range(1, m + 1):
                X[:, i] = x.ravel() ** i

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        sc = schwartz_criterion(len(x), mse, m + 1)

        results[m] = {"mse": mse, "sc": sc, "model": model, "X": X}

    best_m = min(results, key=lambda k: results[k]["sc"])
    return results, best_m


def detailed_cv_analysis(x, y):
    """Perform detailed 5-fold CV analysis"""
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_results = []

    for fold_idx, (train_index, val_index) in enumerate(kf.split(x)):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Trigonometric
        trig_results, trig_m = detailed_polynomial_analysis(
            x_train, y_train, "trigonometric"
        )
        X_val_trig = np.ones((len(x_val), trig_m + 1))
        for i in range(1, trig_m + 1):
            X_val_trig[:, i] = np.cos(2 * np.pi * i * x_val).ravel()

        trig_pred = trig_results[trig_m]["model"].predict(X_val_trig)
        trig_error = np.mean((y_val - trig_pred) ** 2)

        # Algebraic
        alg_results, alg_m = detailed_polynomial_analysis(x_train, y_train, "algebraic")
        X_val_alg = np.ones((len(x_val), alg_m + 1))
        for i in range(1, alg_m + 1):
            X_val_alg[:, i] = x_val.ravel() ** i

        alg_pred = alg_results[alg_m]["model"].predict(X_val_alg)
        alg_error = np.mean((y_val - alg_pred) ** 2)

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "trig_error": trig_error,
                "alg_error": alg_error,
                "trig_m": trig_m,
                "alg_m": alg_m,
            }
        )

    return fold_results


def create_knn_plots():
    """Create plots for k-NN problems"""
    # Load data
    data_2000 = read_data(2000)
    data_2004 = read_data(2004)

    # Extract obesity rates and labels
    obesity_2000 = [d[0] for d in data_2000]
    labels_2000 = [d[1] for d in data_2000]
    obesity_2004 = [d[0] for d in data_2004]
    labels_2004 = [d[1] for d in data_2004]

    # Create scatter plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 2000 data
    republicans_2000 = [
        obesity_2000[i] for i in range(len(obesity_2000)) if labels_2000[i] == 1
    ]
    democrats_2000 = [
        obesity_2000[i] for i in range(len(obesity_2000)) if labels_2000[i] == 0
    ]

    ax1.scatter(
        republicans_2000,
        [1] * len(republicans_2000),
        color="red",
        alpha=0.7,
        label="Republican",
        s=50,
    )
    ax1.scatter(
        democrats_2000,
        [0] * len(democrats_2000),
        color="blue",
        alpha=0.7,
        label="Democrat",
        s=50,
    )
    ax1.set_xlabel("Obesity Rate (%)")
    ax1.set_ylabel("Election Outcome")
    ax1.set_title("2000 Election Results vs Obesity Rate")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Democrat", "Republican"])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2004 data
    republicans_2004 = [
        obesity_2004[i] for i in range(len(obesity_2004)) if labels_2004[i] == 1
    ]
    democrats_2004 = [
        obesity_2004[i] for i in range(len(obesity_2004)) if labels_2004[i] == 0
    ]

    ax2.scatter(
        republicans_2004,
        [1] * len(republicans_2004),
        color="red",
        alpha=0.7,
        label="Republican",
        s=50,
    )
    ax2.scatter(
        democrats_2004,
        [0] * len(democrats_2004),
        color="blue",
        alpha=0.7,
        label="Democrat",
        s=50,
    )
    ax2.set_xlabel("Obesity Rate (%)")
    ax2.set_ylabel("Election Outcome")
    ax2.set_title("2004 Election Results vs Obesity Rate")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Democrat", "Republican"])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("knn_scatter_plots.png", dpi=300, bbox_inches="tight")
    plt.show()


def create_polynomial_plots():
    """Create plots for polynomial regression problems"""
    x, y = generate_data()

    # Get detailed results
    trig_results, best_trig_m = detailed_polynomial_analysis(x, y, "trigonometric")
    alg_results, best_alg_m = detailed_polynomial_analysis(x, y, "algebraic")

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Data and fitted models
    x_plot = np.linspace(0, 1, 100).reshape(-1, 1)

    # Trigonometric fit
    X_trig = np.ones((len(x_plot), best_trig_m + 1))
    for i in range(1, best_trig_m + 1):
        X_trig[:, i] = np.cos(2 * np.pi * i * x_plot).ravel()
    y_trig = trig_results[best_trig_m]["model"].predict(X_trig)

    ax1.scatter(x, y, color="black", alpha=0.7, label="Data points")
    ax1.plot(x_plot, y_trig, "r-", label=f"Trigonometric (m={best_trig_m})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Trigonometric Polynomial Fit")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Algebraic fit
    X_alg = np.ones((len(x_plot), best_alg_m + 1))
    for i in range(1, best_alg_m + 1):
        X_alg[:, i] = x_plot.ravel() ** i
    y_alg = alg_results[best_alg_m]["model"].predict(X_alg)

    ax2.scatter(x, y, color="black", alpha=0.7, label="Data points")
    ax2.plot(x_plot, y_alg, "b-", label=f"Algebraic (m={best_alg_m})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Algebraic Polynomial Fit")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Schwartz Criterion vs complexity
    m_values = list(range(1, 10))
    trig_sc_values = [trig_results[m]["sc"] for m in m_values]
    alg_sc_values = [alg_results[m]["sc"] for m in m_values]

    ax3.plot(m_values, trig_sc_values, "ro-", label="Trigonometric")
    ax3.plot(m_values, alg_sc_values, "bo-", label="Algebraic")
    ax3.axvline(x=best_trig_m, color="red", linestyle="--", alpha=0.7)
    ax3.axvline(x=best_alg_m, color="blue", linestyle="--", alpha=0.7)
    ax3.set_xlabel("Model Complexity (m)")
    ax3.set_ylabel("Schwartz Criterion")
    ax3.set_title("Model Selection via Schwartz Criterion")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: MSE vs complexity
    trig_mse_values = [trig_results[m]["mse"] for m in m_values]
    alg_mse_values = [alg_results[m]["mse"] for m in m_values]

    ax4.plot(m_values, trig_mse_values, "ro-", label="Trigonometric")
    ax4.plot(m_values, alg_mse_values, "bo-", label="Algebraic")
    ax4.set_xlabel("Model Complexity (m)")
    ax4.set_ylabel("Mean Squared Error")
    ax4.set_title("Training MSE vs Model Complexity")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("polynomial_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def generate_all_tables_and_figures():
    """Generate all tables and figures for the homework"""

    print("=" * 80)
    print("HOMEWORK 1 - ENHANCED RESULTS WITH TABLES AND FIGURES")
    print("=" * 80)

    # Problem 2.7 & 2.8 Analysis
    print("\n" + "=" * 50)
    print("PROBLEMS 2.7 & 2.8: k-NN CLASSIFICATION")
    print("=" * 50)

    # Load data
    data_2000 = read_data(2000)
    data_2004 = read_data(2004)

    # Problem 2.7: Train on 2004, Test on 2000
    cv_results_27, best_k_27 = leave_one_out_cv_detailed(data_2004)
    test_errors_27 = 0
    for test_point in data_2000:
        prediction = knn_predict(data_2004, test_point, best_k_27)
        if prediction != test_point[1]:
            test_errors_27 += 1
    test_error_27 = test_errors_27 / len(data_2000)

    # Problem 2.8: Train on 2000, Test on 2004
    cv_results_28, best_k_28 = leave_one_out_cv_detailed(data_2000)
    test_errors_28 = 0
    for test_point in data_2004:
        prediction = knn_predict(data_2000, test_point, best_k_28)
        if prediction != test_point[1]:
            test_errors_28 += 1
    test_error_28 = test_errors_28 / len(data_2004)

    # Table 1: Cross-validation Results Summary
    print("\nTable 1: Cross-Validation Results for Different k Values")
    print("-" * 60)
    print(f"{'k':<3} {'Problem 2.7 (2004→2000)':<25} {'Problem 2.8 (2000→2004)':<25}")
    print("-" * 60)
    for k in range(1, min(16, len(data_2000))):  # Show first 15 k values
        cv_27 = cv_results_27.get(k, "N/A")
        cv_28 = cv_results_28.get(k, "N/A")
        marker_27 = " *" if k == best_k_27 else "  "
        marker_28 = " *" if k == best_k_28 else "  "
        if cv_27 != "N/A":
            print(f"{k:<3} {cv_27:<23.4f}{marker_27} {cv_28:<23.4f}{marker_28}")
    print("-" * 60)
    print("* indicates optimal k value")

    # Table 2: Summary Comparison
    print("\nTable 2: Summary Comparison")
    print("-" * 55)
    print(f"{'Metric':<20} {'Problem 2.7':<15} {'Problem 2.8':<15}")
    print("-" * 55)
    print(f"{'Training Data':<20} {'2004':<15} {'2000':<15}")
    print(f"{'Test Data':<20} {'2000':<15} {'2004':<15}")
    print(f"{'Optimal k':<20} {best_k_27:<15} {best_k_28:<15}")
    print(
        f"{'Resampling Error':<20} {cv_results_27[best_k_27]:<15.4f} {cv_results_28[best_k_28]:<15.4f}"
    )
    print(f"{'Test Error':<20} {test_error_27:<15.4f} {test_error_28:<15.4f}")
    print("-" * 55)

    # Problems 2.11 & 2.12 Analysis
    print("\n" + "=" * 50)
    print("PROBLEMS 2.11 & 2.12: POLYNOMIAL REGRESSION")
    print("=" * 50)

    x, y = generate_data()

    # Problem 2.11: Model Selection
    trig_results, best_trig_m = detailed_polynomial_analysis(x, y, "trigonometric")
    alg_results, best_alg_m = detailed_polynomial_analysis(x, y, "algebraic")

    # Table 3: Model Selection Results (Problem 2.11)
    print("\nTable 3: Model Selection Results (Problem 2.11)")
    print("-" * 70)
    print(f"{'Complexity (m)':<15} {'Trigonometric SC':<20} {'Algebraic SC':<20}")
    print("-" * 70)
    for m in range(1, 10):
        trig_sc = trig_results[m]["sc"]
        alg_sc = alg_results[m]["sc"]
        trig_marker = " *" if m == best_trig_m else "  "
        alg_marker = " *" if m == best_alg_m else "  "
        print(f"{m:<15} {trig_sc:<18.4f}{trig_marker} {alg_sc:<18.4f}{alg_marker}")
    print("-" * 70)
    print("* indicates optimal complexity")

    # Problem 2.12: Cross-validation
    fold_results = detailed_cv_analysis(x, y)

    # Table 4: Cross-validation Results (Problem 2.12)
    print("\nTable 4: 5-Fold Cross-Validation Results (Problem 2.12)")
    print("-" * 65)
    print(
        f"{'Fold':<6} {'Trig Error':<15} {'Alg Error':<15} {'Trig m':<10} {'Alg m':<10}"
    )
    print("-" * 65)
    for result in fold_results:
        print(
            f"{result['fold']:<6} {result['trig_error']:<15.4f} {result['alg_error']:<15.4f} "
            f"{result['trig_m']:<10} {result['alg_m']:<10}"
        )

    avg_trig_error = np.mean([r["trig_error"] for r in fold_results])
    avg_alg_error = np.mean([r["alg_error"] for r in fold_results])

    print("-" * 65)
    print(f"{'Average':<6} {avg_trig_error:<15.4f} {avg_alg_error:<15.4f}")
    print("-" * 65)

    # Table 5: Final Comparison Summary
    print("\nTable 5: Final Method Comparison")
    print("-" * 60)
    print(f"{'Method':<25} {'Problem 2.11 (SC)':<20} {'Problem 2.12 (CV)':<15}")
    print("-" * 60)
    print(
        f"{'Trigonometric':<25} {trig_results[best_trig_m]['sc']:<20.4f} {avg_trig_error:<15.4f}"
    )
    print(
        f"{'Algebraic':<25} {alg_results[best_alg_m]['sc']:<20.4f} {avg_alg_error:<15.4f}"
    )
    print("-" * 60)
    print("SC = Schwartz Criterion, CV = Cross-Validation Error")

    # Generate plots
    print("\n" + "=" * 50)
    print("GENERATING FIGURES...")
    print("=" * 50)

    create_knn_plots()
    create_polynomial_plots()

    print("\nFigures saved:")
    print("- knn_scatter_plots.png: Election results vs obesity rates")
    print("- polynomial_analysis.png: Polynomial regression analysis")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_tables_and_figures()
