import numpy as np
from sklearn.linear_model import LinearRegression


def generate_data(n=10):
    np.random.seed(0)
    x = np.random.rand(n, 1)
    noise = np.random.normal(0, 0.5, (n, 1))
    y = x**2 + 0.1 * x + noise
    return x, y


def schwartz_criterion(n, mse, d):
    return n * np.log(mse) + d * np.log(n)


def trigonometric_polynomial_estimator(x, y):
    best_sc = float("inf")
    best_m = -1

    for m in range(1, 10):
        X = np.ones((len(x), m + 1))
        for i in range(1, m + 1):
            X[:, i] = np.cos(2 * np.pi * i * x).ravel()

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)

        sc = schwartz_criterion(len(x), mse, m + 1)

        if sc < best_sc:
            best_sc = sc
            best_m = m

    return best_m, best_sc


def algebraic_polynomial_estimator(x, y):
    best_sc = float("inf")
    best_m = -1

    for m in range(1, 10):
        X = np.ones((len(x), m + 1))
        for i in range(1, m + 1):
            X[:, i] = x.ravel() ** i

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)

        sc = schwartz_criterion(len(x), mse, m + 1)

        if sc < best_sc:
            best_sc = sc
            best_m = m

    return best_m, best_sc


if __name__ == "__main__":
    x, y = generate_data()

    trig_m, trig_sc = trigonometric_polynomial_estimator(x, y)
    alg_m, alg_sc = algebraic_polynomial_estimator(x, y)

    print("Trigonometric Polynomial Estimator:")
    print(f"  Optimal m: {trig_m}")
    print(f"  Schwartz Criterion: {trig_sc:.4f}")

    print("\nAlgebraic Polynomial Estimator:")
    print(f"  Optimal m: {alg_m}")
    print(f"  Schwartz Criterion: {alg_sc:.4f}")
