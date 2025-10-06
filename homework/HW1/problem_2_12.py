import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


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

    return best_m


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

    return best_m


if __name__ == "__main__":
    x, y = generate_data()
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    trig_errors = []
    alg_errors = []

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Trigonometric
        trig_m = trigonometric_polynomial_estimator(x_train, y_train)
        X_train_trig = np.ones((len(x_train), trig_m + 1))
        X_val_trig = np.ones((len(x_val), trig_m + 1))
        for i in range(1, trig_m + 1):
            X_train_trig[:, i] = np.cos(2 * np.pi * i * x_train).ravel()
            X_val_trig[:, i] = np.cos(2 * np.pi * i * x_val).ravel()

        trig_model = LinearRegression()
        trig_model.fit(X_train_trig, y_train)
        trig_pred = trig_model.predict(X_val_trig)
        trig_errors.append(np.mean((y_val - trig_pred) ** 2))

        # Algebraic
        alg_m = algebraic_polynomial_estimator(x_train, y_train)
        X_train_alg = np.ones((len(x_train), alg_m + 1))
        X_val_alg = np.ones((len(x_val), alg_m + 1))
        for i in range(1, alg_m + 1):
            X_train_alg[:, i] = x_train.ravel() ** i
            X_val_alg[:, i] = x_val.ravel() ** i

        alg_model = LinearRegression()
        alg_model.fit(X_train_alg, y_train)
        alg_pred = alg_model.predict(X_val_alg)
        alg_errors.append(np.mean((y_val - alg_pred) ** 2))

    print("5-Fold Cross-Validation Results:")
    print(
        f"  Trigonometric Polynomial Estimator Average Error: {np.mean(trig_errors):.4f}"
    )
    print(f"  Algebraic Polynomial Estimator Average Error: {np.mean(alg_errors):.4f}")
