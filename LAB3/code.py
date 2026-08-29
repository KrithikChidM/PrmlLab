import math
import matplotlib.pyplot as plt

filename = "noisy_7.txt"

def read_data(filename):
    data = []

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) >= 2:
                x = float(parts[0])
                y = float(parts[1])
                data.append((x, y))

    return data

def shuffle_data(data, seed=42):
    data = data[:]

    for i in range(len(data) - 1, 0, -1):
        seed = (seed * 1103515245 + 12345) % 2147483648
        j = seed % (i + 1)

        data[i], data[j] = data[j], data[i]

    return data

def split_data(data):
    data = shuffle_data(data)

    n = len(data)

    train_end = int(0.60 * n)
    test_end = int(0.80 * n)

    train = data[:train_end]
    test = data[train_end:test_end]
    validation = data[test_end:]

    return train, test, validation

def create_matrix(data, degree):
    size = degree + 1

    A = [[0.0 for j in range(size)] for i in range(size)]
    B = [0.0 for i in range(size)]

    for x, y in data:
        powers = [1.0]

        for i in range(1, 2 * degree + 1):
            powers.append(powers[-1] * x)

        for i in range(size):
            for j in range(size):
                A[i][j] += powers[i + j]

            B[i] += y * powers[i]

    return A, B

def gaussian_elimination(A, B):
    n = len(B)

    for i in range(n):
        max_row = i

        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[max_row][i]):
                max_row = j

        if abs(A[max_row][i]) < 1e-12:
            return None

        A[i], A[max_row] = A[max_row], A[i]
        B[i], B[max_row] = B[max_row], B[i]

        pivot = A[i][i]

        for j in range(i, n):
            A[i][j] /= pivot

        B[i] /= pivot

        for j in range(i + 1, n):
            factor = A[j][i]

            for k in range(i, n):
                A[j][k] -= factor * A[i][k]

            B[j] -= factor * B[i]

    coefficients = [0.0] * n

    for i in range(n - 1, -1, -1):
        coefficients[i] = B[i]

        for j in range(i + 1, n):
            coefficients[i] -= A[i][j] * coefficients[j]

    return coefficients

def train_polynomial(data, degree):
    A, B = create_matrix(data, degree)

    return gaussian_elimination(A, B)

def predict(x, coefficients):
    result = 0.0
    power = 1.0

    for coefficient in coefficients:
        result += coefficient * power
        power *= x

    return result

def mse(data, coefficients):
    error = 0.0

    for x, y in data:
        prediction = predict(x, coefficients)
        error += (y - prediction) ** 2

    return error / len(data)

def rmse(data, coefficients):
    return math.sqrt(mse(data, coefficients))

def r2_score(data, coefficients):
    mean_y = 0.0

    for x, y in data:
        mean_y += y

    mean_y /= len(data)

    total_error = 0.0
    residual_error = 0.0

    for x, y in data:
        prediction = predict(x, coefficients)

        total_error += (y - mean_y) ** 2
        residual_error += (y - prediction) ** 2

    if total_error == 0:
        return 0.0

    return 1.0 - residual_error / total_error

def polynomial_string(coefficients):
    expression = ""
    for i, coefficient in enumerate(coefficients):
        if i == 0:
            expression = "%.6f" % coefficient
        elif coefficient >= 0:
            expression += " + %.6f*x^%d" % (coefficient, i)
        else:
            expression += " - %.6f*x^%d" % (abs(coefficient), i)
    return expression

data = read_data(filename)

if len(data) == 0:
    print("No data found.")
else:

    train, test, validation = split_data(data)

    degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    results = []

    best_degree = None
    best_test_mse = float("inf")
    best_coefficients = None

    print()
    print("Polynomial Regression Experiments")
    print("----------------------------------")
    print(
        "%-8s %-15s %-15s %-15s"
        % ("Degree", "Train MSE", "Test MSE", "Validation MSE")
    )

    for degree in degrees:

        coefficients = train_polynomial(train, degree)

        if coefficients is None:
            print("%-8d Singular matrix" % degree)
            continue

        train_mse = mse(train, coefficients)
        test_mse = mse(test, coefficients)
        validation_mse = mse(validation, coefficients)

        results.append(
            (
                degree,
                train_mse,
                test_mse,
                validation_mse
            )
        )

        print(
            "%-8d %-15.6f %-15.6f %-15.6f"
            % (
                degree,
                train_mse,
                test_mse,
                validation_mse
            )
        )

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_degree = degree
            best_coefficients = coefficients

    print()
    print("Best Polynomial Model: ",best_degree)

    print()
    print("Polynomial:")
    print(polynomial_string(best_coefficients))

    print()
    print("Final Validation Results")
    print("------------------------")

    validation_mse = mse(validation, best_coefficients)
    validation_rmse = rmse(validation, best_coefficients)
    validation_r2 = r2_score(validation, best_coefficients)

    print("Validation MSE  :", validation_mse)
    print("Validation RMSE :", validation_rmse)
    print("Validation R2   :", validation_r2)

x_values = [x for x, y in data]
y_values = [y for x, y in data]

x_min = min(x_values)
x_max = max(x_values)

curve_x = []
curve_y = []

steps = 500

for i in range(steps + 1):
    x = x_min + (x_max - x_min) * i / steps
    curve_x.append(x)
    curve_y.append(predict(x, best_coefficients))

plt.figure(figsize=(10, 6))

plt.scatter(
    [x for x, y in train],
    [y for x, y in train],
    label="Training Data"
)

plt.scatter(
    [x for x, y in test],
    [y for x, y in test],
    label="Test Data"
)

plt.scatter(
    [x for x, y in validation],
    [y for x, y in validation],
    label="Validation Data"
)

plt.plot(
    curve_x,
    curve_y,
    label="Polynomial Degree " + str(best_degree)
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Regression - Noisy Dataset")
plt.legend()
plt.grid(True)
plt.show()
