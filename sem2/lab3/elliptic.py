import fire
import numpy as np
import matplotlib.pyplot as plt

from utilities import str2fun


def solve_analytic(solution, N1, N2, l1, l2):
    h1 = l1 / (N1 - 1)
    h2 = l2 / (N2 - 1)
    u = np.zeros((N1, N2))
    for x in range(N1):
        for y in range(N2):
            u[x][y] = solution(x * h1, y * h2)
    return u


def norm(m: np.ndarray) -> float:
    return np.abs(m).max()


def liebmann_solve(phi1, phi2, phi3, phi4, alpha, beta, N1, N2, l1, l2, f, eps):
    h1 = l1 / (N1 - 1)
    h2 = l2 / (N2 - 1)
    u = np.zeros((N1, N2))
    for j in range(N2):
        u[0, j] = phi1(j * h2)
        u[-1, j] = phi2(j * h2)

    k = 0
    u_prev = u.copy()
    err = []
    while k == 0 or norm(u - u_prev) > eps:
        u_prev = u.copy()
        for i in range(1, N1 - 1):

            for j in range(1, N2 - 1):
                c1 = h2 ** 2 * (u_prev[i + 1, j] + u_prev[i - 1, j])
                c2 = h1 ** 2 * (u_prev[i, j + 1] + u_prev[i, j - 1])
                u[i, j] = (c1 + c2) / (2 * h1 ** 2 + 2 * h2 ** 2 - h1 ** 2 * h2 ** 2)

            u[i, 1] = u[i, 0] + phi3(i * h1) * h2
            u[i, -1] = (h2 * phi4(i * h1) + alpha[3] * u[i, -2]) / (alpha[3] + h2 * beta[3])

        k += 1
        err.append(norm(u - u_prev))
        if k > 5000:
            raise Exception(f"Слишком большое число итераций: {k}")
    return u


def seidel_solve(phi1, phi2, phi3, phi4, alpha, beta, N1, N2, l1, l2, f, eps):
    h1 = l1 / (N1 - 1)
    h2 = l2 / (N2 - 1)
    u = np.zeros((N1, N2))
    for j in range(N2):
        u[0, j] = phi1(j * h2)
        u[-1, j] = phi2(j * h2)

    k = 0
    u_prev = u.copy()
    err = []
    while k == 0 or norm(u - u_prev) > eps:
        u_prev = u.copy()
        for i in range(1, N1 - 1):
            u[i, 1] = u[i, 0] + phi3(i * h1) * h2
            u[i, -1] = (h2 * phi4(i * h1) + alpha[3] * u[i, -2]) / (alpha[3] + h2 * beta[3])
            for j in range(1, N2 - 1):
                c1 = h2 ** 2 * (u_prev[i + 1, j] + u[i - 1, j])
                c2 = h1 ** 2 * (u_prev[i, j + 1] + u[i, j - 1])
                u[i, j] = (c1 + c2) / (2 * h1 ** 2 + 2 * h2 ** 2 - h1 ** 2 * h2 ** 2)

        k += 1
        err.append(norm(u - u_prev))
        if k > 5000:
            raise Exception(f"Слишком большое число итераций: {k}")
    return u


def solve(solver, data, N1, N2):
    phi1 = str2fun(data["phi1"], variables="y")
    phi2 = str2fun(data["phi2"], variables="y")
    phi3 = str2fun(data["phi3"], variables="x")
    phi4 = str2fun(data["phi4"], variables="x")
    eps = data["eps"]
    l1 = data["l1"]
    l2 = data["l2"]
    f = data["f"]
    alpha = data["alpha"]
    beta = data["beta"]
    solution = str2fun(data["solution"], variables="x,y")

    if solver is solve_analytic:
        return solve_analytic(solution, N1, N2, l1, l2)

    return solver(phi1, phi2, phi3, phi4, alpha, beta, N1, N2, l1, l2, f, eps)


def plot3d(ax, u, xn, yn, wireframe=False):
    n1, n2 = u.shape
    x = np.linspace(0, xn, n1)
    y = np.linspace(0, yn, n2)
    X, Y = np.meshgrid(x, y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    if wireframe:
        ax.plot_wireframe(X.T, Y.T, u, color="red")
    else:
        ax.plot_surface(X.T, Y.T, u, cmap="viridis")


def main():
    data = {
        "f": "0",
        "alpha": [0, 0, 1, 1],
        "beta": [1, 1, 0, -1],
        "phi1": "0",
        "phi2": "y",
        "phi3": "sin(x)",
        "phi4": "0",
        "l1": np.pi / 2,
        "l2": 1,
        "solution": "y * sin(x)",
        "eps": 1e-4
    }

    N1 = 10  # количество отрезков x
    N2 = 10  # количество отрезков y

    analytic = solve(solve_analytic, data, N1, N2)
    liebmann = solve(liebmann_solve, data, N1, N2)
    seidel = solve(seidel_solve, data, N1, N2)

    fig = plt.figure()

    ax = fig.add_subplot(121, projection="3d")
    plot3d(ax, liebmann, data["l1"], data["l2"])
    plot3d(ax, analytic, data["l1"], data["l2"], wireframe=True)
    ax.set_title("Liebmann")

    ax = fig.add_subplot(122, projection="3d")
    plot3d(ax, seidel, data["l1"], data["l2"])
    plot3d(ax, analytic, data["l1"], data["l2"], wireframe=True)
    ax.set_title("Seidel")

    plt.figure()
    plt.suptitle("Bounds")
    xt = np.linspace(0, data["l1"], N1)
    yt = np.linspace(0, data["l2"], N2)
    plt.subplot(121)
    plt.plot(yt, analytic[0, :], label="$x=0$ analytic")
    plt.plot(yt, liebmann[0, :], label="$x=0$ liebmann")
    plt.plot(yt, seidel[0, :], label="$x=0$ seidel")
    plt.plot(yt, analytic[-1, :], label="$x=l_1$ analytic")
    plt.plot(yt, liebmann[-1, :], label="$x=l_1$ liebmann")
    plt.plot(yt, seidel[-1, :], label="$x=l_1$ seidel")
    plt.legend()
    plt.xlabel("y")
    plt.ylabel("u")
    plt.grid(True)
    plt.subplot(122)
    plt.plot(xt, analytic[:, 0], label="$y=0$ analytic")
    plt.plot(xt, liebmann[:, 0], label="$y=0$ liebmann")
    plt.plot(xt, seidel[:, 0], label="$y=0$ seidel")
    plt.plot(xt, analytic[:, -1], label="$y=l_2$ analytic")
    plt.plot(xt, liebmann[:, -1], label="$y=l_2$ liebmann")
    plt.plot(xt, seidel[:, -1], label="$y=l_2$ seidel")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u")
    plt.grid(True)

    plt.figure()
    eps = {
        "liebmann": [],
        "seidel": []
    }

    l1 = data["l1"]

    for N in [2, 3, 5, 10, 20, 50][::-1]:

        liebmann = solve(liebmann_solve, data, N, N)
        seidel = solve(seidel_solve, data, N, N)
        analytic = solve(solve_analytic, data, N, N)

        for method, sol in zip(["liebmann", "seidel"], [liebmann, seidel]):
            eps[method].append((norm(analytic - sol), N))

    for method, value in eps.items():
        print(method, sorted(value, key=lambda t: t[1]))
        mean, step = list(zip(*value))
        plt.plot(1 / np.array(step[::-1]), mean, label=method, marker="o")

    plt.xlabel("Шаг")
    plt.ylabel("Погрешность")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
