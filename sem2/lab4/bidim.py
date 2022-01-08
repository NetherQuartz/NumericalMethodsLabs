import fire
import numpy as np
import matplotlib.pyplot as plt

from utilities import str2fun


def analytic_solve(solution, N1, N2, NT, l1, l2, lt, a):
    x = np.linspace(0, l1, N1)
    y = np.linspace(0, l2, N2)
    t = np.linspace(0, lt, NT)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    u = solution(X, Y, T, a)
    return u


def alter_directions_solve(phi1, phi2, phi3, phi4, psi, N1, N2, NT, l1, l2, lt, f, a):
    pass


def explicit_solve(phi1, phi2, phi3, phi4, psi, N1, N2, NT, l1, l2, lt, f, a):
    u = np.zeros((N1, N2, NT))
    h1 = l1 / (N1 - 1)
    h2 = l2 / (N2 - 1)
    ht = lt / (NT - 1)

    for i in range(N1):
        for j in range(N2):
            u[i, j, 0] = psi(i * h1, j * h2, a)

    for k in range(1, NT):
        for j in range(N2):
            u[0, j, k] = phi3(j * h2, k * ht, a)

        for i in range(1, N1 - 1):
            u[i, -1, k] = phi2(i * h1, k * ht, a)
            for j in range(N2 - 1):
                c1 = u[i - 1, j, k - 1] - 2 * u[i, j, k - 1] + u[i + 1, j, k - 1]
                c2 = u[i, j - 1, k - 1] - 2 * u[i, j, k - 1] + u[i, j + 1, k - 1]
                u[i, j, k] = a * ht * (c1 / h1 ** 2 + c2 / h2 ** 2) + u[i, j, k - 1]

            u[i, 0, k] = u[i, 1, k] - h2 * phi1(i * h1, k * ht, a)

        for j in range(1, N2):
            u[-1, j, k] = u[-2, j, k] + h1 * phi4(j * h2, k * ht, a)

    return u


def solve(solver, data, N1, N2, NT):
    phi1 = str2fun(data["phi1"], variables="x,t,a")
    phi2 = str2fun(data["phi2"], variables="x,t,a")
    phi3 = str2fun(data["phi3"], variables="y,t,a")
    phi4 = str2fun(data["phi4"], variables="y,t,a")
    psi = str2fun(data["psi"], variables="x,y,a")
    solution = str2fun(data["solution"], variables="x,y,t,a")
    l1 = data["l1"]
    l2 = data["l2"]
    lt = data["lt"]
    a = data["a"]
    f = data["f"]

    if solver is analytic_solve:
        return analytic_solve(solution, N1, N2, NT, l1, l2, lt, a)

    return solver(phi1, phi2, phi3, phi4, psi, N1, N2, NT, l1, l2, lt, f, a)


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
        "phi1": "cos(2 * x) * exp(-3 * a * t)",
        "phi2": "3/4 * cos(2 * x) * exp(-3 * a * t)",
        "phi3": "sinh(y) * exp(-3 * a * t)",
        "phi4": "-2 * sinh(y) * exp(-3 * a * t)",
        "psi": "cos(2 * x) * sinh(y)",
        "l1": np.pi / 4,
        "l2": np.log(2),
        "lt": 1000,
        "a": 0.0001,
        "solution": "cos(2 * x) * sinh(y) * exp(-3 * a * t)"
    }

    N1 = 10  # количество отрезков x
    N2 = 15  # количество отрезков y
    NT = 100  # количество отрезков времени

    analytic = solve(analytic_solve, data, N1, N2, NT)
    explicit = solve(explicit_solve, data, N1, N2, NT)

    fig = plt.figure()
    plt.suptitle("Метод переменных направлений в три разных момента времени\n(красная сетка — аналитическое решение)")
    ax = fig.add_subplot(131, projection="3d", zlim=(0, 0.8), title="$t=0$")
    plot3d(ax, explicit[:, :, 0], data["l1"], data["l2"])
    plot3d(ax, analytic[:, :, 0], data["l1"], data["l2"], wireframe=True)

    ax = fig.add_subplot(132, projection="3d", zlim=(0, 0.8), title=r"$t=\frac{lt}{2}=" + str(data["lt"] / 2) + "$")
    plot3d(ax, explicit[:, :, explicit.shape[2] // 2], data["l1"], data["l2"])
    plot3d(ax, analytic[:, :, analytic.shape[2] // 2], data["l1"], data["l2"], wireframe=True)

    ax = fig.add_subplot(133, projection="3d", zlim=(0, 0.8), title=f"$t=lt={data['lt']}$")
    plot3d(ax, explicit[:, :, -1], data["l1"], data["l2"])
    plot3d(ax, analytic[:, :, -1], data["l1"], data["l2"], wireframe=True)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
