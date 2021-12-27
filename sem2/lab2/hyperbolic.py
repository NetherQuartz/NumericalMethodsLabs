import fire
import numpy as np
import matplotlib.pyplot as plt

from utilities import str2fun


def tma(a, b, c, d):
    size = len(a)
    p, q = [], []
    p.append(-c[0] / b[0])
    q.append(d[0] / b[0])

    for i in range(1, size):
        p_tmp = -c[i] / (b[i] + a[i] * p[i - 1])
        q_tmp = (d[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1])
        p.append(p_tmp)
        q.append(q_tmp)

    x = [0 for _ in range(size)]
    x[size - 1] = q[size - 1]

    for i in range(size - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return x


def solve_analytic(l, N, K, T, solution):
    h = l / N
    tau = T / K
    u = np.zeros((K, N))
    for k in range(K):
        for j in range(N):
            u[k][j] = solution(j * h, k * tau)
    return u


def implicit_solver(a, b, c, alpha, beta, gamma, delta, psi1, psi2, f, phi0, phil, h, tau, N, K):
    A = np.zeros(N)
    B = np.zeros(N)
    C = np.zeros(N)
    D = np.zeros(N)
    u = np.zeros((K, N))

    for j in range(N):
        u[0][j] = psi1(j * h)
        u[1][j] = psi1(j * h) + psi2(j * h) * tau

    sigma = (tau ** 2) / (h ** 2)

    for k in range(2, K):
        for j in range(1, N - 1):
            A[j] = b * (tau ** 2) / (2 * h) - (a ** 2) * sigma
            B[j] = 1 + 2 * (a ** 2) * sigma - c * (tau ** 2)
            C[j] = -b * (tau ** 2) / (2 * h) - (a ** 2) * sigma
            D[j] = 2 * u[k - 1][j] - u[k - 2][j]

        D[0] = phi0(k * tau)
        D[-1] = phil(k * tau)

        B[0] = 1
        A[0] = 0
        B[-1] = 1
        C[-1] = 0

        u[k] = tma(A, B, C, D)

    return u


def explicit_solver(a, b, c, alpha, beta, gamma, delta, psi1, psi2, f, phi0, phil, h, tau, N, K):
    u = np.zeros((K, N))
    for j in range(N):
        u[0][j] = psi1(j * h)
        u[1][j] = psi1(j * h) + psi2(j * h) * tau

    for k in range(2, K):
        for j in range(1, N - 1):
            p1 = (a ** 2) * (tau ** 2) / (h ** 2)
            p2 = b * (tau ** 2) / (2 * h)
            p3 = c * (tau ** 2)

            u[k][j] = p1 * (u[k - 1][j + 1] - 2 * u[k - 1][j] + u[k - 1][j - 1]) \
                + p2 * (u[k - 1][j + 1] - u[k - 1][j - 1]) + p3 * u[k - 1][j] + 2 * u[k - 1][j] - u[k - 2][j]

        u[k][0] = -alpha / h / (beta - alpha / h) * u[k][1] + phi0(k * tau) / (beta - alpha / h)
        u[k][-1] = gamma / h / (delta + gamma / h) * u[k][-2] + phil(k * tau) / (delta + gamma / h)

    return u


def solve(solver, data, N, K, T):
    a = data["a"]
    b = data["b"]
    c = data["c"]
    alpha = data["alpha"]
    beta = data["beta"]
    gamma = data["gamma"]
    delta = data["delta"]
    l = data["l"]
    psi1 = str2fun(data["psi1"], variables="x")
    psi2 = str2fun(data["psi2"], variables="x")
    f = str2fun(data["f"], variables="x,t")
    phi0 = str2fun(data["phi0"], variables="t")
    phil = str2fun(data["phil"], variables="t")
    solution = str2fun(data["solution"], variables="x,t")

    h = l / N
    tau = T / K
    sigma = tau / (h ** 2)

    print(sigma)

    if solver is solve_analytic:
        return solve_analytic(l, N, K, T, solution)

    return solver(a, b, c, alpha, beta, gamma, delta, psi1, psi2, f, phi0, phil, h, tau, N, K)


def main():
    data = {
        "a": 1,
        "b": 2,
        "c": -2,
        "l": np.pi / 2,
        "alpha": 0,
        "beta": 1,
        "gamma": 0,
        "delta": 1,
        "f": "0",
        "phi0": "cos(2 * t)",
        "phil": "0",
        "psi1": "exp(-x) * cos(x)",
        "psi2": "0",
        "solution": "exp(-x) * cos(x) * cos(2 * t)",
    }

    # l — верхняя граница x
    N = 40  # количество отрезков x
    K = 10000  # количество отрезков t
    T = 10  # верхняя граница времени

    explicit = solve(explicit_solver, data, N, K, T)
    implicit = solve(implicit_solver, data, N, K, T)

    xaxis = np.linspace(0, data["l"], N)
    analytic = solve(solve_analytic, data, N, K, T)

    t = 0.5

    plt.plot(xaxis, analytic[0], label="analytical, t=0")
    plt.plot(xaxis, explicit[0], label="explicit, t=0")
    plt.plot(xaxis, implicit[0], label="implicit, t=0")

    plt.plot(xaxis, analytic[int(K / T * t)], label=f"analytical, t={t}")
    plt.plot(xaxis, explicit[int(K / T * t)], label=f"explicit, t={t}")
    plt.plot(xaxis, implicit[int(K / T * t)], label=f"implicit, t={t}")

    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    plt.show()

    eps = {
        "explicit": [],
        "implicit": []
    }

    l = data["l"]

    for N, K, T in [(5, 800, 10),
                    (10, 1000, 10),
                    (20, 4000, 10),
                    (40, 15000, 10)]:

        explicit = solve(explicit_solver, data, N, K, T)
        implicit = solve(implicit_solver, data, N, K, T)
        analytic = solve(solve_analytic, data, N, K, T)

        analytic_sol = analytic[int(K / T * t), :]
        explicit_sol = explicit[int(K / T * t), :]
        implicit_sol = implicit[int(K / T * t), :]

        for method, sol in zip(["explicit", "implicit"],
                               [explicit_sol, implicit_sol]):
            eps[method].append((np.mean(np.abs(analytic_sol - sol)), l / N))

    for method, value in eps.items():
        print(method, value)
        mean, step = list(zip(*value))
        plt.plot(step, mean, label=method, marker="o")

    plt.xlabel("Шаг")
    plt.ylabel("Погрешность")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)
