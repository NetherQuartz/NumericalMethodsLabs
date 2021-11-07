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


def implicit_solver(l, psi, f, phi0, phil, h, tau, sigma, N, K):
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)
    u = np.zeros((K, N))

    for i in range(1, N - 1):
        u[0][i] = psi(i * h)
    u[0][-1] = 0

    for k in range(1, K):
        for j in range(1, N - 1):
            a[j] = sigma
            b[j] = -(1 + 2 * sigma)
            c[j] = sigma
            d[j] = -u[k - 1][j] - tau * f(j * h, k * tau)

        a[0] = 0
        b[0] = 1
        c[0] = 0
        d[0] = phi0(k * tau)
        a[-1] = -1
        b[-1] = 1
        c[-1] = 0
        d[-1] = h * phil(k * tau)

        u[k] = tma(a, b, c, d)

    return u


def explicit_solver(l, psi, f, phi0, phil, h, tau, sigma, N, K):
    u = np.zeros((K, N))
    for j in range(1, N - 1):
        u[0][j] = psi(j * h)

    for k in range(1, K):
        u[k][0] = phi0(k * tau)
        for j in range(1, N - 1):
            u[k][j] = sigma * u[k - 1][j + 1] + (1 - 2 * sigma) * u[k - 1][j] + sigma * u[k - 1][j - 1] \
                      + tau * f(j * h, k * tau)

        u[k][-1] = u[k][-2] + phil(k * tau) * h

    return u


def crank_nicholson_solver(l, psi, f, phi0, phil, h, tau, sigma, N, K):
    theta = 0.5
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)
    u = np.zeros((K, N))
    for j in range(1, N - 1):
        u[0][j] = psi(j * h)

    for k in range(1, K):
        for j in range(1, N - 1):
            a[j] = sigma
            b[j] = -(1 + 2 * sigma)
            c[j] = sigma
            d[j] = -u[k - 1][j] - tau * f(j * h, k * tau)

        a[0] = 0
        b[0] = 1
        c[0] = 0
        d[0] = phi0(k * tau)
        a[-1] = -1
        b[-1] = 1
        c[-1] = 0
        d[-1] = h * phil(k * tau)

        tmp_imp = tma(a, b, c, d)

        tmp_exp = np.zeros(N)
        tmp_exp[0] = phi0(k * tau)
        for j in range(1, N - 1):
            tmp_exp[j] = sigma * u[k - 1][j + 1] + (1 - 2 * sigma) * u[k - 1][j] \
                         + sigma * u[k - 1][j - 1] + tau * f(j * h, k * tau)

        tmp_exp[-1] = tmp_exp[-2] + phil(k * tau) * h

        for j in range(N):
            u[k][j] = theta * tmp_imp[j] + (1 - theta) * tmp_exp[j]

    return u


def solve(solver, data, N, K, T):
    l = data["l"]
    psi = str2fun(data["psi"], variables="x")
    f = str2fun(data["f"], variables="x,t")
    phi0 = str2fun(data["phi0"])
    phil = str2fun(data["phil"])
    solution = str2fun(data["solution"], variables="x,t")

    h = l / N
    tau = T / K
    sigma = tau / (h ** 2)

    print(sigma)

    if solver is solve_analytic:
        return solve_analytic(l, N, K, T, solution)

    return solver(l, psi, f, phi0, phil, h, tau, sigma, N, K)


def main():
    data = {
        "l": np.pi / 2,
        "psi": "0",
        "f": "cos(x) * (cos(t) + sin(t))",
        "phi0": "sin(t)",
        "phil": "-sin(t)",
        "solution": "sin(t) * cos(x)",
    }

    # l — верхняя граница x
    N = 10  # количество отрезков x
    K = 1000  # количество отрезков t
    T = 10  # верхняя граница времени

    analytic = solve(solve_analytic, data, N, K, T)
    explicit = solve(explicit_solver, data, N, K, T)
    implicit = solve(implicit_solver, data, N, K, T)
    crank_nicholson = solve(crank_nicholson_solver, data, N, K, T)

    xaxis = np.linspace(0, data["l"], N)

    analytic_f = str2fun(data["solution"], variables="t,x")

    t = 0.5

    plt.plot(xaxis, analytic_f(0, xaxis), label="analytical, t=0")
    plt.plot(xaxis, explicit[0, :], label="explicit, t=0")
    plt.plot(xaxis, implicit[0, :], label="implicit, t=0")
    plt.plot(xaxis, crank_nicholson[0, :], label="crank_nicholson, t=0")

    plt.plot(xaxis, analytic_f(t, xaxis), label=f"analytical, t={t}")
    plt.plot(xaxis, explicit[int(K / T * t), :], label=f"explicit, t={t}")
    plt.plot(xaxis, implicit[int(K / T * t), :], label=f"implicit, t={t}")
    plt.plot(xaxis, crank_nicholson[int(K / T * t), :], label=f"crank_nicholson, t={t}")

    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    plt.show()

    eps = {
        "explicit": [],
        "implicit": [],
        "crank_nicholson": []
    }

    l = data["l"]

    for N, K, T in [(5, 800, 10),
                    (10, 1000, 10),
                    (20, 4000, 10),
                    (40, 15000, 10)]:
        xaxis = np.linspace(0, data["l"], N)

        explicit = solve(explicit_solver, data, N, K, T)
        implicit = solve(implicit_solver, data, N, K, T)
        crank_nicholson = solve(crank_nicholson_solver, data, N, K, T)

        analytic_sol = analytic_f(t, xaxis)
        explicit_sol = explicit[int(K / T * t), :]
        implicit_sol = implicit[int(K / T * t), :]
        crank_nicholson_sol = crank_nicholson[int(K / T * t), :]

        for method, sol in zip(["explicit", "implicit", "crank_nicholson"],
                               [explicit_sol, implicit_sol, crank_nicholson_sol]):
            eps[method].append((np.mean(np.abs(analytic_sol - sol)), l / N))

    for method, value in eps.items():
        print(method, value)
        mean, step = list(zip(*value))
        plt.plot(step, mean, label=method)

    plt.xlabel("Шаг")
    plt.ylabel("Погрешность")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)
