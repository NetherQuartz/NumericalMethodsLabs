import fire
import numpy as np
import matplotlib.pyplot as plt

from sem1.lab1_2.tdma import tdma_solve
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


def implicit_solver(l, psi, f, phi0, phil, h, tau, sigma, N, K, bound_type):
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

        if bound_type == 'a1p1':
            a[0] = 0
            b[0] = -(1 + 2 * sigma)
            c[0] = sigma
            d[0] = -(u[k - 1][0] + sigma * phi0(k * tau))
            a[-1] = sigma
            b[-1] = -(1 + 2 * sigma)
            c[-1] = 0
            d[-1] = -(u[k - 1][-1] + sigma * phil(k * tau))
        elif bound_type == 'a1p2':
            a[0] = 0
            b[0] = -(1 + 2 * sigma)
            c[0] = sigma
            d[0] = -(u[k - 1][0] + sigma * phi0(k * tau)) - tau * f(0, k * tau)
            a[-1] = sigma
            b[-1] = -(1 + 2 * sigma)
            c[-1] = 0
            d[-1] = -(u[k - 1][-1] + sigma * phil(k * tau)) - tau * f((N - 1) * h, k * tau)
        elif bound_type == 'a1p3':
            a[0] = 0
            b[0] = -(1 + 2 * sigma)
            c[0] = sigma
            d[0] = -((1 - sigma) * u[k - 1][1] + sigma / 2 * u[k - 1][0]) - tau * f(0, k * tau) - sigma * phi0(k * tau)
            a[-1] = sigma
            b[-1] = -(1 + 2 * sigma)
            c[-1] = 0
            d[-1] = phil(k * tau) + f((N - 1) * h, k * tau) * h / (2 * tau) * u[k - 1][-1]

        u[k] = tma(a, b, c, d)

    return u


def explicit_solver(l, psi, f, phi0, phil, h, tau, sigma, N, K, bound_type):
    u = np.zeros((K, N))
    for j in range(1, N - 1):
        u[0][j] = psi(j * h)

    for k in range(1, K):
        u[k][0] = phi0(k * tau)
        for j in range(1, N - 1):
            u[k][j] = sigma * u[k - 1][j + 1] + (1 - 2 * sigma) * u[k - 1][j] + sigma * u[k - 1][j - 1] \
                      + tau * f(j * h, k * tau)

        if bound_type == 'a1p1':
            u[k][-1] = u[k][-2] + phil(k * tau) * h
        elif bound_type == 'a1p2':
            u[k][-1] = phil(k * tau)
        elif bound_type == 'a1p3':
            u[k][-1] = (phil(k * tau) + u[k][-2] / h + 2 * tau * u[k - 1][-1] / h) / (1 / h + 2 * tau / h)

    return u


def crank_nicholson_solver(l, psi, f, phi0, phil, h, tau, sigma, N, K, bound_type):
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

        if bound_type == 'a1p1':
            a[0] = 0
            b[0] = -(1 + 2 * sigma)
            c[0] = sigma
            d[0] = -(u[k - 1][0] + sigma * phi0(k * tau))
            a[-1] = sigma
            b[-1] = -(1 + 2 * sigma)
            c[-1] = 0
            d[-1] = -(u[k - 1][-1] + sigma * phil(k * tau))
        elif bound_type == 'a1p2':
            a[0] = 0
            b[0] = -(1 + 2 * sigma)
            c[0] = sigma
            d[0] = -(u[k - 1][0] + sigma * phi0(k * tau)) - tau * f(0, k * tau)
            a[-1] = sigma
            b[-1] = -(1 + 2 * sigma)
            c[-1] = 0
            d[-1] = -(u[k - 1][-1] + sigma * phil(k * tau)) - tau * f((N - 1) * h, k * tau)
        elif bound_type == 'a1p3':
            a[0] = 0
            b[0] = -(1 + 2 * sigma)
            c[0] = sigma
            d[0] = -((1 - sigma) * u[k - 1][1] + sigma / 2 * u[k - 1][0]) - tau * f(0, k * tau) - sigma * phi0(k * tau)
            a[-1] = sigma
            b[-1] = -(1 + 2 * sigma)
            c[-1] = 0
            d[-1] = phil(k * tau) + f((N - 1) * h, k * tau) * h / (2 * tau) * u[k - 1][-1]

        tmp_imp = tma(a, b, c, d)

        tmp_exp = np.zeros(N)
        tmp_exp[0] = phi0(tau)
        for j in range(1, N - 1):
            tmp_exp[j] = sigma * u[k - 1][j + 1] + (1 - 2 * sigma) * u[k - 1][j] \
                         + sigma * u[k - 1][j - 1] + tau * f(j * h, k * tau)
        tmp_exp[-1] = phil(tau)

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
    bound_type = data["bound_type"]

    h = l / N
    tau = T / K
    sigma = tau / (h ** 2)

    if solver is solve_analytic:
        return solve_analytic(l, N, K, T, solution)

    return solver(l, psi, f, phi0, phil, h, tau, sigma, N, K, bound_type)


def main():
    data = {
        "l": np.pi,
        "psi": "0",
        "f": "cos(x) * (cos(t) + sin(t))",
        "phi0": "sin(t)",
        "phil": "-sin(t)",
        "solution": "sin(t) * cos(x)",
        "bound_type": "a1p2"
    }

    N = 30
    K = 100
    T = 30

    analytic = solve(solve_analytic, data, N, K, T)
    explicit = solve(explicit_solver, data, N, K, T)
    implicit = solve(implicit_solver, data, N, K, T)
    crank_nicholson = solve(crank_nicholson_solver, data, N, K, T)

    t = 20
    x = 20

    # plt.title(f"$t={t}$")
    plt.title(f"$x={x}$")

    plt.plot(analytic[x, :], label="analytic")
    plt.plot(implicit[x, :], label="implicit")
    # plt.plot(explicit[x, :], label="explicit")
    # plt.plot(crank_nicholson[x, :], label="crank_nicholson")

    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)