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


def tma(a, b, c, d):
    size = len(a)
    p = [-c[0] / b[0]]
    q = [d[0] / b[0]]

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


def alter_directions_solve(phi1, phi2, phi3, phi4, psi, N1, N2, NT, l1, l2, lt, f, a):
    u = np.zeros((N1, N2, NT))
    h1 = l1 / (N1 - 1)
    h2 = l2 / (N2 - 1)
    ht = lt / (NT - 1)

    # s1 = a * ht / h1 ** 2
    # s2 = a * ht / h2 ** 2

    s1 = 2 * a / (ht * h1 ** 2)
    s2 = 2 * a / (ht * h2 ** 2)

    for i in range(N1):
        for j in range(N2):
            u[i, j, 0] = psi(i * h1, j * h2, a)

    u_sec = np.zeros((N1, N2))
    A = ([], [])
    B = ([], [])
    C = ([], [])

    for _ in range(N1):
        A[0].append(-s1)
        B[0].append(2 * s1 + 1)
        C[0].append(-s1)

    A[0][0] = 0
    B[0][0] = 1
    C[0][0] = 0
    A[0][-1] = -1
    B[0][-1] = 1
    C[0][-1] = 0

    for _ in range(N2):
        A[1].append(-s2)
        B[1].append(2 * s2 + 1)
        C[1].append(-s2)

    A[1][0] = 0
    B[1][0] = -1
    C[1][0] = 1
    A[1][-1] = 0
    B[1][-1] = 1
    C[1][-1] = 0

    for k in range(1, NT):
        t1 = (k - 1/2) * ht
        t2 = k * ht
        d = np.zeros(N1)

        # for i in range(N1):
            # u_sec[i, -1] = phi2(i * h1, t1, a)
            # u_sec[i, 0] = u_sec[i, 1] - phi1(i * h1, t1, a) * h2

        for j in range(1, N2 - 1):
            d[0] = phi3(j * h2, t1, a)
            d[-1] = phi4(j * h2, t1, a) * h1
            for i in range(1, N1 - 1):
            # for i in range(N1):
                d[i] = u[i, j, k - 1] + s2 * (u[i, j + 1, k - 1] - 2 * u[i, j, k - 1] + u[i, j - 1, k - 1])

            ux = tma(A[0], B[0], C[0], d)
            for i in range(N1):
                u_sec[i, j] = ux[i]

        for i in range(N1):
            u_sec[i, 0] = u_sec[i, 1] - phi1(i * h1, t1, a) * h2
            u_sec[i, -1] = phi2(i * h1, t1, a)

        d = np.zeros(N2)

        # for j in range(N2):
        #     u[0, j, k] = phi3(j * h2, t2, a)
        #     u[-1, j, k] = u[-2, j, k] + phi4(j * h2, t2, a) * h1

        for i in range(1, N1 - 1):
            d[0] = phi1(i * h1, t2, a) * h2
            d[-1] = phi2(i * h1, t2, a)
            for j in range(1, N2 - 1):
            # for j in range(N2):
                d[j] = u_sec[i, j] + s1 * (u_sec[i + 1, j] - 2 * u_sec[i, j] + u_sec[i - 1, j])

            uy = tma(A[1], B[1], C[1], d)
            for j in range(N2):
                u[i, j, k] = uy[j]

        for j in range(N2):
            u[0, j, k] = phi3(j * h2, t2, a)
            u[-1, j, k] = u[-2, j, k] + phi4(j * h2, t2, a) * h1

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
        "lt": 1,
        "a": 0.0000001,
        "solution": "cos(2 * x) * sinh(y) * exp(-3 * a * t)"
    }

    N1 = 10  # количество отрезков x
    N2 = 10  # количество отрезков y
    NT = 100  # количество отрезков времени

    analytic = solve(analytic_solve, data, N1, N2, NT)
    explicit = solve(explicit_solve, data, N1, N2, NT)
    alter_dir = solve(alter_directions_solve, data, N1, N2, NT)

    fig = plt.figure()
    plt.suptitle("Явный метод в три разных момента времени\n(красная сетка — аналитическое решение)")
    ax = fig.add_subplot(131, projection="3d", zlim=(0, 0.8), title="$t=0$")
    plot3d(ax, explicit[:, :, 0], data["l1"], data["l2"])
    plot3d(ax, analytic[:, :, 0], data["l1"], data["l2"], wireframe=True)

    ax = fig.add_subplot(132, projection="3d", zlim=(0, 0.8), title=r"$t=\frac{lt}{2}=" + str(data["lt"] / 2) + "$")
    plot3d(ax, explicit[:, :, explicit.shape[2] // 2], data["l1"], data["l2"])
    plot3d(ax, analytic[:, :, analytic.shape[2] // 2], data["l1"], data["l2"], wireframe=True)

    ax = fig.add_subplot(133, projection="3d", zlim=(0, 0.8), title=f"$t=lt={data['lt']}$")
    plot3d(ax, explicit[:, :, -1], data["l1"], data["l2"])
    plot3d(ax, analytic[:, :, -1], data["l1"], data["l2"], wireframe=True)

    fig = plt.figure()
    plt.suptitle("Метод переменных направлений метод в три разных момента времени\n(красная сетка — аналитическое решение)")
    ax = fig.add_subplot(131, projection="3d", zlim=(0, 0.8), title="$t=0$")
    plot3d(ax, alter_dir[:, :, 0], data["l1"], data["l2"])
    plot3d(ax, analytic[:, :, 0], data["l1"], data["l2"], wireframe=True)

    ax = fig.add_subplot(132, projection="3d", zlim=(0, 0.8), title=r"$t=\frac{lt}{2}=" + str(data["lt"] / 2) + "$")
    plot3d(ax, alter_dir[:, :, explicit.shape[2] // 2], data["l1"], data["l2"])
    plot3d(ax, analytic[:, :, analytic.shape[2] // 2], data["l1"], data["l2"], wireframe=True)

    ax = fig.add_subplot(133, projection="3d", zlim=(0, 0.8), title=f"$t=lt={data['lt']}$")
    plot3d(ax, alter_dir[:, :, -1], data["l1"], data["l2"])
    plot3d(ax, analytic[:, :, -1], data["l1"], data["l2"], wireframe=True)

    plt.figure()
    K = 60
    plt.suptitle(f"Метод переменных направлений при $k={K}$")
    x = np.linspace(0, data["l1"], N1)
    y = np.linspace(0, data["l2"], N2)

    plt.subplot(121)
    plt.plot(x, analytic[:, 0, K], label=f"Analytic $y=0$")
    plt.plot(x, analytic[:, -1, K], label=f"Analytic $y={data['l2']:.2f}$")
    plt.plot(x, analytic[:, analytic.shape[1] // 2, K], label=f"Analytic $y={data['l2'] / 2:.2f}$")


    plt.plot(x, explicit[:, 0, K], label=f"Explicit $y=0$")
    plt.plot(x, explicit[:, -1, K], label=f"Explicit $y={data['l2']:.2f}$")
    plt.plot(x, explicit[:, explicit.shape[1] // 2, K], label=f"Explicit $y={data['l2'] / 2:.2f}$")

    plt.plot(x, alter_dir[:, 0, K], label=f"Alt. dir. $y=0$")
    plt.plot(x, alter_dir[:, -1, K], label=f"Alt. dir. $y={data['l2']:.2f}$")
    plt.plot(x, alter_dir[:, explicit.shape[1] // 2, K], label=f"Alt. dir. $y={data['l2'] / 2:.2f}$")

    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)

    plt.subplot(122)
    plt.plot(y, analytic[0, :, K], label=f"Analytic $x=0$")
    plt.plot(y, analytic[-1, :, K], label=f"Analytic $x={data['l1']:.2f}$")
    plt.plot(y, analytic[analytic.shape[0] // 2, :, K], label=f"Analytic $x={data['l1'] / 2:.2f}$")

    plt.plot(y, explicit[0, :, K], label=f"Explicit $x=0$")
    plt.plot(y, explicit[-1, :, K], label=f"Explicit $x={data['l1']:.2f}$")
    plt.plot(y, explicit[explicit.shape[0] // 2, :, K], label=f"Explicit $y={data['l1'] / 2:.2f}$")

    plt.plot(y, alter_dir[0, :, K], label=f"Alt. dir. $x=0$")
    plt.plot(y, alter_dir[-1, :, K], label=f"Alt. dir. $x={data['l1']:.2f}$")
    plt.plot(y, alter_dir[alter_dir.shape[0] // 2, :, K], label=f"Alt. dir. $x={data['l1'] / 2:.2f}$")

    plt.xlabel("y")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.suptitle("Погрешность в зависимости от шага $h=max(h1,h2,ht)$ для шести разных моментов $t$")
    NT = 11
    for j, k in enumerate(range(0, NT, 2)):
        eps = {
            "explicit": [],
            "alter. dir.": []
        }
        plt.subplot(2, 3, j + 1)
        plt.title(f"$t={data['lt'] * (j + 1) / 6:.3f}$")
        for N in [2, 3, 5, 10, 20, 50]:

            explicit = solve(explicit_solve, data, N, N, NT)
            alter_dir = solve(alter_directions_solve, data, N, N, NT)
            analytic = solve(analytic_solve, data, N, N, NT)

            h1 = data["l1"] / (N - 1)
            h2 = data["l2"] / (N - 1)
            ht = data["lt"] / (NT - 1)
            h = max(h1, h2, ht)

            for method, sol in zip(["explicit", "alter. dir."], [explicit, alter_dir]):
                eps[method].append((np.sqrt(np.sum((analytic[:, :, k] - sol[:, :, k]) ** 2)), h))

        i = 0
        for method, value in eps.items():
            print(method, sorted(value, key=lambda t: t[1]))
            mean, step = list(zip(*value))
            plt.plot(step, mean, label=method, marker="o", linewidth=2, linestyle="dotted" if i == 1 else "solid")
            i += 1

        if j + 1 > 3:
            plt.xlabel("Шаг")
        plt.ylabel("Погрешность")
        plt.grid(True)
        plt.legend()
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
