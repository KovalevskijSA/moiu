import numpy as np
import scipy.optimize as sc
import sympy as sm


def convex(B0, B_i, c0, c_i, alpha, x_star, J0, x, m, n):
    f = 0.5*np.dot(np.dot(np.dot(x, B0.transpose()), B0), x) + np.dot(c0.transpose(), x)
    g = np.array([sm.Symbol("0") for i in range(m)])
    for i in range(m):
        g[i] = 0.5*np.dot(np.dot(np.dot(x, B_i[i].transpose()), B_i[i]), x) + np.dot(c_i[i].transpose(), x) + alpha[i]

    f_star = f.subs(x[0], x_star[0])
    for i in range(1, n):
        f_star = f_star.subs(x[i], x_star[i])

    g_star = np.array([g[i].subs(x[0], x_star[0]) for i in range(m)])
    for i in range(m):
        for j in range(1, n):
            g_star[i] = g_star[i].subs(x[j], x_star[j])

    I0 = []
    for i in range(m):
        if (g_star[i] > -1e-4) and (g_star[i] < 1e-4):
            I0.append(i)

    diff_f = []
    for i in range(n):
        diff_f.append(sm.diff(f, x[i]))

    for i in range(n):
        for j in range(n):
            diff_f[i] = diff_f[i].subs(x[j], x_star[j])

    diff_g = [[] for i in range(len(I0))]
    for i in range(len(I0)):
        for j in range(n):
            diff_g[i].append(sm.diff(g[I0[i]], x[j]))

    for i in range(len(I0)):
        for j in range(n):
            for k in range(n):
                diff_g[i][j] = diff_g[i][j].subs(x[k], x_star[k])

    d = [(0, 1) if i in J0 else (-1, 1) for i in range(n)]
    b = np.array([0 for i in range(len(I0))])

    result = sc.linprog(np.array(diff_f), A_ub=np.array(diff_g), b_ub=b, bounds=d)
    func = result.fun
    l = result.x
    if func == 0:
        print("Оптимальный план:")
        return x_star

    x_ = np.array([0 for i in range(n)])
    alpha = 1
    h = 0.01
    delta_x = x_ - x_star
    while(True):
        if (np.dot(diff_f, l) + alpha*np.dot(diff_f, delta_x)) < 0:
            break
        alpha += h

    t = 0.5
    h1 = 0.1
    while(True):
        xt = x_star + t*(l + alpha*delta_x)
        f_xt = f.subs(x[0], xt[0])
        for i in range(1, n):
            f_xt = f_xt.subs(x[i], xt[i])

        g_xt = np.array([g[i].subs(x[0], xt[0]) for i in range(m)])
        for i in range(m):
            for j in range(1, n):
                g_xt[i] = g_xt[i].subs(x[j], xt[j])

        if all([g_xt[i] < 0 for i in range(m)]) and (f_xt < f_star):
            break
        t += h1

    print("F(x*):")
    print(f_star)
    print("F(x(t)):")
    print(f_xt)
    print("x(t):")
    return xt


if __name__ == '__main__':
    m = 5
    n = 8
    x = np.array([sm.Symbol("x" + str(i)) for i in range(n)])
    B0 = np.array([
        [2.0, 1.0, 0.0, 4.0, 0.0, 3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 3.0, 1.0, 1.0, 3.0, 2.0],
        [1.0, 3.0, 0.0, 5.0, 0.0, 4.0, 0.0, 4.0]
    ])
    B_i = np.array([
        np.array([
            [0.0, 0.0, 0.5, 2.5, 1.0, 0.0, -2.5, -2.0],
            [0.5, 0.5, -0.5, 0.0, 0.5, -0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5, 0.0, 0.5, 1.0, 2.5, 4.0]
        ]),
        np.array([
            [1.0, 2.0, -1.5, 3.0, -2.5, 0.0, -1.0, -0.5],
            [-1.5, -0.5, -1.0, -2.5, 3.5, -3.0, -1.5, -0.5],
            [1.5, 2.5, -1.0, 1.0, 2.5, 1.5, 3.0, 0.0]
        ]),
        np.array([
            [0.75, 0.5, -1.0, 0.25, 0.25, 0.0, 0.25, 0.75],
            [-1.0, 1.0, 4.0, 0.75, 0.75, 0.5, 7.0, -0.75],
            [0.5, -0.25, 0.5, 0.75, 0.5, 1.25, -0.75, -0.25]
        ]),
        np.array([
            [1.5, -1.5, -1.5, 2.0, 1.5, 0.0, 0.5, -1.5],
            [-0.5, -2.5, -0.5, -5.0, -2.5, 3.5, 1.0, 2.0],
            [-2.5, 1.0, -2.0, -1.5, -2.5, 0.5, 8.5, -2.5]
        ]),
        np.array([
            [1.0, 0.25, -0.5, 1.25, 1.25, -0.5, 0.25, -0.75],
            [-1.0, -0.75, -0.75, 0.5, -0.25, 1.25, 0.25, -0.5],
            [0.0, 0.75, 0.5, -0.5, -1.0, 1.0, -1.0, 1.0]
        ])
    ])
    c0 = np.array([-1.0, -1.0, -1.0, -1.0, -2.0, 0.0, -2.0, -3.0])
    c_i = np.array([
        np.array([0.0, 60.0, 80.0, 0.0, 0.0, 0.0, 40.0, 0.0]),
        np.array([2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 3.0, 0.0]),
        np.array([0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, -2.0, 1.0, 2.0, 0.0, 0.0, -2.0, 1.0]),
        np.array([-4.0, -2.0, 6.0, 0.0, 4.0, -2.0, 60.0, 2.0])
    ])
    alpha = np.array([-687.1250, -666.6250, -349.5938, -254.6250, -45.1563])
    x_star = np.array([0.0, 8.0, 2.0, 1.0, 0.0, 4.0, 0.0, 0.0])
    J0 = [0, 4, 6, 7]
    res = convex(B0, B_i, c0, c_i, alpha, x_star, J0, x, m, n)
    print(res)

    # m = 5
    # n = 8
    # x = np.array([sm.Symbol("x" + str(i)) for i in range(n)])
    # B0 = np.array([
    #     [2.0, 1.0, 0.0, 4.0, 0.0, 3.0, 0.0, 0.0],
    #     [0.0, 4.0, 0.0, 3.0, 1.0, 1.0, 3.0, 2.0],
    #     [1.0, 3.0, 0.0, 5.0, 0.0, 4.0, 0.0, 4.0]
    # ])
    # B_i = np.array([
    #     np.array([
    #         [0.0, 0.0, 0.5, 2.5, 1.0, 0.0, -2.5, -2.0],
    #         [0.5, 0.5, -0.5, 0.0, 0.5, -0.5, -0.5, -0.5],
    #         [0.5, 0.5, 0.5, 0.0, 0.5, 1.0, 2.5, 4.0]
    #     ]),
    #     np.array([
    #         [1.0, 2.0, -1.5, 3.0, -2.5, 0.0, -1.0, -0.5],
    #         [-1.5, -0.5, -1.0, 2.5, 3.5, 3.0, -1.5, -0.5],
    #         [1.5, 2.5, 1.0, 1.0, 2.5, 1.5, 3.0, 0.0]
    #     ]),
    #     np.array([
    #         [0.75, 0.5, -1.0, 0.25, 0.25, 0.0, 0.25, 0.75],
    #         [-1.0, 1.0, 1.0, 0.75, 0.75, 0.5, 1.0, -0.75],
    #         [0.5, -0.25, 0.5, 0.75, 0.5, 1.25, -0.75, -0.25]
    #     ]),
    #     np.array([
    #         [1.5, -1.5, -1.5, 2.0, 1.5, 0.0, 0.5, -1.5],
    #         [-0.5, -2.5, -0.5, -1.0, -2.5, 2.5, 1.0, 2.0],
    #         [-2.5, 1.0, -2.0, -1.5, -2.5, 0.5, 2.5, -2.5]
    #     ]),
    #     np.array([
    #         [1.0, 0.25, -0.5, 1.25, 1.25, -0.5, 0.25, -0.75],
    #         [-1.0, -0.75, -0.75, 0.5, -0.25, 1.25, 0.25, -0.5],
    #         [0.0, 0.75, 0.5, -0.5, -1.0, 1.0, -1.0, 1.0]
    #     ])
    # ])
    # c0 = np.array([-1.0, -1.0, -1.0, -1.0, -2.0, 0.0, -2.0, -3.0])
    # c_i = np.array([
    #     np.array([0.0, 60.0, 80.0, 0.0, 0.0, 0.0, 40.0, 0.0]),
    #     np.array([2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 3.0, 0.0]),
    #     np.array([0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    #     np.array([0.0, -2.0, 1.0, 2.0, 0.0, 0.0, -2.0, 1.0]),
    #     np.array([-4.0, -2.0, 6.0, 0.0, 4.0, -2.0, 60.0, 2.0])
    # ])
    # alpha = np.array([-51.7500, -436.7500, -33.7813, -303.3750, -41.7500])
    # x_star = np.array([1.0, 0.0, 0.0, 2.0, 4.0, 2.0, 0.0, 0.0])
    # J0 = [1, 2, 6, 7]
    # res = convex_programming(B0, B_i, c0, c_i, alpha, x_star, J0, x, m, n)
    # print(res)
    #
