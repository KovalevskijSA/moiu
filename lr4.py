import numpy as np
from numpy.linalg import linalg


def inv_matrix(A_inv, i, x, n):
    l = np.dot(A_inv, x)
    li = l[i]
    if l[i] == 0:
        return
    l[i] = -1
    l_ = [(-1 / li) * x for x in l]
    Q = np.eye(n)
    for j in range(0, n):
        Q[j][i] = l_[j]
    ans = np.dot(Q, A_inv)
    return ans


def dual_simplex_method(a, b, c, y, jb):
    m, n = len(a), len(c)
    ab = (np.array([ a[:,j] for j in jb])).transpose()
    ab_inv = linalg.inv(ab)


    koplan = np.dot(y, a) - c

    iter = 0
    while True:
        print('iter', iter)
        print('koplan', koplan)
        kapa_b = np.dot(ab_inv, b)
        print('kapa_b', kapa_b)
        if min(kapa_b) >= 0:
            kapa = [0]*n
            for j, i in zip(kapa_b, jb):
                kapa[i] = j
            print("\nОптимальный план: \n", kapa)
            return
        k = np.argmin(kapa_b)
        j_n = [j for j in range(n) if j not in jb]
        e = np.zeros(m)
        e[k] = 1
        mu = e.dot(ab_inv.dot(a))
        print('mu=', mu)
        if all(mu[i] >= 0 for i in j_n):
            print("Задача несовместна")
            return

        s = [np.inf] * n
        for j in j_n:
            if mu[j] < 0:
                s[j] = -koplan[j] / mu[j]
        print('step=', s)
        j0 = np.argmin(s)
        print('j0', j0)
        y = y + (s[j0] * e).dot(b)
        print('y_new=', y)
        koplan = koplan + s[j0] * mu
        print('koplan_new', koplan)
        jb[k] = j0
        print('j_new=%s', jb)
        ab_inv = inv_matrix(ab_inv, k, a[:,j0], m)


def main():
    c = [2, 2, 1, -10, 1, 4, -2, -3]
    b = [-2, 4, 3]
    a = [[-2, -1, 1, -7, 0, 0, 0, 2],
         [4, 2, 1, 0, 1, 5, -1, -5],
         [1, 1, 0, -1, 0, 3, -1, 1,]]
    y = [1, 1, 1]
    jb = [1, 4, 6]

    # c = [2, 2, 1, -10, 1, 4, 0, -3]
    # b = [-2, 4, -3]
    # a = [[-2, -1, 1, -7, 0, 0, 0, 2],
    #      [4, 2, 1, 0, 1, 5, -1, -5],
    #      [1, 1, 0, -1, 0, 3, 1, 1, ]]
    # y = [1, 1, 1]
    # jb = [1, 4, 6]



    #3
    # c = [12, -2, -6, 20, -18, -5, -7, -20]
    # b = [-2, 8, -2]
    # a = [[-2, -1, 1, -7, 1, 0, 0, 2],
    #      [-4, 2, 1, 0, 5, 1, -1, 5],
    #      [1, 1, 0, 1, 4, 3, 1, 1 ]]
    # y = [-3, -2, -1]
    # jb = [1, 3, 5]

    a = np.array(a)
    dual_simplex_method(a, b, c, y, jb)

if __name__ == '__main__':
    main()
