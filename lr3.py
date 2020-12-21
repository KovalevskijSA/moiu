import numpy as np
from numpy.linalg import linalg
eps = 1e-9

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

def second_phase(a, c, x, jb, debug=False):
    iter_count = 0
    n = len(x)
    m = len(jb)
    ab = np.array([a[:, j].copy() for j in jb]).transpose()
    ab_inv = linalg.inv(ab)

    while True:
        iter_count += 1
        if debug:
            print("\nИтерация " + str(iter_count))
            print('Текущий план x = ' + str(x))
            print('Базис плана = ' + str(np.array([j + 1 for j in jb])))
            print('Текущее значение целевой функции - ' + str(c.dot(x)))
            print()
        cb = [c[i] for i in jb]
        u = np.dot(cb, ab_inv)
        ua = np.dot(u, a)
        delta = [ua[i] - c[i] for i in range(0, n)]

        is_optimal = True
        j0 = -1
        for i in range(len(delta)):
            if i not in jb:
                if delta[i] < 0:
                    is_optimal = False
                    j0 = i
                    break

        if is_optimal:
            if debug:
                print("Оптимальный план x: ", x)
                print("Базисные индексы: ", jb)
            return x, jb

        aj0 = [a[i][j0] for i in range(0, m)]
        z = np.dot(ab_inv, aj0)
        tetta = []
        for i in range(0, m):
            if z[i] > 0:
                tetta.append(x[jb[i]] / z[i])
            else:
                tetta.append(np.inf)

        tetta0 = min(tetta)
        tetta_index = tetta.index(tetta0)

        if tetta0 == np.inf:
            return None

        jb[tetta_index] = j0
        for i in range(0, n):
            if i not in jb:
                x[i] = 0

        x[j0] = tetta0
        for i in range(0, m):
            if i != tetta_index:
                x[jb[i]] = x[jb[i]] - tetta0 * z[i]
        new_col = [a[i][j0] for i in range(0, m)]
        ab_inv = np.array(inv_matrix(ab_inv, tetta_index, new_col, m))
        for i in range(0, m):
            ab[i][tetta_index] = a[i][j0]

def first_phase(a, b, c, debug=False):
    m, n = len(a), len(c)
    for i in range(m):
        if b[i] < 0:
            a[i] *= -1
            b[i] *= -1
    e = np.eye(m)
    na = np.hstack((a.copy(), e))#new a
    nc = np.array([0. if i < n else -1. for i in range(n + m)])
    jb = list(range(n, n + m))
    x = np.array([0. if i < n else b[i - n] for i in range(n + m)])
    x, jb = second_phase(na, nc, x, jb)
    if x[n:].max() > eps:
        return None

    while not (max(jb) < n):
        m = len(jb)
        nab = np.array([na[:, j] for j in jb]).transpose()
        nab_inv = linalg.inv(nab)
        for k in range(m):
            if jb[k] >= n:
                e_k = np.array([0]*m)
                e_k[k] = 1
                for i in range(n):
                    if i not in jb:
                        if abs(e_k.dot(nab_inv).dot(na[:, i])) > eps:
                            jb[k] = i
                            break
                else:
                    if debug:
                        print('Условие ' + str(jb[k] - n + 1) + ' линейно зависит от других')
                    jb_k = jb[k]
                    b = np.array([b[i] for i in range(m) if i != jb_k - n])
                    jb.remove(jb_k)
                    for i in range(len(jb)):
                        if jb[i] > jb_k:
                            jb[i] -= 1
                    na = np.array([na[i] for i in range(m) if i != jb_k - n])
                    na = np.array([na[:, i] for i in range(n + m) if i != jb_k]).transpose()
                break
    return na[:, :n], b, jb, x[:n]


def all_phases(a, b, c):
    first = first_phase(a, b, c, True)
    if first is None:
        print('Система несовместна')
        return
    a, b, jb, x = first
    second = second_phase(a, c, x, jb)
    if second is None:
        print('Неограничено')
        return
    print('оптимальный план ', second[0])
    print('базисные индексы', second[1])
    print('f ', np.dot(c, second[0]))


def main():
    c = [1, 1, 0, 0, 0]
    A = [[-1, 1, 1, 0, 0],
         [1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1],
         [0, 1, 0, 0, 1]]
    b = [1, 3, 2, 2]

    c = [0, 0, 0]
    A = [[1, 1, 1],
         [2, 2, 2]]
    b = [0, 0]

    c = [3.5, -1, 0, 0, 0]
    b = [15, 6, 0]
    A = [[5, -1, 1, 0, 0],
         [-1, 2, 0, 1, 0],
         [-7, 2, 0, 0, 1]]
    # c = [-5, -2, 3, -4, -6, 0, 1, -5]
    # A = [[0, 1, 4, 1, 0, -3, 1, 0],
    #      [1, -1, 0, 1, 0, 0, 0, 0],
    #      [0, 7, -1, 0, -1, 3, -1, 0],
    #      [1, 1, 1, 1, 0, 3, -1, 1]]
    # b = [6, 10, -2, 15]
    # x = [10, 0, 1.5, 0, 0.5, 0, 0, 3.5]
    # jb = [0, 3, 4, 7]

    # c = [1, 2, 3, -4]
    # A = [[1, 1, -1, 1],
    #      [1, 14, 10, -10]]
    # b = [2, 24, -2, 15]

    # c = [1, -5, -1, 1]
    # A = [[-1, -3, -3, -1],
    #      [2, 0, 3, -1]]
    # b = [3, 4]
    # несовместно

    # c = [1, 2, 1, -2, 1, -2]
    # A = [[1, -1, 1, -1, 1, -1],
    #      [2, 3, -2, -3, 2, 3],
    #      [3, 2, -1, -4, 3, 2]]
    # b = [7, 0, 10]

    # несовместна

    c = [7, -2, 6, 0, 5, 2]
    b = [-8, 22, 30]
    A = [[1, -5, 3, 1, 0, 0],
         [4, -1, 1, 0, 1, 0],
         [2, 4, 2, 0, 0, 1]]
    A = np.array(A)
    all_phases(A, b, c)

if __name__ == '__main__':
    main()
