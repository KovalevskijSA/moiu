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


def get_h_ast(j_ast, d, a):
    k, m = len(j_ast), len(a)
    h_ast = np.array([[0.] * (k + m) for _ in range(k + m)])
    for i in range(k):
        for j in range(k):
            h_ast[i][j] = d[j_ast[i]][j_ast[j]]

    for i in range(m):
        for j in range(k):
            h_ast[k+i][j] = a[i][j_ast[j]]

    for i in range(m):
        for j in range(k):
            h_ast[j][k+i] = a[i][j_ast[j]]

    return h_ast


def square(a, b, c, d, j_op, j_ast, x_n, debug=False):

    m, n = len(b), len(c)
    a_op = np.array(np.array([a[:,j] for j in j_op]).transpose())
    print(a_op)
    a_op_inv = linalg.inv(a_op)
    x = x_n
    cnt = 0
    while True:

        if debug:
            print("\n\nИтерация номер ", cnt)
            cnt += 1
            print("Текущий x = ", str(x))
            print("Текущая целевая функция равна", c.dot(x) + x.dot(d).dot(x) / 2)
            print("Текущий опорный план = ", str(j_op))
            print("Текущий правильный опорный план = ", str(j_ast))

        cx = c + d.dot(x)
        u = -np.array([cx[i] for i in j_op]).dot(a_op_inv)
        delta = np.array(u.dot(a) + cx)

        if debug:
            print("Текущий cx = ", str(cx))
            print("Текущий вектор потенциалов = ", str(u))
            print("Текущие оценки = ", str(delta))

        if delta.min() > -eps:
            print("Текущая целевая функция равна", c.dot(x) + x.dot(d).dot(x) / 2)
            print("Оптимальный план x = ", str(x))
            return
        # max->min
        j0 = min([x for x in range(n) if delta[x] < -eps])

        if debug:
            print("Текущий j0 = ", str(j0))

        l = np.array([0.] * n)
        l[j0] = 1

        h_ast = get_h_ast(j_ast, d, a)
        h_j0 = np.array([d[j_ast[x]][j0] for x in range(len(j_ast))] + [x for x in a[:, j0]])
        l_ast = -linalg.inv(h_ast).dot(h_j0)[:len(j_ast)]

        if debug:
            print("Текущая матрица H = \n", str(h_ast))
            print("Текущая обратная матрица H^(-1) = \n", str(linalg.inv(h_ast)))
            print("Текущий вектор h_j0 = ", str(h_j0))
            print("Текущий вектор l_ast = ", str(l_ast))

        for i in range(len(j_ast)):
            l[j_ast[i]] = l_ast[i]

        if debug:
            print("Текущий l = ", str(l))

        theta_j = np.array([np.inf if l[j] > -eps else -x[j] / l[j] for j in j_ast])
        delta_d = l.dot(d).dot(l)
        theta_j0 = np.inf if abs(delta_d) < eps else abs(delta[j0]) / delta_d

        if debug:
            print("Текущий theta_j = ", str(theta_j))
            print("Текущий delta_d = ", str(delta_d))
            print("Текущий theta_j0 = ", str(theta_j0))

        theta_0 = min([theta_j.min(), theta_j0])
        if theta_0 == np.inf:
            print("Функция неогранияена снизу")
            return
        s = -1

        for i in range(n):
            if i == j0:
                if abs(theta_0 - theta_j0) < eps:
                    s = j0
                    break
            elif i in j_ast:
                if abs(theta_0 - (-x[i] / l[i])) < eps:
                    s = i
                    break

        x = x + theta_0 * l

        if debug:
            print("Текущий s = ", str(s))

        if s == j0:
            j_ast.append(j0)
        elif s in j_ast and s not in j_op:
            j_ast.remove(s)
        else:
            index = j_op.index(s)
            for i in range(n):
                if i not in j_op and i in j_ast and abs(a_op_inv.dot(a[:, i])[index]) > eps:
                    j_op[index] = i
                    j_ast.remove(s)
                    a_op_inv = inv_matrix(a_op_inv, index, a[:,i], m)
                    break
            else:
                j_op[index] = j0
                j_ast[j_ast.index(s)] = j0
                a_op_inv = inv_matrix(a_op_inv, index, a[:, j0], m)


def main():
    a = np.array(
        [[1, 2, 0, 1, 0, 4, -1, -3],
         [1, 3, 0, 0, 1, -1, -1, 2],
         [1, 4, 1, 0, 0, 2, -2, 0]
         ]
    )
    b = np.array([4, 5, 6])

    c = np.array([-10, -31, 7, 0, -21, -16, 11, -7])

    b1 = np.array([[1, 1, -1, 0, 3, 4, -2, 1],
                   [2, 6, 0, 0, 1, -5, 0, -1],
                   [-1, 2, 0, 0, -1, 1, 1, 1]])
    d1 = np.array([7, 3, 3])
    x_n = np.array([0, 0, 6, 4, 5, 0, 0, 0])
    j_op = [2, 3, 4] # 3 4 5
    j_ast = [2, 3, 4]
#p1
    # a = np.array(
    #     [[11, 0, 0, 1, 0, -4, -1, 1],
    #      [1, 1, 0, 0, 1, -1, -1, 1],
    #      [1, 1, 1, 0, 1, 2, -2, 1]
    #      ]
    # )
    # b = np.array([8, 2, 5])
    #
    # b1 = np.array([[1, -1, 0, 3, -1, 5, -2, 1],
    #                [2, 5, 0, 0, -1, 4, 0, 0],
    #                [-1, 3, 0, 5, 4, -1, -2, 1]])
    # d1 = np.array([6, 10, 9])
    # x_n = np.array([0.7273, 1.2727, 3., 0, 0, 0, 0, 0])
    # j_op = [0, 1, 2]  # 3 4 5
    # j_ast = [0, 1, 2]
    #
    #
    # c = -d1.transpose().dot(b1)
    # d = b1.transpose().dot(b1)
#2
    # a = np.array(
    #     [[2, -3, 1, 1, 3, 0, 1, 2],
    #      [-1, 3, 1, 0, 1, 4, 5, -6],
    #      [1, 1, -1, 0, 1, -2, 5, 8]
    #      ]
    # )
    # b = np.array([8, 4, 14])
    #
    # b1 = np.array([[1, 0, 0, 3, -1, 5, 0, 1],
    #                [2, 5, 0, 0, 0, 4, 0, 0],
    #                [-1, 9, 0, 5, 2, -1, -1, 5]])
    # d1 = np.array([6, 10, 9])
    # x_n = np.array([0, 2, 0, 0, 4, 0, 0, 1])
    # j_op = [1, 4, 7]  # 3 4 5
    # j_ast = [1, 4, 7]
    #
    # c = np.array([-13, -217, 0, -117, -27, -71, 18, -99])
    d = b1.transpose().dot(b1)
    square(a, b, c, d, j_op, j_ast, x_n, debug=True)
    # x = np.array([0.2977, 1.0404, 5.368, 0, -0, 1.3007, 0.7599, 2.199])
    # print(c.dot(x) + (x.transpose()).dot(d).dot(x)/2)

if __name__ == "__main__":
    main()