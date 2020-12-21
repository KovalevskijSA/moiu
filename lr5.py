import numpy as np
import copy

def balance_check(a, b, c):
    sum_a = sum(a)
    sum_b = sum(b)
    if sum_a == sum_b:
        return a, b, c

    m, n = c.shape
    if sum_a > sum_b:
        b.append(sum_a - sum_b)
        c = np.append(c, np.zeros((m, 1)), axis=1)
        return a, b, c

    if sum_b > sum_a:
        a.append(sum_b - sum_a)
        c =  np.append(c, np.zeros((1, n)), axis=0)
        return a, b, c


def plan_builder(a , b, c):
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    m, n = len(a), len(b)
    plan = np.array([[0] * n for _ in range(m)])
    a_p, b_p = 0, 0
    j_b = set()
    while a_p < m and b_p < n:
        j_b.add((a_p, b_p))
        if a[a_p] < b[b_p]:
            b[b_p] -= a[a_p]
            plan[a_p][b_p] = a[a_p]
            a_p += 1
        else:
            a[a_p] -= b[b_p]
            plan[a_p][b_p] = b[b_p]
            b_p += 1

    for x in range(m):
        for y in range(n):
            if (x, y) not in j_b and len(j_b) < m + n - 1:
                j_b.add((x, y))
                s = get_cycle(m, n, j_b)
                if sum([sum(i) for i in s]) != 0:
                    j_b.remove((x, y))
    return plan, list(j_b)


def get_potentials(m, n, c, j_b):
    u, v = [None] * m, [None] * n
    v[n - 1] = 0
    for _ in range(m + n):
        for x, y in j_b:
            if u[x] is not None and v[y] is None:
                v[y] = c[x][y] - u[x]
                break
            elif u[x] is None and v[y] is not None:
                u[x] = c[x][y] - v[y]
                break
    return u, v

def get_cycle(m, n, j_b):
    mat = np.zeros((m, n))
    for i, j in j_b:
        mat[i, j] = 1
    c = True
    while c:
        c = False
        for i, row in enumerate(mat):
            if sum(row) != 1:
                continue
            mat[i] = np.zeros(n)
            c = True
            print('\n',mat)
        for j in range(n):
            if sum(mat[:, j]) != 1:
                continue
            mat[:, j] = np.zeros(m)
            print('\n', mat)
            c = True
    return mat


def update_plan(plan, cycle, j_b, i0, j0):
    m, n = plan.shape
    cycle_c = copy.deepcopy(cycle)
    ni = i = i0
    nj = j = j0

    min_i = i0
    min_j = j0
    min = np.inf
    l = 1
    k_n = int(sum([sum(i) for i in cycle]))
    for _ in range(k_n):
        cycle_c[i, j] = l
        cycle[i, j] = 0
        if l == -1 and plan[i, j] < min:
            min = plan[i, j]
            min_i = i
            min_j = j

        l = -l
        for jj in range(0, n):
            if cycle[i, jj] == 1:
                j = jj
                break
        else:
            for ii in range(0, m):
                if cycle[ii, j] == 1:
                    i = ii
                    break
    print('min = {} (min_i, min_j) = {}'.format(min, (min_i, min_j)))
    cycle = cycle_c*min
    print(cycle)
    plan = plan + cycle
    j_b.remove((min_i, min_j))
    return plan, j_b


def solve_tt(a, b, c):
    a, b, c = balance_check(a, b, c)
    print('a', a)
    print('b', b)
    m, n = c.shape
    plan, j_b = plan_builder(a, b, c)
    print('plan', plan)
    print('j_b', j_b)

    j_n = [(i, j) for i in range(m) for j in range(n) if (i, j) not in j_b]
    iter = 1
    while True:
        print('iter ', iter)
        print('plan ', plan)
        print('j_b ', j_b)
        u, v = get_potentials(m, n, c, j_b)
        print("u ", u)
        print("v ", v)
        delta =  np.array([[c[i][j] - u[i] - v[j] for j in range(n)] for i in range(m)])
        print('delta ', delta)
        if delta.min() >= 0:
            print("optimal plan ", plan)
            print('j_b ', j_b)
            s = 0
            for i in range(m):
                for j in range(n):
                    s += plan[i, j] * c[i, j]
            print("fun ", s)
            return

        def get_min():
            min = 0
            i0 = 0
            j0 = 0
            for i in range(m):
                for j in range(n):
                    if delta[i][j] < min and (i, j) not in j_b:
                        min = delta[i][j]
                        i0 = i
                        j0 = j
            return i0, j0
        i0, j0 = get_min()
        # i0, j0 = min([(i, j) for i in range(m) for j in range(n) if delta[i][j] < 0 and (i, j) not in j_b])
        print("imin ", i0)
        print("jmin ", j0)
        j_b.append((i0, j0))
        cycle = get_cycle(m, n, j_b)
        print('cycle', cycle)
        plan, j_b = update_plan(plan, cycle, j_b, i0, j0)
        iter+=1
        print('-'*20)

def main():
    a = [20, 30, 25]
    b = [10, 10, 10, 10, 10]

    c = np.array([
      [2, 8, -5, 7, 10],
      [11, 5, 8, -8, -4],
      [1, 3, 7, 4, 2]
    ])

    # a = [20,11,18,27]
    # b = [11,4,10,12,8,9,10,4]
    #
    # c = np.array([
    #   [-3, 6, 7, 12, 6, -3, 2, 16],
    #   [4, 3, 7, 10, 0, 1, -3, 7],
    #   [19, 3, 2, 7, 3, 7, 8, 15],
    #   [1, 4, -7, -3, 9, 13, 17, 22]
    # ])
    #
    # a = [15,12,18,20]
    # b = [5,5,10,4,6,20,10,5]
    #
    # c = np.array([
    #   [3,10,70,-3,7,4,2,-20],
    #   [3,5,8,8,0,1,7,-10],
    #   [-15, 1, 0, 0, 13, 5, 4, 5],
    #   [1, -5, 9, -3, -4, 7, 16, 25]
    # ])
    #
    # a = [53, 20, 45, 38]
    # b = [15, 31, 10, 3, 18]
    #
    # c = np.array([
    #   [3, 0, 3, 1, 6],
    #   [2, 4, 10, 5, 7],
    #   [-2, 5, 3, 2, 9],
    #   [1, 3, 5, 1, 9]
    # ])

    # a = [0, 0]
    # b = [0, 0]

    # c = np.array([
    #     [0, 0],
    #     [0, 0],
    # ])


    # a = [0, 0, 0, 0]
    # b = [0, 0, 0]
    #
    # c = np.array([
    #     [0, 0, 0],
    #     [0, 0, 0],
    #     [0, 0, 0],
    #     [0, 0, 0]
    # ])

    # a = [0, 0, 0]
    # b = [0, 0, 0]
    #
    # c = np.array([
    #     [0, 0, 0],
    #     [0, 0, 0],
    #     [0, 0, 0]
    # ])

    a = [30, 20, 10, 10]
    b = [35, 5, 20, 5, 5]

    c = np.array([
        [5, 0, 4, 0, 4],
        [2, 2, -3, 3, 0],
        [-2, -5, 0, -3, 3],
        [1, 1, 3, 5, -5]
    ])
    solve_tt(a, b, c)

if __name__ == '__main__':
    main()
