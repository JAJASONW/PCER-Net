import numpy as np

EPS = 1e-10

def norm_n(N, eps=1e-10):
    N_mode = (N[0] ** 2 + N[1] ** 2 + N[2] ** 2) ** 0.5 + eps
    return N / N_mode

def Onedirection_get_intersection(a, b, c, d, e, f, g, h, C1, C2, r):
    Fi = b * g - f * c
    P = [1, ((d - a) * g - c * (h - e)) / (Fi), -((d - a) * f - (h - e) * b) / (Fi)]
    direction = [1, (c * e - g * a) / (Fi), (f * a - b * e) / (Fi)]

    direction = norm_n(direction)

    b_1 = 2 * direction[0] * (P[0] - C1[0]) + 2 * direction[1] * (P[1] - C1[1]) + 2 * direction[2] * (
                P[2] - C1[2])  # t^2+bt+c=0  have two intersection points

    c_1 = (P[0] - C1[0]) ** 2 + (P[1] - C1[1]) ** 2 + (P[2] - C1[2]) ** 2 - r ** 2

    b_2 = 2 * direction[0] * (P[0] - C2[0]) + 2 * direction[1] * (P[1] - C2[1]) + 2 * direction[2] * (
                P[2] - C2[2])  # t^2+bt+c=0  have two intersection points
    c_2 = (P[0] - C2[0]) ** 2 + (P[1] - C2[1]) ** 2 + (P[2] - C2[2]) ** 2 - r ** 2
    Delta1 = b_1 ** 2 - 4 * c_1
    Delta2 = b_2 ** 2 - 4 * c_2

    if Delta1 < 0 or Delta2 < 0:
        Point_left = np.array([0, 0, 0])
        Point_right = np.array([0, 0, 0])

        return Point_left, Point_right

    t_11 = (-b_1 + (b_1 ** 2 - 4 * c_1) ** 0.5) / 2
    t_12 = (-b_1 - (b_1 ** 2 - 4 * c_1) ** 0.5) / 2
    Point11 = P + t_11 * direction
    Point12 = P + t_12 * direction
    # print('t11=',t_11,'t12=',t_12)
    # print('Point11=',Point11,'Point12=',Point12)

    t_21 = (-b_2 + (b_2 ** 2 - 4 * c_2) ** 0.5) / 2
    t_22 = (-b_2 - (b_2 ** 2 - 4 * c_2) ** 0.5) / 2
    Point21 = P + t_21 * direction
    Point22 = P + t_22 * direction
    # print('t21=',t_21,'t22=',t_22)
    # print('Point21=',Point21,'Point22=',Point22)

    if np.min(np.array([t_11, t_12])) > np.max(np.array([t_21, t_22])) or np.min(np.array([t_21, t_22])) > np.max(
            np.array([t_11, t_12])):
        # print('two circles have no intersection!')
        Point_left = np.array([0, 0, 0])
        Point_right = np.array([0, 0, 0])
        # Point_left = np.array([np.nan,np.nan,np.nan])
        # Point_right = np.array([np.nan,np.nan,np.nan])
        return Point_left, Point_right

    t_list = [t_11, t_12, t_21, t_22]
    for i in range(4):
        for j in range(3):
            if t_list[j] > t_list[j + 1]:
                t_list[j], t_list[j + 1] = t_list[j + 1], t_list[j]

    Point_left = P + t_list[1] * direction
    Point_right = P + t_list[2] * direction
    # print('Point_left =', Point_left, 'Point_right=', Point_right)
    return Point_left, Point_right

def get_intersection(C1, N1, C2, N2, r):
    a = N1[0]
    b = N1[1]
    c = N1[2]
    d = N1[0] * C1[0] + N1[1] * C1[1] + N1[2] * C1[2]

    e = N2[0]
    f = N2[1]
    g = N2[2]
    h = N2[0] * C2[0] + N2[1] * C2[1] + N2[2] * C2[2]

    if ((abs(N1[0])) - abs(N2[0])) ** 2 + ((abs(N1[1])) - abs(N2[1])) ** 2 + (
            (abs(N1[2])) - abs(N2[2])) ** 2 < EPS ** 2:
        # for the situation : N1 // N2
        Point_left = np.array([0, 0, 0])
        Point_right = np.array([0, 0, 0])
        return Point_left, Point_right
    if abs(b * g - f * c) < EPS:
        if abs(a * f - e * b) < EPS:
            # z
            T = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            TR = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            C_t1 = T @ C1
            C_t2 = T @ C2
            n1 = T @ N1
            n2 = T @ N2
            Pointl, Pointr = Onedirection_get_intersection(n1[0], n1[1], n1[2], d, n2[0], n2[1], n2[2], h, C_t1, C_t2,
                                                           r)
            Point_left = TR @ Pointl
            Point_right = TR @ Pointr
            return Point_left, Point_right
        else:
            # y
            T = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            TR = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            C_t1 = T @ C1
            C_t2 = T @ C2
            n1 = T @ N1
            n2 = T @ N2
            Pointl, Pointr = Onedirection_get_intersection(n1[0], n1[1], n1[2], d, n2[0], n2[1], n2[2], h, C_t1, C_t2,
                                                           r)
            Point_left = TR @ Pointl
            Point_right = TR @ Pointr
            return Point_left, Point_right
        # denominator -> 0

    Pointl, Pointr = Onedirection_get_intersection(a, b, c, d, e, f, g, h, C1, C2, r)
    # print("left:{} right:{}", Pointl, Pointr)
    return Pointl, Pointr


if __name__ == '__main__':
    C1 = np.array([1, 0.002, 0.002])
    N1 = np.array([0.002, 0.002, 1])
    C2 = np.array([0.002, 0.002, 1])
    N2 = np.array([1, 0.002, 0.002])
    r = 1
    get_intersection(C1, N1, C2, N2, r)
