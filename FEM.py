import numpy as np
import math
import scipy


def generate_Gauss_reference_1D(Gauss_point_number):
    global Gauss_point_reference_1D, Gauss_coefficient_reference_1D
    if Gauss_point_number == 4:
        Gauss_coefficient_reference_1D = [0.3478548451, 0.3478548451, 0.6521451549, 0.6521451549]
        Gauss_point_reference_1D = [0.8611363116, -0.8611363116, 0.3399810436, -0.3399810436]

    elif Gauss_point_number == 8:
        Gauss_coefficient_reference_1D = [0.1012285363, 0.1012285363, 0.2223810345, 0.2223810345, 0.3137066459,
                                          0.3137066459, 0.3626837834, 0.3626837834]
        Gauss_point_reference_1D = [0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774, 0.5255324099,
                                    -0.5255324099, 0.1834346425, -0.1834346425]
    elif Gauss_point_number == 2:
        Gauss_coefficient_reference_1D = [1, 1]
        Gauss_point_reference_1D = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
    # print(Gauss_coefficient_reference_1D)
    return Gauss_coefficient_reference_1D, Gauss_point_reference_1D


def generate_M_T_1D(left, right, h_partition, basis_type):
    h = h_partition
    if basis_type == 101:
        N = int((right - left) / h)
        # print(N)
        M = np.zeros((1, N + 1))
        T = np.zeros((2, N))

        for i in range(1, N + 2):
            M[0][i - 1] = left + (i - 1) * h

        for i in range(1, N + 1):
            T[0][i - 1] = int(i)
            T[1][i - 1] = int(i + 1)   # matrix needs modify
    elif basis_type == 102:
        N = int((right - left) / h)
        dh = h / 2
        dN = N * 2
        M = np.zeros((1, dN + 1))
        T = np.zeros((2, dN))

        for i in range(1, dN + 2):
            M[1][i-1]  = left + (i - 1) * dh

        for i in range(1, N + 1):
            T[0][i-1] = 2 * i - 1
            T[1][i-1] = 2 * i + 1
            T[2][i-1] = 2 * i

    return M, T



def generate_Gauss_local_1D(Gauss_coefficient_reference_1D, Gauss_point_reference_1D, lower_bound, upper_bound):
    # print(type(Gauss_coefficient_reference_1D))
    Gauss_coefficient_local_1D = [(upper_bound - lower_bound) / 2 * x for x in Gauss_coefficient_reference_1D]
    Gauss_point_local_1D = [(upper_bound - lower_bound) / 2 * x + (upper_bound + lower_bound) / 2 for x in
                            Gauss_point_reference_1D]  # + [
    # (upper_bound + lower_bound) / 2]
    return Gauss_coefficient_local_1D, Gauss_point_local_1D


def local_basis_1D(x, vertices, basis_type, basis_index, derivative_degree):
    if basis_type == 101:

        if derivative_degree == 0:

            if basis_index == 1:
                result = (vertices[0][1] - x) / (vertices[0][1] - vertices[0][0])
            elif basis_index == 2:
                result = (x - vertices[0][0]) / (vertices[0][1] - vertices[0][0])

        elif derivative_degree == 1:

            if basis_index == 1:
                result = 1 / (vertices[0][0] - vertices[0][1])
            elif basis_index == 2:
                result = 1 / (vertices[0][1] - vertices[0][0])
    return result


def Gauss_quadrature_for_1D_integral_trial_test(coefficient_function_name, Gauss_coefficient_local_1D,
                                                Gauss_point_local_1D,
                                                trial_vertices, trial_basis_type, trial_basis_index,
                                                trial_derivative_degree,
                                                test_vertices, test_basis_type, test_basis_index,
                                                test_derivative_degree):
    Gpn = len(Gauss_coefficient_local_1D)
    # print(Gpn)

    result = 0
    for i in range(1, Gpn + 1):
        # function_a
        # coefficient_function_name = np.exp
        result += Gauss_coefficient_local_1D[i - 1] * coefficient_function_name(
            Gauss_point_local_1D[i - 1]) * local_basis_1D(
            Gauss_point_local_1D[i - 1], trial_vertices, trial_basis_type, trial_basis_index,
            trial_derivative_degree) * local_basis_1D(Gauss_point_local_1D[i - 1], test_vertices, test_basis_type,
                                                      test_basis_index, test_derivative_degree)
    return result


def assemble_matrix_from_1D_integral(coefficient_function_name, M_partition, T_partition, T_basis_trial, T_basis_test,
                                     number_of_trial_local_basis, number_of_test_local_basis, number_of_elements,
                                     matrix_size, Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
                                     trial_basis_type, trial_derivative_degree, test_basis_type,
                                     test_derivative_degree):
    result = np.zeros((matrix_size[0], matrix_size[1]))  # sparse
    # print(result)
    for n in range(1, number_of_elements + 1):

        vertices = M_partition[:, (T_partition[:, n - 1]).astype(int)-1]
        # print(vertices)
        lower_bound = min(vertices[0][0], vertices[0][1])
        upper_bound = max(vertices[0][0], vertices[0][1])
        [Gauss_coefficient_local_1D, Gauss_point_local_1D] = generate_Gauss_local_1D(Gauss_coefficient_reference_1D,
                                                                                     Gauss_point_reference_1D,
                                                                                     lower_bound, upper_bound)
        # print(Gauss_point_local_1D)

        for alpha in range(1, number_of_trial_local_basis + 1):
            for beta in range(1, number_of_test_local_basis + 1):
                temp = Gauss_quadrature_for_1D_integral_trial_test(coefficient_function_name,
                                                                   Gauss_coefficient_local_1D, Gauss_point_local_1D,
                                                                   vertices, trial_basis_type, alpha,
                                                                   trial_derivative_degree, vertices, test_basis_type,
                                                                   beta, test_derivative_degree)
                # print(temp)
                result[T_basis_test[beta - 1, n - 1].astype(int)-1][T_basis_trial[alpha - 1, n - 1].astype(int)-1] += temp
    return result


def Gauss_quadrature_for_1D_integral_test(coefficient_function_name, Gauss_coefficient_local_1D, Gauss_point_local_1D,
                                          test_vertices, test_basis_type,
                                          test_basis_index, test_derivative_degree):
    Gpn = len(Gauss_coefficient_local_1D)

    result = 0
    for i in range(1, Gpn + 1):
        result = result + Gauss_coefficient_local_1D[i - 1] * coefficient_function_name(
            Gauss_point_local_1D[i - 1]) * local_basis_1D(
            Gauss_point_local_1D[i - 1],
            test_vertices, test_basis_type, test_basis_index, test_derivative_degree)
    return result


def assemble_vector_from_1D_integral(coefficient_function_name, M_partition, T_partition, T_basis_test,
                                     number_of_test_local_basis, number_of_elements,
                                     vector_size, Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
                                     test_basis_type, test_derivative_degree):
    result = np.zeros((vector_size, 1))
    for n in range(1, number_of_elements + 1):

        vertices = M_partition[:, T_partition[:, n - 1].astype(int)-1]
        lower_bound = min(vertices[0][0], vertices[0][1])
        upper_bound = max(vertices[0][0], vertices[0][1])
        [Gauss_coefficient_local_1D, Gauss_point_local_1D] = generate_Gauss_local_1D(Gauss_coefficient_reference_1D,
                                                                                     Gauss_point_reference_1D,
                                                                                     lower_bound, upper_bound)

        for beta in range(1, number_of_test_local_basis + 1):
            temp = Gauss_quadrature_for_1D_integral_test(coefficient_function_name, Gauss_coefficient_local_1D,
                                                         Gauss_point_local_1D, vertices, test_basis_type, beta,
                                                         test_derivative_degree)
            result[T_basis_test[beta - 1, n - 1].astype(int)-1][0] += temp
    return result


def generate_boundary_nodes_1D(N_basis):
    boundary_nodes = np.zeros((3, 2))
    boundary_nodes[0][0] = -1
    boundary_nodes[1][0] = 1
    boundary_nodes[2][0] = -1
    boundary_nodes[0][1] = -1
    boundary_nodes[1][1] = N_basis + 1
    boundary_nodes[2][1] = 1
    # print(boundary_nodes)
    return boundary_nodes


def treat_Dirichlet_boundary_1D(Dirichlet_boundary_function_name, A, b, boundary_nodes, M_basis):
    nbn = boundary_nodes.shape[1]
    # print(nbn)
    for k in range(1, nbn + 1):
        if boundary_nodes[0, k - 1] == -1:
            i = int(boundary_nodes[1][k-1])
            A[i-1][:] = 0
            A[i-1][i-1] = 1
            # print(M_basis[0][i - 1])
            b[i-1][0] = Dirichlet_boundary_function_name(M_basis[0][i - 1])

    return A, b


def function_g(xX):
    left = 0
    right = 1
    # result = np.zeros_like(x)
    # result = np.where(x == left, 0, result)
    # result = np.where(x == right, np.cos(1), result)
    if xX <= 0:
        result = 0
    elif xX >= 1:
        result = np.cos(1)
    # print(result)
    return result


def function_a(x):
    result = np.exp(x)
    return result


def function_f(x):
    result = -math.exp(x) * (np.cos(x) - 2 * np.sin(x) - x * np.cos(x) - x * np.sin(x))
    return result


def poisson_solver_1D(left, right, h_partition, basis_type, Gauss_point_number):
    global number_of_trial_local_basis
    N_partition = int((right - left) / h_partition)
    # print(N_partition)

    if basis_type == 101:
        N_basis = N_partition
    elif basis_type == 102:
        N_basis = N_partition * 2

    [M_partition, T_partition] = generate_M_T_1D(left, right, h_partition, 101)
    # print(M_partition)

    if basis_type == 101:
        M_basis = M_partition
        T_basis = T_partition
    elif basis_type == 102:
        [M_basis, T_basis] = generate_M_T_1D(left, right, h_partition, 102)

    # print(M_basis)

    [Gauss_coefficient_reference_1D, Gauss_point_reference_1D] = generate_Gauss_reference_1D(Gauss_point_number)
    # print(Gauss_point_reference_1D)

    number_of_elements = N_partition
    # print(number_of_elements)
    matrix_size = [N_basis + 1, N_basis + 1]
    vector_size = N_basis + 1

    if basis_type == 101:
        number_of_trial_local_basis = 2
        number_of_test_local_basis = 2

    # def fuction_a(x):
    #     np.exp(x)

    A = assemble_matrix_from_1D_integral(function_a, M_partition, T_partition, T_basis, T_basis,
                                         number_of_trial_local_basis, number_of_test_local_basis, number_of_elements,
                                         matrix_size, Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
                                         basis_type, 1, basis_type, 1)
    # print(A)
    b = assemble_vector_from_1D_integral(function_f, M_partition, T_partition, T_basis, number_of_test_local_basis,
                                         number_of_elements,
                                         vector_size, Gauss_coefficient_reference_1D, Gauss_point_reference_1D,
                                         basis_type, 0)
    # print(b)
    boundary_nodes = generate_boundary_nodes_1D(N_basis)
    [A, b] = treat_Dirichlet_boundary_1D(function_g, A, b, boundary_nodes, M_basis)
    # print(b)

    result = np.linalg.solve(A, b)
    # print(result.shape)   (5,1)

    if basis_type == 101:
        h_basis = h_partition

    def get_maximum_error_1D(solution, N_basis, left, h_basis):
        maxerror = 0
        for i in range(1, N_basis + 2):
            x = left + (i - 1) * h_basis
            temp = solution[i-1]-(x*np.cos(x))
            if abs(maxerror) < abs(temp):
                maxerror = temp
        return maxerror

    max_error = get_maximum_error_1D(result, N_basis, left, h_basis)
    print(max_error)
    # print(result)
    return result


if __name__ == "__main__":
    left = 0
    right = 1
    # bottom=0
    # top=1
    basis_type = 101
    Gauss_point_number = 4
    h_partition = 1 / 4
    poisson_solver_1D(left, right, h_partition, basis_type, Gauss_point_number)

    h_partition = 1 / 8
    poisson_solver_1D(left, right, h_partition, basis_type, Gauss_point_number)

    h_partition = 1 / 16
    poisson_solver_1D(left, right, h_partition, basis_type, Gauss_point_number)

    h_partition = 1 / 32
    poisson_solver_1D(left, right, h_partition, basis_type, Gauss_point_number)

