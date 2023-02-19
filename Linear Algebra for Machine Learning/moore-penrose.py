import numpy.random


def get_matrix():
    print('Enter the dimensions :')
    n = int(input())
    m = int(input())
    return numpy.random.randint(-1000, 1000, (n, m))

def compare_mat(m1, m2):
    print("Matrix by library function: ")
    print(m1)
    print("\n\n")
    print("Matrix by equation: ")
    print(m2)
    print("\n\n")
    if (numpy.allclose(m1, m2)):
        print("Pseudoinverse matrix matched!")
    else:
        print("Pseudoinverse matrix didn't match.")


def moore_penrose(matrix):
    U, D, Vt = numpy.linalg.svd(matrix, full_matrices=False)
    D = numpy.diag(D)
    D[D != 0] = 1 / D[D != 0]
    return numpy.linalg.pinv(matrix), numpy.matmul(numpy.matmul(Vt.T, D.T), U.T)

def main():
    pseudoinverse_by_lib, pseudoinverse_by_eqn = moore_penrose(get_matrix())
    compare_mat(pseudoinverse_by_lib, pseudoinverse_by_eqn)


if __name__ == "__main__":
    main()