import numpy.random

def get_matrix():
    print('Enter the dimension:')
    n = int(input())
    matrix = numpy.random.randint(-1000, 1000, (n, n))
    while(numpy.linalg.det(matrix)==0):
        matrix = numpy.random.randint(-1000, 1000, (n, n))
    return matrix

def random_eigen(matrix):
    print("Original Matrix: ")
    print(matrix)
    print("\n\n")
    l, V = numpy.linalg.eig(matrix)
    reconstructed_matrix = numpy.matmul(numpy.matmul(V, numpy.diag(l)), numpy.linalg.inv(V))
    print("Reconstructed Matrix: ")
    print(reconstructed_matrix)
    print("\n\n")
    if(numpy.allclose(matrix, reconstructed_matrix)):
        print("Matrix matched!")
    else:
        print("Matrix didn't match.")



def main():
    random_eigen(get_matrix())


    
if __name__ == "__main__":
    main()