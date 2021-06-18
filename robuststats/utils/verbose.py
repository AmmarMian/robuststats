
def matprint(mat, fmt="g"):
    """ Pretty print a matrix in Python 3 with numpy.
    Source: https://gist.github.com/lbn/836313e283f5d47d2e4e
    """

    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
