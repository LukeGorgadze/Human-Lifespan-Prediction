import numpy as np


def ModGramSchmidt(A):
    """
    For (n x m) of full rank matrices, meaning n > m,
    there exists Q^(n x m) with orthonormal columns and upper triangular R with (m x m) 
    such that A = QR - Credits to Prof. Ramaz
    """

    m = A.shape[1]
    q = np.zeros(A.shape)
    r = np.zeros((m, m))

    for j in range(0, m):
        # y = A_j
        y = []
        for rndx in range(0, A.shape[0]):
            y.append(A[rndx][j])
        y = np.array(y)

        for i in range(0, j):
            qCol = []
            for rndx in range(0, A.shape[0]):
                qCol.append(q[rndx][i])
            qCol = np.array(qCol)
            r[i][j] = np.matmul(qCol, y)
            y = y - r[i][j] * qCol

        r[j][j] = np.linalg.norm(y, ord=2)
        if r[j][j] != 0:
            qCol = y / r[j][j]
        else:
            qCol = y 

        for rndx in range(0, A.shape[0]):
            q[rndx][j] = qCol[rndx]

    return (q, r)


def NormalEquation(A, b):
    # AT Ax = AT b
    At = np.transpose(A)
    AtA = At@A
    AtAi = np.linalg.inv(AtA)
    x = AtAi @ At @ b
    return x


def ProjectionMat(u):
    return 2 * (np.outer(u,u)) / (u.T @ u)

def TurnPeopleIntoMatrix(people):
    """
    This function takes the list of Human objects and returns a matrix A 
    """
    A = []
    b = []
    for human in people:
        row = [human.getMidAge(),human.getSleepTime(),human.getWorkoutHrs()]
        A.append(row)
        b.append(human.getLifeSpan())
    b = np.array(b)
    A = np.array(A)
    return (A,b)

def TurnPeopleIntoMatrix2(people):
    """
    This function takes the list of Human objects and returns a matrix A 
    """
    A = []
    b = []
    for human in people:
        row = [human.getAge(),human.getWorkHrs(),human.getStudyHrs(),human.getTechIndex()]
        A.append(row)
        b.append(human.getIncome())
    b = np.array(b)
    A = np.array(A)
    return (A,b)

def TurnHumanIntoMatrix(human):
    A = [human.getMidAge(),human.getSleepTime(),human.getWorkoutHrs()]
    return np.array(A)

def TurnHumanIntoMatrix2(human):
    A = [human.getAge(),human.getWorkHrs(),human.getStudyHrs(),human.getTechIndex()]
    return np.array(A)

def FixMatrix(A):
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            if abs(A[r][c]) < 1e-10:
                A[r][c] = 0
    return A

def HouseHolder(A):
    colNum = A.shape[1]
    Atranspose = A.T
    HList = []
    Hdimension = Atranspose.shape[1]
    counter = 0
    while counter < colNum:
        n = Atranspose.shape[1]
        x = Atranspose[0]
        xLen = np.linalg.norm(x,ord=2)
        wAux = np.array([0] * n)
        wAux[0] = 1
        w = xLen * wAux
        v = w - x
        H = np.eye(n) - ProjectionMat(v)

        # Make H Hdimensional
        if H.shape[0] < Hdimension:
            zeroMat = np.zeros((Hdimension,Hdimension))
            for i in range(0,zeroMat.shape[0]):
                for j in range(0,zeroMat.shape[0]):
                    if i == j:
                        zeroMat[i][j] = 1

            offset = Hdimension - H.shape[0]
            for i in range(offset, zeroMat.shape[0]):
                for j in range(offset,zeroMat.shape[0]):
                    zeroMat[i][j] = H[i-offset][j-offset]
            MyH = zeroMat.copy()
            HList.append(MyH)
        else:
            MyH = H
            HList.append(H)

        An = (H @ Atranspose.T )
        An = FixMatrix(An)

        # Remove First row and then first column
        AnReduced = np.delete(An.T,0,0)
        newAt = []
        for r in range(AnReduced.shape[0]):
            row = []
            for c in range(1,AnReduced.shape[1]):
                row.append(AnReduced[r][c])
            newAt.append(row)
        newAt = np.array(newAt)
        newAt = FixMatrix(newAt)
        if (newAt.shape[0] == 1 and newAt.shape[1] == 1):
            break
        Atranspose = newAt 
        counter += 1

    R = A
    for h in HList:
        R = h @ R

    R = FixMatrix(R)
    QT = HList[0]
    for i in range(1,len(HList)):
        QT = HList[i] @ QT

    Q = QT.T
    
    return (Q,R)



