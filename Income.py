import matplotlib.pyplot as plt
import random
import numpy as np
from LukaUtils import ModGramSchmidt, NormalEquation, HouseHolder, TurnHumanIntoMatrix2, TurnPeopleIntoMatrix2

# Constants are regenerated every time
ageC = random.random() * 1 + 1
workoutC = random.random() * 10 + 1
studyC = random.random()  + 0.1
techC = random.random() * 10 + 1


class Human:
    """
    The Human class is defined, with a constructor method that 
    initializes the various attributes of a Human 
    object (age, name, gender, work hours, study hours, tech index, and income). 
    There are also several methods defined to get the values of these attributes 
    and to calculate the income of a Human based on a formula using the other attributes.
    """

    def __init__(self, age, fullName, gender, workHrs, studyHrs, techIndex):
        if gender == "F":
            self.gender = "Female"
        else:
            self.gender = "Male"
        self.age = age
        self.fullName = fullName
        self.workHrs = workHrs
        self.studyHrs = studyHrs
        self.techIndex = techIndex
        self.calculateIncome()

    def getGender(self):
        return self.gender

    def getFullName(self):
        return self.fullName

    def getWorkHrs(self):
        return self.workHrs

    def getAge(self):
        return self.age

    def getStudyHrs(self):
        return self.studyHrs

    def getTechIndex(self):
        return self.techIndex

    def calculateIncome(self):  # I came up with this equation ^_^
        self.income = int(10 * (ageC * (self.age) + workoutC * self.workHrs +
                            studyC * self.studyHrs + techC * self.techIndex))

    def getIncome(self):
        return self.income

    def __str__(self):
        return f"{self.fullName} -- {self.age} -- {self.gender} -- Work Hours: {self.workHrs} -- Study Hours : {self.studyHrs} -- Tech Index: {self.techIndex} -- Income: {self.income} USD"


def GenerateHumans(PeopleAmount, fNamesPath, lNamesPath):
    """
    The GenerateHumans function is defined, which generates a specified 
    number of random Human objects using data from two text 
    files containing first and last names.
    """
    firstNames = []
    lastNames = []

    # Get first names from data
    with open(fNamesPath) as f:
        lines = f.readlines()
        firstNames = list(map(lambda name: name[:len(name)-1], lines))

    # Get last names from data
    with open(lNamesPath) as f:
        lines = f.readlines()
        lastNames = list(map(lambda lName: lName[:len(lName)-1], lines))

    count = 0
    fSize = len(firstNames)
    lSize = len(lastNames)
    peopleList = []

    while count < PeopleAmount:
        fName = firstNames[random.randint(0, fSize-1)]
        gender = fName[-1]  # Only males and females !!! xD
        fName = fName[:len(fName)-2]
        lName = lastNames[random.randint(0, lSize-1)]
        fullName = fName + " " + lName

        age = random.randint(18, 70)
        workHrs = random.randint(1, 8)
        studyHrs = max(12 - workHrs - 2, 0)
        techIndex = random.randint(0, 10)
        newHuman = Human(age, fullName, gender, workHrs, studyHrs, techIndex)
        peopleList.append(newHuman)
        count += 1

    return peopleList

print("Choose Program Mode:")
print("Modified GramSchmidt - 1")
print("HouseHolder - 2")
mode = int(input("Choose: "))
HouseOrGram = True
if mode == 1:
    HouseOrGram = True
elif mode == 2:
    HouseOrGram = False
else:
    raise ValueError("1,2 ONLY!!!!")

people = GenerateHumans(50, "Data-1\ElvenFirstNames.txt",
                        "Data-1\ElvenLastNames.txt")
for human in people:
    print(human)

# Cx = d
C = np.array([0, 0, 0, 1])
d = np.array([5])


A, b = TurnPeopleIntoMatrix2(people)
xhat = 0

"""
The QR decomposition is performed on A using either 
the Householder or modified Gram-Schmidt method,
 depending on the value of the HouseOrGram flag.
"""
if HouseOrGram:
    Q, R = ModGramSchmidt(A)

    A = A.tolist()
    A.append(C)
    A = np.array(A)
    Qq,Rr = ModGramSchmidt(A)

    Q1 = Qq[:A.shape[0]-1]
    Q2 = Qq[A.shape[0]-1:]

    Qhat, Rhat = ModGramSchmidt(Q2.T)

    u = np.linalg.solve(Rhat.T, d)
    c = np.matmul(Qhat.T , Q1.T) @ b - u
    w = np.linalg.solve(Rhat, c)
    y = np.matmul(Q1.T , b) - np.matmul(Q2.T , w)
    xhat = np.linalg.solve(R, y)

else:
    m = A.shape[0]
    n = A.shape[1]
    Q, R = HouseHolder(A)
    for i in range(m-n):
        lst = Q.shape[1]-1
        Q = np.delete(Q,lst,1)

    for i in range(m-n):
        lst = R.shape[0]-1
        R = np.delete(R,lst,0)

    A = A.tolist()
    A.append(C)
    A = np.array(A)
    Qq,Rr = HouseHolder(A)
    m = A.shape[0]
    n = A.shape[1]
    for i in range(m-n):
        lst = Qq.shape[1]-1
        Qq = np.delete(Qq,lst,1)
    
    for i in range(m-n):
       lst = Rr.shape[0]-1
       Rr = np.delete(Rr,lst,0)

    Q2 = Qq[A.shape[0]-1:]
    Q1 = Qq[:A.shape[0]-1]


    Qhat, Rhat = HouseHolder(Q2.T)
    m = Q2.T.shape[0]
    n = Q2.T.shape[1]

    for i in range(m-n):
        lst = Qhat.shape[1]-1
        Qhat = np.delete(Qhat,lst,1)

    for i in range(m-n):
        lst = Rhat.shape[0]-1
        Rhat = np.delete(Rhat,lst,0)


    u = np.linalg.solve(Rhat.T, d)
    c = np.matmul(Qhat.T , Q1.T) @ b - u
    w = np.linalg.solve(Rhat, c)
    y = np.matmul(Q1.T , b) - np.matmul(Q2.T , w)
    xhat = np.linalg.solve(R, y)



# One concrete example
hooman = Human(50,"John Doe","M",4,5,10)
hoomanName = hooman.getFullName()
hooman = TurnHumanIntoMatrix2(hooman)
res = int(np.matmul(xhat,hooman))
print(hoomanName,"has income of",res,"USD")

# VISUALIS THAT COMPUTED DATA IS REALLY CORRECT
# IT'S IMPOSSIBLE TO MAP 3+ DIMENSIONAL EQUATIONS ON 2D ACCURATELY, IT'S JUST INTERPRETATION
    # But it will give you an idea that result is in range of trained data
# BLUE -> Training data
# RED -> New data, for which we predict the age
# ------------------------------------------------------

xpoints = list(map(lambda human: (human[0]) ,enumerate(people)))
xpoints = np.array(xpoints)

ypoints = list(map(lambda human : human.getIncome() ,people))
ypoints = np.array(ypoints)


experimentPeople = GenerateHumans(20,"Data-1\ElvenFirstNames.txt","Data-1\ElvenLastNames.txt")
newXpoints = list(map(lambda human:human[0],enumerate(experimentPeople)))
newXpoints = np.array(newXpoints)
newYpoints = list(map(lambda human : int(np.matmul(TurnHumanIntoMatrix2(human),xhat)) ,experimentPeople))
newYpoints = np.array(newYpoints)


plt.plot(xpoints, ypoints, 'o')
plt.plot(newXpoints, newYpoints, 'o',color='r')
plt.show()