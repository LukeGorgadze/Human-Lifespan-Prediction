import matplotlib.pyplot as plt
import numpy as np
import random
from LukaUtils import ModGramSchmidt,NormalEquation,HouseHolder, TurnHumanIntoMatrix, TurnPeopleIntoMatrix

# Constants are regenerated every time
ageC = random.random() * 1 + 1
sleepC = random.random() * 2 + 1
workOutC = random.random() * 2 + 1

class Human:

    """
    The Human class is defined, with a constructor method that 
    initializes the various attributes of a Human object
    (name, gender, sleep time, workout hours, and lifespan). 
    There are also several methods defined to get the values 
    of these attributes and to calculate the lifespan of a 
    Human based on a formula using the other attributes.
    """

    def __init__(self,fullName,gender,sleepTime,workOutHrs):
        self.fullName = fullName
        if gender == "F":
            self.gender = "Female"
        else:
            self.gender = "Male"
        self.sleepTime = sleepTime
        self.workoutHrs = workOutHrs
        self.midAge = 60
        self.calculateLifespan()

    def getGender(self):
        return self.gender

    def getFullName(self):
        return self.fullName

    def getSleepTime(self):
        return self.sleepTime
    
    def getWorkoutHrs(self):
        return self.workoutHrs
    
    def calculateLifespan(self): # I came up with this equation myself xD
        self.lifeSpan = int(ageC * self.midAge + sleepC * self.sleepTime + workOutC * self.workoutHrs)

    def getMidAge(self):
        return self.midAge
    
    def getLifeSpan(self):
        return self.lifeSpan

    def __str__(self):
        return f"{self.fullName} -- {self.gender} -- SleepTime: {self.sleepTime} -- WeeklyWorkoutHours : {self.workoutHrs} -- LifeSpan: {self.lifeSpan}"


def GenerateHumans(PeopleAmount,fNamesPath,lNamesPath):
    """
    The GenerateHumans function is defined, which generates a specified number of random 
    Human objects using data from two text files containing first and last names.
    """
    firstNames = []
    lastNames = []

    # Get first names from data
    with open(fNamesPath) as f:
        lines = f.readlines()
        firstNames = list(map(lambda name: name[:len(name)-1],lines))

    # Get last names from data
    with open(lNamesPath) as f:
        lines = f.readlines()
        lastNames = list(map(lambda lName: lName[:len(lName)-1],lines))

    count = 0

    fSize = len(firstNames)
    lSize = len(lastNames)
    peopleList = []
    while count < PeopleAmount:
        fName = firstNames[random.randint(0,fSize-1)]
        gender = fName[-1] # Only males and females !!! xD
        fName = fName[:len(fName)-2]
        lName = lastNames[random.randint(0,lSize-1)]
        fullName = fName + " " + lName

        sleepTime = random.randint(1,12)
        workoutHrs = random.randint(0,14)
        newHuman = Human(fullName,gender,sleepTime,workoutHrs)
        peopleList.append(newHuman)
        count += 1
    
    return peopleList

print("Choose Program Mode:")
print("Normal Equation - 0")
print("Modified GramSchmidt - 1")
print("HouseHolder - 2")
mode = int(input("Choose: "))
people = GenerateHumans(100,"ElvenFirstNames.txt","ElvenLastNames.txt")
for human in people:
    print(human)

x = []
A,b = TurnPeopleIntoMatrix(people)
if mode == 0:
    x = NormalEquation(A,b)
elif mode == 1:   
    Q,R = ModGramSchmidt(A)
    # We need to solve R x = Q.T b
    x =  (np.linalg.pinv(R) @ Q.T) @ b
elif mode == 2:
    Q,R = HouseHolder(A)
    # We need to solve R x = Q.T b
    x =  (np.linalg.pinv(R) @ Q.T) @ b
else:
    raise ValueError("0,1,2 ONLY!!!!")




# One concrete example
hooman = Human("John Doe","M",5,5)
hoomanName = hooman.getFullName()
hooman = TurnHumanIntoMatrix(hooman)
res = int(np.matmul(x,hooman))
print(hoomanName,"will live",res,"years")




# VISUALIS THAT COMPUTED DATA IS REALLY CORRECT
# IT'S IMPOSSIBLE TO MAP 3+ DIMENSIONAL EQUATIONS ON 2D ACCURATELY, IT'S JUST INTERPRETATION
    # But it will give you an idea that result is in range of trained data
# BLUE -> Training data
# RED -> New data, for which we predict the age
# ------------------------------------------------------

xpoints = list(map(lambda human: human[0],enumerate(people)))
xpoints = np.array(xpoints)

ypoints = list(map(lambda human : human.getLifeSpan() ,people))
ypoints = np.array(ypoints)

experimentPeople = GenerateHumans(50,"ElvenFirstNames.txt","ElvenLastNames.txt")
newXpoints = list(map(lambda human:human[0],enumerate(experimentPeople)))
newXpoints = np.array(newXpoints)
newYpoints = list(map(lambda human : int(np.matmul(TurnHumanIntoMatrix(human),x)) ,experimentPeople))
newYpoints = np.array(newYpoints)

plt.plot(xpoints, ypoints, 'o')
plt.plot(newXpoints, newYpoints, 'o',color='r')
plt.show()