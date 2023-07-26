import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
from LukaUtils import ModGramSchmidt, NormalEquation, HouseHolder, TurnHumanIntoMatrix, TurnPeopleIntoMatrix

# Streamlit header and explanations
st.title("Human Lifespan Prediction App")
st.write("This app generates and predicts the lifespan of human individuals based on their characteristics.")
st.write("Choose a program mode, and the app will generate random human data and make predictions.")

# Constants are regenerated every time
ageC = random.random() * 1 + 1
sleepC = random.random() * 2 + 1
workOutC = random.random() * 2 + 1

class Human:

    def __init__(self, fullName, gender, sleepTime, workOutHrs):
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
    
    def calculateLifespan(self):
        self.lifeSpan = int(ageC * self.midAge + sleepC * self.sleepTime + workOutC * self.workoutHrs)

    def getMidAge(self):
        return self.midAge
    
    def getLifeSpan(self):
        return self.lifeSpan

    def __str__(self):
        return f"{self.fullName} -- {self.gender} -- SleepTime: {self.sleepTime} -- WeeklyWorkoutHours : {self.workoutHrs} -- LifeSpan: {self.lifeSpan}"


def GenerateHumans(PeopleAmount, fNamesPath, lNamesPath):
    firstNames = []
    lastNames = []

    with open(fNamesPath) as f:
        lines = f.readlines()
        firstNames = list(map(lambda name: name[:len(name)-1], lines))

    with open(lNamesPath) as f:
        lines = f.readlines()
        lastNames = list(map(lambda lName: lName[:len(lName)-1], lines))

    count = 0

    fSize = len(firstNames)
    lSize = len(lastNames)
    peopleList = []
    while count < PeopleAmount:
        fName = firstNames[random.randint(0, fSize-1)]
        gender = fName[-1]
        fName = fName[:len(fName)-2]
        lName = lastNames[random.randint(0, lSize-1)]
        fullName = fName + " " + lName

        sleepTime = random.randint(1, 12)
        workoutHrs = random.randint(0, 14)
        newHuman = Human(fullName, gender, sleepTime, workoutHrs)
        peopleList.append(newHuman)
        count += 1
    
    return peopleList



# Streamlit sidebar for user input
st.sidebar.header("Choose Program Mode:")
mode = st.sidebar.radio("Select mode:", ["Normal Equation", "Modified GramSchmidt", "HouseHolder"])

people = GenerateHumans(100, "Data-1\ElvenFirstNames.txt", "Data-1\ElvenLastNames.txt")

# Display generated human data in a table
st.header("Generated Human Data")
data = {
    "Name": [human.getFullName() for human in people],
    "Gender": [human.getGender() for human in people],
    "Sleep Time": [human.getSleepTime() for human in people],
    "Workout Hours": [human.getWorkoutHrs() for human in people]
}
st.table(data)

x = []
A, b = TurnPeopleIntoMatrix(people)

if mode == "Normal Equation":
    x = NormalEquation(A, b)
elif mode == "Modified GramSchmidt":
    Q, R = ModGramSchmidt(A)
    x = (np.linalg.pinv(R) @ Q.T) @ b
else:   
    Q, R = HouseHolder(A)
    x = (np.linalg.pinv(R) @ Q.T) @ b

# Explanation of how the prediction is calculated
st.header("Lifespan Prediction Explanation")
st.write("The app uses a linear regression model to predict the lifespan of a human based on their characteristics.")
st.write("The model is trained on a randomly generated dataset of 100 individuals with their attributes such as gender, sleep time, and workout hours.")
st.write("The coefficients of the model are determined using one of the three methods: Normal Equation, Modified GramSchmidt, or HouseHolder, depending on the chosen mode.")
st.write("Once the coefficients are calculated, the model can be used to predict the lifespan of new individuals.")

# Prediction for a concrete example
hooman = Human("John Doe", "M", 5, 5)
hoomanName = hooman.getFullName()
hooman = TurnHumanIntoMatrix(hooman)
res = int(np.matmul(x, hooman))

st.header("Lifespan Prediction")
st.write(f"{hoomanName} is predicted to live {res} years.")

# Visualization of training data and predictions
xpoints = np.array(list(range(len(people))))
ypoints = np.array([human.getLifeSpan() for human in people])

experimentPeople = GenerateHumans(50, "Data-1\ElvenFirstNames.txt", "Data-1\ElvenLastNames.txt")
newXpoints = np.array(list(range(len(experimentPeople))))
newYpoints = np.array([int(np.matmul(TurnHumanIntoMatrix(human), x)) for human in experimentPeople])

fig, ax = plt.subplots()
ax.plot(xpoints, ypoints, 'o', label="Training Data")
ax.plot(newXpoints, newYpoints, 'o', color='r', label="Predicted Data")
ax.set_xlabel('Individuals')
ax.set_ylabel('Lifespan')
ax.set_title('Human Lifespan Prediction')
ax.legend()

st.pyplot(fig)