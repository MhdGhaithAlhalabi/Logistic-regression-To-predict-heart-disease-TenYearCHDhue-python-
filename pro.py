
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
chd_data = pd.read_csv("C:\\Users\\Ghaith\\Desktop\\framingham.csv")


sns.countplot('TenYearCHD', data=chd_data)
plt.show()


chd_data.male.plot.hist()
plt.show()

chd_data.age.plot.hist()
plt.show()

chd_data.education.plot.hist()
plt.show()

chd_data.currentSmoker.plot.hist()
plt.show()

chd_data.cigsPerDay.plot.hist()
plt.show()

chd_data.BPMeds.plot.hist()
plt.show()

chd_data.prevalentStroke.plot.hist()
plt.show()

chd_data.prevalentStroke.plot.hist()
plt.show()

chd_data.prevalentHyp.plot.hist()
plt.show()

chd_data.diabetes.plot.hist()
plt.show()

chd_data.diabetes.plot.hist()
plt.show()

chd_data.totChol.plot.hist()
plt.show()

chd_data.sysBP.plot.hist()
plt.show()

chd_data.diaBP.plot.hist()
plt.show()

chd_data.BMI.plot.hist()
plt.show()

chd_data.heartRate.plot.hist()
plt.show()

chd_data.glucose.plot.hist()
plt.show()

chd_data.TenYearCHD.plot.hist()
plt.show()

sns.countplot('TenYearCHD', data=chd_data, hue='male')
plt.show()

sns.countplot('TenYearCHD', data=chd_data, hue='age')
plt.show()

sns.countplot('TenYearCHD', data=chd_data, hue='education')
plt.show()

sns.countplot('TenYearCHD', data=chd_data, hue='currentSmoker')
plt.show()

sns.countplot('TenYearCHD', data=chd_data, hue='cigsPerDay')
plt.show()

sns.countplot('TenYearCHD', data=chd_data, hue='BPMeds')
plt.show()

sns.countplot('TenYearCHD', data=chd_data, hue='prevalentStroke')
plt.show()

sns.countplot('TenYearCHD', data=chd_data, hue='prevalentHyp')
plt.show()

sns.countplot('TenYearCHD', data=chd_data, hue='diabetes')
plt.show()

sns.countplot('TenYearCHD', data=chd_data, hue='totChol')
plt.show()

sns.countplot('TenYearCHD', data=chd_data, hue='sysBP')
plt.show()

sns.countplot('TenYearCHD', data=chd_data, hue='diaBP')
plt.show()


sns.countplot('TenYearCHD', data=chd_data, hue='heartRate')
plt.show()

sns.countplot('TenYearCHD', data=chd_data, hue='glucose')
plt.show()


sns.heatmap(chd_data.isnull())
plt.show()

chd_data.drop(['currentSmoker', 'education', 'glucose', 'heartRate',
               'prevalentHyp', 'prevalentStroke', 'BPMeds', 'diabetes'], axis=1, inplace=True)
chd_data.info()

chd_data.dropna(inplace=True)

sns.heatmap(chd_data.isnull())
plt.show()

X = chd_data.drop("TenYearCHD", axis=1)
Y = chd_data["TenYearCHD"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=30)

logmodel = LogisticRegression()

logmodel.fit(X_train, Y_train)


prediction = logmodel.predict(X_test)


classification_report(Y_test, prediction)


confusion_matrix(Y_test, prediction)


accuracy_score(Y_test, prediction)
