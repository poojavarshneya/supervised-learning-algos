import csv
import numpy as np
import pandas as pd
import pylab as pl
import statsmodels.api as sm

# read and in the data and show summary statistics
data = pd.read_csv('/home/vagrant/repos/datasets/admission_data.csv')
print "\n****DESCRIPTION OF THE DATA****"
print data.describe()


# show histogram of the data
data.hist()
pl.show()

# this is just pandas notation to get columns 1...n
# we want to do this because our input variables are in columns 1...n
# while our target is in column 1 (0=not admitted, 1=admitted)
training_columns = data.columns[1:]
logit = sm.Logit(data["admit"][:320], data[training_columns][:320])
result = logit.fit()
print result.summary()

def predict(gre, gpa, prestige):
    """
    Outputs predicted probability of admission to graduate program
    given gre, gpa and prestige of the institution where
    the student did their undergraduate
    """
    return result.predict([gre, gpa, prestige])[0]


print "\nPrediction for GRE: 400, GPA: 3.59, and Tier 3 Undergraduate degree is..."
print predict(600, 4.0, 1)

# Now let me check the accuracy on the remaining ~80 rows (about 20% of the data)
count = 0
for i, row in data[320:].iterrows():
    probability = predict(row["gre"], row["gpa"], row["prestige"])
    if (probability > .5 and row["admit"]) or (probability <.5 and not row["admit"]):
        continue
    else:
        count += 1
print count
print count / 320.0
