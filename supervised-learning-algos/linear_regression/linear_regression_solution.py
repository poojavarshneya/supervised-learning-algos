import csv
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

# open csv file and get the rows
with open('/home/vagrant/repos/datasets/health_data.csv', 'rb') as f:
    rows = [row for row in csv.reader(f)]

# the first row is the header, 
# let's exclude the header from the data
header = rows[0]

#Read the data
rows = rows[1:]
np.random.shuffle(rows)

def predict1(age, years_education, income):
    # All of the data has been read in as strings, so we need to cast them to ints
    # Let's separate the data into our input data and our target
    input_data = np.array([
        [int(row[0]), float(row[1]), float(row[2])] for row in rows])
    
    input_data = sm.add_constant(input_data)
    # Let's pull out the target data
    target = np.array([float(row[3]) for row in rows])

    # OLS stands for Ordinary Least Squares, the most common method of linear regression
    regression_model1 = sm.OLS(target, input_data)
    results1 = regression_model1.fit()
    # Run through our input data and see what our model would output
    predictions = results1.predict(input_data)
    # Show us some statistical summary of the model
    #print results1.summary()

    return results1.predict([1, age, years_education, income])

def predict2(x1, x2, index_x1, index_x2):
    # All of the data has been read in as strings, so we need to cast them to ints
    # Let's separate the data into our input data and our target
    input_data = np.array([
        [int(row[index_x1]), float(row[index_x2])] for row in rows])
    
    input_data = sm.add_constant(input_data)
    # Let's pull out the target data
    target = np.array([float(row[3]) for row in rows])

    # OLS stands for Ordinary Least Squares, the most common method of linear regression
    regression_model2 = sm.OLS(target, input_data)
    results2 = regression_model2.fit()
    # Run through our input data and see what our model would output
    predictions = results2.predict(input_data)
    # Show us some statistical summary of the model
    #print results2.summary()
    
    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(input_data[:, index_x1], input_data[:, index_x2], target)
    X, Y = np.meshgrid(input_data[:, index_x1], input_data[:, index_x2])
    surface = ax.plot_surface(X, Y, predictions, color='yellow')

    return results2.predict([1, x1, x2])

print "\n"
num_doctor_visit1 = predict1(50,11,.5)[0]
num_doctor_visit2 = predict2(50,11,0,1)[0]
num_doctor_visit3 = predict2(50,.5,0,2)[0]
print "Now let's predict number of doctor's visits for a person with age=50, years of education=11," 
print "income = 5000 with model 1.... {} times".format(num_doctor_visit1)

print "Now let's predict number of doctor's visits for a person with age=50, years of education=11" 
print "with model 2.... {} times".format(num_doctor_visit2)

print "Now let's predict number of doctor's visits for a person with age=50, income=5000" 
print "with model 3.... {} times".format(num_doctor_visit3)

