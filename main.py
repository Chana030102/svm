
from sklearn import svm
import sklearn.metrics as metric
import matplotlib.pyplot as plt
import numpy

#========= Data Import and Preprocessing ==========
DATA_PATH = "../spambase.data"
DELIMITER = ","
data = numpy.loadtxt(DATA_PATH,delimiter=DELIMITER)

# Split data in half: one for testing, one for training
# keep labels in separate array and remove from samples
index =round(len(data)) # get the index for half of the data 
data_test = data[:index]
label_test = data_test[:,-1]
data_test = numpy.delete(data_test,-1,axis=1)

data_train = data[index:]
label_train = data_train[:,-1]
data_train = numpy.delete(data_train,-1,axis=1)

# Scale training data
mean = numpy.mean(data_train,axis=1) # Calculate mean of each feature (columns)
std = numpy.std(data_train,axis=1)   # Calculate standard deviation of each feature
scaled_train = (data_train-mean)/std

# Scale test data using mean and std from training data
scaled_test = (data_test-mean)/std

#========= Experiment 1 ==========
model = svm.SVC(kernel="linear")
model.fit(data_train,label_train)
prediction = model.predict(data_test)

recall    = metric.recall_score(label_test,prediction)
precision = metric.precision_score(label_test,prediction)
accuracy  = metric.accuracy_score(label_test,prediction)

print("Recall = {}%\tPrecision = {}%\tAccuracy = {}%".format(recall, precision,accuracy))



