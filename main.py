
from sklearn import svm
import sklearn.metrics as metric
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy

#========= Data Import and Preprocessing ==========
DATA_PATH = "../spambase.data"
DELIMITER = ","
data  = numpy.loadtxt(DATA_PATH,delimiter=DELIMITER)
label = data[:,-1]
data  = numpy.delete(data,-1,axis=1)

# Randomize data and split in half for test and train sets
traind, testd, trainl, testl = train_test_split(data, label, test_size=0.5, random_state=0)

# Scale train and test data with mean and std from train data
mean = numpy.mean(traind,axis=0) # Calculate mean of each feature (columns)
std = numpy.std(traind,axis=0)   # Calculate standard deviation of each feature
mean = mean.reshape(1,-1)            # reshape into columns to scale data
std = std.reshape(1,-1)
scaled_train = (traind-mean)/std  
scaled_test = (testd-mean)/std

#========= Experiment 1 ==========
model = svm.SVC(kernel="linear")
score = model.fit(scaled_train,trainl).decision_function(scaled_test)
prediction = model.predict(scaled_test)

fpr, tpr, _ = metric.roc_curve(testl,score)
roc_auc = metric.auc(fpr,tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr,tpr, lw=2, label='ROC curve (area=%0.2f)' % roc_auc)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve - Spam Database')
#plt.xlim([0.0,1.0])
#plt.ylim([0.0,1.0])
plt.show()

recall    = metric.recall_score(testl,prediction)
precision = metric.precision_score(testl,prediction)
accuracy  = metric.accuracy_score(testl,prediction)

print("Recall = {}%\tPrecision = {}%\tAccuracy = {}%".format(recall, precision,accuracy))