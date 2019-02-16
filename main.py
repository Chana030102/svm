
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
mean = mean.reshape(1,-1)        # reshape into columns to scale data
std = std.reshape(1,-1)
scaled_train = (traind-mean)/std  
scaled_test = (testd-mean)/std

#========= Experiment 1 ==========
model = svm.SVC(kernel="linear")
score = model.fit(scaled_train,trainl).decision_function(scaled_test)
prediction = model.predict(scaled_test)

fpr, tpr, _ = metric.roc_curve(testl,score)
roc_auc = metric.auc(fpr,tpr)

recall    = metric.recall_score(testl,prediction)
precision = metric.precision_score(testl,prediction)
accuracy  = metric.accuracy_score(testl,prediction)

print("Experiment 1 Results:")
print("Recall = {0:.2%}".format(recall))
print("Precision = {0:.2%}".format(precision))
print("Accuracy = {0:.2%}".format(accuracy))

# Plot ROC curve
plt.figure()
plt.plot(fpr,tpr, lw=2, label='ROC curve (area=%0.2f)' % roc_auc)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve - Spam Database')
plt.show()

#========= Experiment 2 ==========
print("Experiment 2 Start")
w = model.coef_   # Retrieve weight vector
x = numpy.arange(2,58)
a = []            # store accuracy

top = numpy.argpartition(w,-5)[:,-5:].reshape(5)
print("Top 5 features have the following indices: {}".format(top))

for m in range(2,58):
    index  = numpy.argpartition(w,-m)[:,-m:].reshape(m)
    dtrain = scaled_train[:,index]
    dtest = scaled_test[:,index]
    model = svm.SVC(kernel="linear")
    prediction = model.fit(dtrain, trainl).predict(dtest)
    a.append(metric.accuracy_score(testl,prediction))

plt.figure()
plt.plot(x, a, lw=2)
plt.xlabel('m')
plt.ylabel('Accuracy')
plt.title('Feature Selection Accuracy')
plt.show()
    

#========= Experiment 3 ==========
print("Experiment 3 Start")
b = []    # store accuracy
for m in range(2,58):
    index = numpy.random.choice(57,m)
    dtrain = scaled_train[:,index]
    dtest = scaled_test[:,index]
    model = svm.SVC(kernel="linear")
    prediction = model.fit(dtrain, trainl).predict(dtest)
    b.append(metric.accuracy_score(testl,prediction))

plt.figure()
plt.plot(x, b, lw=2)
plt.xlabel('m')
plt.ylabel('Accuracy')
plt.title('Random Feature Selection Accuracy')
plt.show()