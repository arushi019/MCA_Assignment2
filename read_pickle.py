import pickle
from sklearn.svm import SVC
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
f = open('noise_train_x_mfcc.pkl','rb')
train_x = pickle.load(f)
f = open('noise_train_y_mfcc.pkl','rb')
train_y = pickle.load(f)
f = open('noise_test_x_mfcc.pkl','rb')
test_x = pickle.load(f)
f = open('noise_test_y_mfcc.pkl','rb')
test_y = pickle.load(f)
tr_x,tr_y = shuffle(train_x,train_y)
te_x,te_y = shuffle(test_x,test_y)
print("shuffle done")
clf = SVC(probability = True)
print(tr_x.shape)
clf.fit(tr_x,tr_y)
#pred_y_scores = clf.predict_proba(te_x)
pred_y = clf.predict(te_x)
print("Training done")
model = open('noise_mfcc_model.pkl','wb')
pickle.dump(clf,model)
model.close()
print("model saved")
acc = accuracy_score(pred_y,te_y)
print(acc)
cm_matrix = np.zeros((10,10))
for i in range(len(te_y)):
  true = te_y[i]
  predicted = pred_y[i]
  cm_matrix[predicted][true] += 1
plt.figure(figsize = (10,7))
plt.xlabel("True Class")
plt.ylabel("Predicted Class")
sn.heatmap(cm_matrix, annot=True)
