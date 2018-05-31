#Gender classification using 4 kind of scikit learn classifiers. Accuracy scores measured based on the same data.

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC




# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


#Decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction1 = clf.predict(X)
s_tree=accuracy_score(Y,prediction1)*100
prediction = clf.predict([[162, 55, 38]])
print("prediction score by DecisionTreeClassifier %d  \n"%(s_tree))

#k neighbors classifier 
neigh=KNeighborsClassifier()
neigh.fit(X,Y)
prediction2=neigh.predict(X)
s_neigh=accuracy_score(Y,prediction2)*100
print("prediction score by KNeighborsClassifier %d  \n"%(s_neigh))


#MLP classifier
MLP=MLPClassifier()
MLP.fit(X,Y)
prediction3=MLP.predict(X)
s_MLP=accuracy_score(Y,prediction3)*100
print("prediction score by MLPClassifier %d  \n"%(s_MLP))



#Support vector classifier
SVC=SVC()
SVC.fit(X,Y)
prediction4=SVC.predict(X)
s_SVC=accuracy_score(Y,prediction4)*100
print("prediction score by SVC {}  \n".format(s_SVC))









