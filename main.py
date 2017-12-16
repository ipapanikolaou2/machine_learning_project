
import pandas as pd
from numpy import mean
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



#reading csv file

df = pd.read_csv('data.csv')

#seperating target from data
X=df.drop('y', axis=1)
X=X.drop(X.columns[0], axis=1)
y_before_bin=df['y']

y = y_before_bin.replace([2, 3, 4, 5], 0)

#splitting the data into training and testing data

#spliting_data
X_train, X_test, y_train, y_test = train_test_split(X,y)

#TODO do nnc with split


#TODO knn with split




#--------Determining the most effective number of neighbours----------

#range of possible number of neighbours. They must me odd
#TODO change the range to a bigger number before we send it to berbe
neighbors = range(1, 9, 2)

# empty list that will hold cv scores
cv_scores = []

# perform accuracy testing
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)

    # fitting the model
    knn.fit(X_train, y_train)

    # predict the response
    pred = knn.predict(X_test)
    current_score=accuracy_score(y_test, pred)
    cv_scores.append(current_score)

# determining best k
max_cv_score_index=cv_scores.index(max(cv_scores))
optimal_k = neighbors[max_cv_score_index]
print("The optimal number of neighbors for knn classification is %d" % optimal_k)

# plot accuracy vs k

plot1 = plt.figure(1)
plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy')
plot1.savefig("determining_k_for_knn.png")
plt.close(plot1)




#TODO nnc cross validation
#TODO knn cross validation
# --------performing cross validation for knn on all the data---------------
cross_score = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
plot2 = plt.figure(2)
plt.plot(cross_score)
plt.xlabel('folds')
plt.ylabel('accuracy')

plt.title('The mean cross validation score for the knn algorithm is '+str(mean(cross_score)))
plot2.savefig("cross_validation_for_knn.png")
plt.close(plot2)

#TODO PCA nnc
#TODO PCA for knn

#TODO nnc cross validation 2
#TODO knn cross validation 2

#TODO PUT COMMENTS!!


