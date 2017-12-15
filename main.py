
import pandas as pd
#reading csv file

df = pd.read_csv('data.csv')

#seperating target from data
X=df.drop('y', axis=1)
X=X.drop(X.columns[0], axis=1)
y_before_bin=df['y']

y = y_before_bin.replace([2, 3, 4, 5], 0)

#splitting the data into training and testing data
from sklearn.model_selection import train_test_split, cross_val_score

#spliting_data
X_train, X_test, y_train, y_test = train_test_split(X,y)

#TODO do nnc with split


#TODO knn with split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



# subsetting just the odd ones
neighbors = [1,3,5,7,9]

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
    print("for "+str(k)+" neighbors the score is "+str(current_score))
    cv_scores.append(current_score)

# determining best k
max_cv_score_index=cv_scores.index(max(cv_scores))
optimal_k = neighbors[max_cv_score_index]
print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
import matplotlib.pyplot as plt

plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy')
plt.show()

#TODO nnc cross validation
#TODO knn cross validation

#TODO PCA nnc
#TODO PCA for knn

#TODO nnc cross validation 2
#TODO knn cross validation 2

#TODO PUT COMMENTS!!


