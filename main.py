
import pandas as pd
from numpy import mean
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler




#reading csv file

df = pd.read_csv('data.csv')

#seperating target from data
X=df.drop('y', axis=1)
X=X.drop(X.columns[0], axis=1) # the first column is irrelevant
y_before_bin=df['y']
#binarizing the target data
y = y_before_bin.replace([2, 3, 4, 5], 0)

#splitting the data into training and testing data

#spliting_data
X_train, X_test, y_train, y_test = train_test_split(X,y)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#TODO do nnc with split


#TODO knn with split
# ----------Performing KNN algorithm for k=3 and testing the algorithm on the test data-----------

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(y_test, pred))



# --------Determining the most effective number of neighbours while performing cross validation----------

# range of possible number of neighbours. They must me odd
#TODO change the range to a bigger number before we send it to berbe
neighbors = range(1, 50, 2)

# empty list that will hold cv scores
cv_scores_mean = []
cv_scores = []

# perform accuracy testing
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)

    # k-fold cross validating the train data
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores_mean.append(scores.mean())
    cv_scores.append(scores)
# determining best k
max_cv_score_index = cv_scores_mean.index(max(cv_scores_mean))
optimal_k = neighbors[max_cv_score_index]
optimal_cv_scores= cv_scores[max_cv_score_index]
print("The optimal number of neighbors for knn classification is %d" % optimal_k)

# plot accuracy vs k

plot1 = plt.figure(1)
plt.plot(neighbors, cv_scores_mean)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy')
plot1.savefig("determining_k_for_knn.png")
plt.title('The optimal k for KNN algorithm is '+str(optimal_k))
plt.close(plot1)

#plot cross validation for optimal k
plot2 = plt.figure(2)
plt.plot(optimal_cv_scores)
plt.title('The mean of cross validation scores for optimal k = '+str(optimal_k)+' is '+str(cv_scores_mean[max_cv_score_index]))
plt.savefig("Cross_Validation_Scores_for_Optimal_k")
plt.close(plot2)




#TODO nnc cross validation
#TODO knn cross validations

# --------performing cross validation for knn on all the data---------------
knn=KNeighborsClassifier(n_neighbors=1)
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


rows, columns = X.shape
print(X.shape)
search_space=[{'pca__n_components': [columns-2,columns-3,columns-4]}]
knn=KNeighborsClassifier(n_neighbors=1)
pca=decomposition.PCA()



pipe= Pipeline(steps=[('pca',pca),('knn',knn)])

estimator = GridSearchCV(pipe, search_space, cv=10, refit=True, verbose=3)
estimator.fit(X, y)

print('Best Number Of Principal Components:', estimator.best_estimator_.get_params())


#TODO nnc cross validation 2
#TODO knn cross validation 2

#TODO PUT COMMENTS!!



