
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler




#reading csv file

df = pd.read_csv('data.csv')

#seperating target from data
X=df.drop('y', axis=1)
X=X.drop(X.columns[0], axis=1)  # the first column is irrelevant
y_before_bin=df['y']
#binarizing the target data
y = y_before_bin.replace([2, 3, 4, 5], 0)
#scaling X data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)



#splitting the data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(X,y)

#TODO do nnc with split

#TODO knn with split
# ----------Performing KNN algorithm for k=3 and testing the algorithm on the test data-----------

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print('---------knn classification report for k = 3--------------')
print(classification_report(y_test, pred))
print('---------knn confusion matrix report for k = 3--------------')
print(confusion_matrix(y_test,pred))


# --------Determining the most effective number of neighbours while performing cross validation----------

# range of possible number of neighbours. They must me odd
#TODO change the range to a bigger number before we send it to berbe
neighbors = range(1, 30, 6)

# empty list that will hold cv scores (means and analytical cross validation scores)
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
# obtaining cross validation scores
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





# TODO PCA nnc

# TODO PCA for knn

# creating array of possible principal components
i = [ 20, 40, 60, 80, 100, 120, 140, 160]
# creating empty arrays to fill with the cross validation scores
scores_knn_after_pca = []
scores_mlp_after_pca = []
# knn algorithm with optimal k
knn = KNeighborsClassifier(n_neighbors=optimal_k)
# TODO change as variable
mlp = MLPClassifier(120)
# applying PCA for each possible number of principal components and performing cross validation
for components in i:
    # PCA with i principal components
    pca = decomposition.PCA(components)
    pca.fit(X)
    X_pca = pca.transform(X)

    # cross validation for knn and saving the mean score of cross validation
    knn_scores=(cross_val_score(knn,X_pca,y, cv=10, scoring='accuracy')).mean()
    scores_knn_after_pca.append(knn_scores)

    # cross validation for mlp and saving the mean score of cross validation
    mlp_scores=(cross_val_score(mlp,X_pca,y,cv=10, scoring='accuracy')).mean()
    scores_mlp_after_pca.append(mlp_scores)

# plotting cross validation scores while showing the optimal number of components for knn algorithm
plot3 = plt.figure(3)
plt.title('cross validation of knn for each number of principal components \n optimal components = '+str(i[scores_knn_after_pca.index(max(scores_knn_after_pca))]))
plt.plot(i,scores_knn_after_pca)
plt.xlabel('n_components')
plt.ylabel('accuracy')
plt.axvline(x=i[scores_knn_after_pca.index(max(scores_knn_after_pca))], linestyle='--')

plt.savefig("cross validation of knn for each number of principal components")
plt.close(plot3)

# plotting cross validation scores while showing the optimal number of components for mlp algorithm
plot4 = plt.figure(4)
plt.title('cross validation of mlp for each number of principal components  \n optimal components = '+str(i[scores_mlp_after_pca.index(max(scores_mlp_after_pca))]))
plt.plot(i,scores_mlp_after_pca)
plt.xlabel('n_components')
plt.ylabel('accuracy')
plt.axvline(x=i[scores_mlp_after_pca.index(max(scores_mlp_after_pca))], linestyle='--')
plt.savefig("cross validation of mlp for each number of principal components")
plt.close(plot4)


#TODO nnc cross validation 2
#TODO knn cross validation 2

#TODO PUT COMMENTS!!




