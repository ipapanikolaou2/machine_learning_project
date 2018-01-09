import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



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

# ----------Performing Multiple Layer Percepton algorithm for 1 hidden layer and 15 neurons using stochastic gradient descent and testing the algorithm on the test data-----------
mlp = MLPClassifier(hidden_layer_sizes=(15),solver='sgd',max_iter=700)
print(mlp.fit(X_train,y_train))
# predict the response
predictions = mlp.predict(X_test)
#evaluate accuracy
print('---------mlp classification report for 15 neurons-------------- \n')
print(confusion_matrix(y_test,predictions))
print('---------mlp confusion matrix report for 15 neurons-------------- \n')
print(classification_report(y_test,predictions))
print('\n')

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
print('\n')

# plot accuracy vs k

plot1 = plt.figure(1)
plt.plot(neighbors, cv_scores_mean)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy')
plot1.savefig("determining_k_for_knn.png")
plt.title('The optimal k for KNN algorithm is '+str(optimal_k))
plt.show()
plt.close(plot1)

#plot cross validation for optimal k
plot2 = plt.figure(2)
plt.plot(optimal_cv_scores)
plt.title('The mean of cross validation scores for optimal k = '+str(optimal_k)+' is '+str(cv_scores_mean[max_cv_score_index]))
plt.savefig("Cross_Validation_Scores_for_Optimal_k")
plt.show()
plt.close(plot2)



#performing 10-fold cross validation for MLP and using the number of neurons as hyperparameter with one hidden layer

neurons = [ 20, 70, 120, 170]
gs_mlp = GridSearchCV(MLPClassifier(solver='sgd',max_iter=700), param_grid={'hidden_layer_sizes':neurons},refit=True,verbose=3,cv=10)
gs_mlp.fit(X_train,y_train)

# plot accuracy vs hyperparameter

results_mlp=gs_mlp.cv_results_
plot3=plt.figure(3)
plt.title("GridsearchCV cross-valdiation")
plt.xlabel("number of neurons in the hidden layer")
plt.ylabel("Score")
plt.plot(neurons,results_mlp['mean_test_score'].data)
plt.savefig("Cross_Validation_Scores_for_hyperparameter_neurons")
plt.show()
plt.close(plot3)

# determining best number of neurons from gridsearchcv results

optimal_hyper_parameter = gs_mlp.best_params_['hidden_layer_sizes']
print('the best number of neurons is: ',optimal_hyper_parameter,', with mean score: ',gs_mlp.best_score_)

#evaluating accuracy

gs_mlp_predictions = gs_mlp.predict(X_test)
print('---------mlp classification report after cross validation for the best hyperparameter:',optimal_hyper_parameter,'-------------- \n')
print(confusion_matrix(y_test,gs_mlp_predictions))
print('---------mlp confusion matrix report after cross validation for the best hyperparameter:',optimal_hyper_parameter,'-------------- \n')
print(classification_report(y_test,gs_mlp_predictions))


#Principal Component Analysis

# creating array of possible principal components
i = [ 20, 40, 60, 80, 100, 120, 140, 160]
# creating empty arrays to fill with the cross validation scores
scores_knn_after_pca = []
scores_mlp_after_pca = []
# knn algorithm with optimal k
knn_best = KNeighborsClassifier(n_neighbors=optimal_k)
# neural network (mlp) algorithm with optimal number of neurons
mlp_best = MLPClassifier(optimal_hyper_parameter)
# applying PCA for each possible number of principal components and performing cross validation
for components in i:
    # PCA with i principal components
    pca = PCA(components)
    pca.fit(X)
    X_pca = pca.transform(X)

    # cross validation for knn and saving the mean score of cross validation
    knn_scores=(cross_val_score(knn_best,X_pca,y, cv=10, scoring='accuracy')).mean()
    scores_knn_after_pca.append(knn_scores)

    # cross validation for mlp and saving the mean score of cross validation
    mlp_scores=(cross_val_score(mlp_best,X_pca,y, cv=10, scoring='accuracy')).mean()
    scores_mlp_after_pca.append(mlp_scores)

# plotting cross validation scores while showing the optimal number of components for knn algorithm
plot4 = plt.figure(4)
plt.title('cross validation of knn for each number of principal components \n optimal components = '+str(i[scores_knn_after_pca.index(max(scores_knn_after_pca))]))
plt.plot(i,scores_knn_after_pca)
plt.xlabel('n_components')
plt.ylabel('accuracy')
plt.axvline(x=i[scores_knn_after_pca.index(max(scores_knn_after_pca))], linestyle='--')
plt.savefig("cross validation of knn for each number of principal components")
plt.show()
plt.close(plot4)

# plotting cross validation scores while showing the optimal number of components for mlp algorithm
plot5 = plt.figure(5)
plt.title('cross validation of mlp for each number of principal components  \n optimal components = '+str(i[scores_mlp_after_pca.index(max(scores_mlp_after_pca))]))
plt.plot(i,scores_mlp_after_pca)
plt.xlabel('n_components')
plt.ylabel('accuracy')
plt.axvline(x=i[scores_mlp_after_pca.index(max(scores_mlp_after_pca))], linestyle='--')
plt.savefig("cross validation of mlp for each number of principal components")
plt.show()
plt.close(plot5)


#TODO nnc cross validation 2
#TODO knn cross validation 2

#TODO PUT COMMENTS!!