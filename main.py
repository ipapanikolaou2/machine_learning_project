
import pandas as pd
#reading csv file
df = pd.read_csv('data.csv')

#seperating target from data
X=df.drop('y', axis=1)
X=X.drop(X.columns[0], axis=1)
y_before_bin=df['y']

y = y_before_bin.replace([2, 3, 4, 5], 0)

#splitting the data into training and testing data
from sklearn.model_selection import train_test_split
#spliting_data
X_train, X_test, y_train, y_test = train_test_split(X,y)

#TODO do nnc with split
print("hello")


#TODO knn with split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print (accuracy_score(y_test, pred))

#TODO nnc cross validation
#TODO knn cross validation

#TODO PCA nnc
#TODO PCA for knn

#TODO nnc cross validation 2
#TODO knn cross validation 2

#TODO PUT COMMENTS!!


