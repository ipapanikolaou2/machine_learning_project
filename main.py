
import pandas as pd
#reading csv file
df = pd.read_csv('data.csv')

#seperating target from data
X=df.drop('y', axis = 1)
X=X.drop(X.columns[0],axis = 1)
y_before_bin=df['y']

y=y_before_bin.replace([2,3,4,5],0)
print(y)



#TODO do nnc with split
from sklearn.model_selection import train_test_split
#spliting_data
X_train, X_test, y_train, y_test = train_test_split(X,y)

#TODO knn with split

#TODO nnc cross validation
#TODO knn cross validation

#TODO PCA nnc
#TODO PCA for knn

#TODO nnc cross validation 2
#TODO knn cross validation 2


