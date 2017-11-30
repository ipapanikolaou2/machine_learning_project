
import pandas as pd
#reading csv file
df = pd.read_csv('data.csv')

#seperating target from data
X=df.drop('y', axis = 1)
X=X.drop(X.columns[0],axis = 1)
y=df['y']


from sklearn.model_selection import train_test_split
#spliting_data
X_train, X_test, y_train, y_test = train_test_split(X,y)


#preproccessing data, normalizing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neural_network import MLPClassifier;
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
print(X.head())


def neural_network_model(X_train,X_test,y_train,y_test):
    # preproccessing data, normalizing
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier;
    mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
    mlp.fit(X_train, y_train)
    print(X.head())

#for each
#xt,yt,xte,yt
neural_network_model()