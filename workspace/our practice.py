import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
import pickle


def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


if __name__ == "__main__":
    
    # read the data
    df=pd.read_csv('data.csv')
    train , test = data_split(df, 0.2)
    x_train= train[['fever','Body pain','Age','Runny nose','Diff Breathing']].to_numpy()
    x_teast= test[['fever','Body pain','Age','Runny nose','Diff Breathing']].to_numpy()
    
    
    y_train= train[['Infection prob']].to_numpy().reshape(3141)#reshape is used to change the sahpe like row into column shape
    y_test= test[['Infection prob']].to_numpy().reshape(785)
    
    
    clf=LogisticRegression()
    clf.fit(x_train,y_train)
    
    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')
    
    # dump information to that file
    pickle.dump(clf, file)
    file.close()    