import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from sklearn import preprocessing
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.svm import SVC  

def linear_svm():
    # download data set: https://drive.google.com/file/d/13nw-uRXPY8XIZQxKRNZ3yYlho-CYm_Qt/view
    # info: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

    # load data
    bankdata = pd.read_csv("/tmp/bill_authentication.csv")  

    # see the data
    bankdata.shape  

    # see head
    bankdata.head()  

    # data processing
    X = bankdata.drop('Class', axis=1)  
    y = bankdata['Class']  

    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # train the SVM
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(X_train, y_train)  

    # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix  
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))  

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# Iris dataset  https://archive.ics.uci.edu/ml/datasets/iris4
def import_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    irisdata = pd.read_csv(url, names=colnames) 

    # process
    X = irisdata.drop('Class', axis=1)  
    y = irisdata['Class']  

    # train
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
    return irisdata

def preProcess(iris):
    x1 = iris['sepal-length']
    x2 = iris['sepal-width']

    X=np.array(list(zip(x1,x2)), dtype = float)
    print(X.shape)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris['Class'])

    X0, X1 = x1, x2
    xx, yy = make_meshgrid(X0, X1)

    return X,y,xx,yy,X0,X1

def polynomial_kernel(X,y,xx,yy,X0,X1):
    print("-------Polynomial Kernel--------")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    svclassifier = SVC(kernel='poly', degree=8,gamma='auto')  
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred,labels=np.unique(y_pred)))  
    print(classification_report(y_test, y_pred,labels=np.unique(y_pred)))   
    
    fig, ax = plt.subplots(1, 1)
    title = 'SVC with Polynomial kernel'
    clf = svclassifier
    plot_contours(ax, clf, xx, yy,
                cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

    plt.show()
    

def gaussian_kernel(X,y,xx,yy,X0,X1):
    print("-------Gaussian Kernel--------")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 
    
    svclassifier = SVC(kernel='rbf',gamma='auto')  
    svclassifier.fit(X_train, y_train)  
    y_pred = svclassifier.predict(X_test) 

    print(confusion_matrix(y_test, y_pred,labels=np.unique(y_pred)))  
    print(classification_report(y_test, y_pred,labels=np.unique(y_pred))) 

    fig, ax = plt.subplots(1, 1)
    title = 'SVC with Gaussuan kernel'
    clf = svclassifier
    plot_contours(ax, clf, xx, yy,
                cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

    plt.show()
 

def sigmoid_kernel(X,y,xx,yy,X0,X1):
    print("-------Sigmoid Kernel--------")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 
    
    svclassifier = SVC(kernel='sigmoid',gamma='auto')  
    svclassifier.fit(X_train, y_train)  
    y_pred = svclassifier.predict(X_test) 
    
    print(confusion_matrix(y_test, y_pred,labels=np.unique(y_pred)))  
    print(classification_report(y_test, y_pred,labels=np.unique(y_pred)))   
    
    fig, ax = plt.subplots(1, 1)
    title = 'SVC with Sigmoid kernel'
    clf = svclassifier
    plot_contours(ax, clf, xx, yy,
                cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

    plt.show()

def test():
    iris= import_iris()
    X,y,xx,yy,X0,X1 = preProcess(iris)
    polynomial_kernel(X,y,xx,yy,X0,X1)
    gaussian_kernel(X,y,xx,yy,X0,X1)
    sigmoid_kernel(X,y,xx,yy,X0,X1)

test()
