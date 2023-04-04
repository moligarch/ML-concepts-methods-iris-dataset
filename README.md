# Introduction
First of all: Hello Every one!

My name is Moein Verkiani but You will know me as @Moligarch or **Kian**!
in this file we are going to do some hands-on ML exercise on IRIS dataset in order to learn more and practice what we learnt.

First of all, let's prepare our environment for further operation:

+ import libraries
+ modifie environment variable
+ define dataset


```python
#prepare environment
%reset -f
import os
from sklearn import datasets, tree, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, LeaveOneOut,LeavePOut
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import io
from itertools import cycle 


iris_data = pd.read_csv('iris.csv')
iris = datasets.load_iris()
Xsk = iris['data']
ysk = iris['target']
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
```


```python
print(type(Xsk),Xsk.shape,type(ysk),ysk.shape)
#print(X,y)
```

    <class 'numpy.ndarray'> (150, 4) <class 'numpy.ndarray'> (150,)
    

# Basic Statistics

In this part We're going to calculate some basic statistics metrics in order to understand our dataset better.

IRIS dataset is a collection of 4 features (petal/sepal lenght and width) and 1 target that contains 3 spicies:

* Setosa
* Versicolor
* Virginica

so that statistics metrics mentioned before, have better definition if we run them on these spiceis separatly.


```python
#Calculate Features mean, with respect to kind of flower
X_arr=np.array(pd.DataFrame(Xsk , columns=iris['feature_names']))
setosa_mean = [np.mean(X_arr[:50, i]) for i in range(4)]
versicolor_mean = [np.mean(X_arr[50:100, i]) for i in range(4)]
virginica_mean = [np.mean(X_arr[100:150, i]) for i in range(4)]

spicies = {'setosa': setosa_mean, 'versicolor': versicolor_mean, 'virginica': virginica_mean}

Xmean_df = pd.DataFrame(spicies, index=['sepal length', 'sepal width', 'petal length', 'petal width'])
print('Features Mean\n',Xmean_df)
```

    Features Mean
                   setosa  versicolor  virginica
    sepal length   5.006       5.936      6.588
    sepal width    3.428       2.770      2.974
    petal length   1.462       4.260      5.552
    petal width    0.246       1.326      2.026
    


```python
#Calculate Features Standard Deviation

setosa_std = [np.std(X_arr[:50, i]) for i in range(4)]
versicolor_std = [np.std(X_arr[50:100, i]) for i in range(4)]
virginica_std = [np.std(X_arr[100:150, i]) for i in range(4)]
X_std=[np.std(X_arr[:150, i]) for i in range(4)]
categ = {'Total':X_std, 'setosa': setosa_std, 'versicolor': versicolor_std, 'virginica': virginica_std}

Xstd_df=pd.DataFrame(categ, index=['sepal length', 'sepal width', 'petal length', 'petal width'])
print('Features Standard Deviation\n',Xstd_df)
```

    Features Standard Deviation
                      Total    setosa  versicolor  virginica
    sepal length  0.825301  0.348947    0.510983   0.629489
    sepal width   0.434411  0.375255    0.310644   0.319255
    petal length  1.759404  0.171919    0.465188   0.546348
    petal width   0.759693  0.104326    0.195765   0.271890
    


```python
#Calculate Features Variance

setosa_var = [np.var(X_arr[:50, i]) for i in range(4)]
versicolor_var = [np.var(X_arr[50:100, i]) for i in range(4)]
virginica_var = [np.var(X_arr[100:150, i]) for i in range(4)]
X_var=[np.var(X_arr[:150, i]) for i in range(4)]
categ = {'Total':X_var, 'setosa': setosa_var, 'versicolor': versicolor_var, 'virginica': virginica_var}

Xvar_df=pd.DataFrame(categ, index=['sepal length', 'sepal width', 'petal length', 'petal width'])
print('Features Variance\n',Xvar_df)
```

    Features Variance
                      Total    setosa  versicolor  virginica
    sepal length  0.681122  0.121764    0.261104   0.396256
    sepal width   0.188713  0.140816    0.096500   0.101924
    petal length  3.095503  0.029556    0.216400   0.298496
    petal width   0.577133  0.010884    0.038324   0.073924
    

## Scale

When your data has different values, and even different measurement units, it can be difficult to compare them.What is kilograms compared to meters? Or altitude compared to time?

The answer to this problem is scaling. We can scale data into new values that are easier to compare.

The standardization method uses this formula:

```
z = (x - u) / s
```
Where `z` is the new value, `x` is the original value, `u` is the mean and `s` is the standard deviation.

sklearn do all of this with a single command:


```python
scale = StandardScaler()

scaledX = scale.fit_transform(Xsk)
print(scaledX,scaledX.shape)
```

    [[-9.00681170e-01  1.01900435e+00 -1.34022653e+00 -1.31544430e+00]
     [-1.14301691e+00 -1.31979479e-01 -1.34022653e+00 -1.31544430e+00]
     [-1.38535265e+00  3.28414053e-01 -1.39706395e+00 -1.31544430e+00]
     [-1.50652052e+00  9.82172869e-02 -1.28338910e+00 -1.31544430e+00]
     [-1.02184904e+00  1.24920112e+00 -1.34022653e+00 -1.31544430e+00]
     [-5.37177559e-01  1.93979142e+00 -1.16971425e+00 -1.05217993e+00]
     [-1.50652052e+00  7.88807586e-01 -1.34022653e+00 -1.18381211e+00]
     [-1.02184904e+00  7.88807586e-01 -1.28338910e+00 -1.31544430e+00]
     [-1.74885626e+00 -3.62176246e-01 -1.34022653e+00 -1.31544430e+00]
     [-1.14301691e+00  9.82172869e-02 -1.28338910e+00 -1.44707648e+00]
     [-5.37177559e-01  1.47939788e+00 -1.28338910e+00 -1.31544430e+00]
     [-1.26418478e+00  7.88807586e-01 -1.22655167e+00 -1.31544430e+00]
     [-1.26418478e+00 -1.31979479e-01 -1.34022653e+00 -1.44707648e+00]
     [-1.87002413e+00 -1.31979479e-01 -1.51073881e+00 -1.44707648e+00]
     [-5.25060772e-02  2.16998818e+00 -1.45390138e+00 -1.31544430e+00]
     [-1.73673948e-01  3.09077525e+00 -1.28338910e+00 -1.05217993e+00]
     [-5.37177559e-01  1.93979142e+00 -1.39706395e+00 -1.05217993e+00]
     [-9.00681170e-01  1.01900435e+00 -1.34022653e+00 -1.18381211e+00]
     [-1.73673948e-01  1.70959465e+00 -1.16971425e+00 -1.18381211e+00]
     [-9.00681170e-01  1.70959465e+00 -1.28338910e+00 -1.18381211e+00]
     [-5.37177559e-01  7.88807586e-01 -1.16971425e+00 -1.31544430e+00]
     [-9.00681170e-01  1.47939788e+00 -1.28338910e+00 -1.05217993e+00]
     [-1.50652052e+00  1.24920112e+00 -1.56757623e+00 -1.31544430e+00]
     [-9.00681170e-01  5.58610819e-01 -1.16971425e+00 -9.20547742e-01]
     [-1.26418478e+00  7.88807586e-01 -1.05603939e+00 -1.31544430e+00]
     [-1.02184904e+00 -1.31979479e-01 -1.22655167e+00 -1.31544430e+00]
     [-1.02184904e+00  7.88807586e-01 -1.22655167e+00 -1.05217993e+00]
     [-7.79513300e-01  1.01900435e+00 -1.28338910e+00 -1.31544430e+00]
     [-7.79513300e-01  7.88807586e-01 -1.34022653e+00 -1.31544430e+00]
     [-1.38535265e+00  3.28414053e-01 -1.22655167e+00 -1.31544430e+00]
     [-1.26418478e+00  9.82172869e-02 -1.22655167e+00 -1.31544430e+00]
     [-5.37177559e-01  7.88807586e-01 -1.28338910e+00 -1.05217993e+00]
     [-7.79513300e-01  2.40018495e+00 -1.28338910e+00 -1.44707648e+00]
     [-4.16009689e-01  2.63038172e+00 -1.34022653e+00 -1.31544430e+00]
     [-1.14301691e+00  9.82172869e-02 -1.28338910e+00 -1.31544430e+00]
     [-1.02184904e+00  3.28414053e-01 -1.45390138e+00 -1.31544430e+00]
     [-4.16009689e-01  1.01900435e+00 -1.39706395e+00 -1.31544430e+00]
     [-1.14301691e+00  1.24920112e+00 -1.34022653e+00 -1.44707648e+00]
     [-1.74885626e+00 -1.31979479e-01 -1.39706395e+00 -1.31544430e+00]
     [-9.00681170e-01  7.88807586e-01 -1.28338910e+00 -1.31544430e+00]
     [-1.02184904e+00  1.01900435e+00 -1.39706395e+00 -1.18381211e+00]
     [-1.62768839e+00 -1.74335684e+00 -1.39706395e+00 -1.18381211e+00]
     [-1.74885626e+00  3.28414053e-01 -1.39706395e+00 -1.31544430e+00]
     [-1.02184904e+00  1.01900435e+00 -1.22655167e+00 -7.88915558e-01]
     [-9.00681170e-01  1.70959465e+00 -1.05603939e+00 -1.05217993e+00]
     [-1.26418478e+00 -1.31979479e-01 -1.34022653e+00 -1.18381211e+00]
     [-9.00681170e-01  1.70959465e+00 -1.22655167e+00 -1.31544430e+00]
     [-1.50652052e+00  3.28414053e-01 -1.34022653e+00 -1.31544430e+00]
     [-6.58345429e-01  1.47939788e+00 -1.28338910e+00 -1.31544430e+00]
     [-1.02184904e+00  5.58610819e-01 -1.34022653e+00 -1.31544430e+00]
     [ 1.40150837e+00  3.28414053e-01  5.35408562e-01  2.64141916e-01]
     [ 6.74501145e-01  3.28414053e-01  4.21733708e-01  3.95774101e-01]
     [ 1.28034050e+00  9.82172869e-02  6.49083415e-01  3.95774101e-01]
     [-4.16009689e-01 -1.74335684e+00  1.37546573e-01  1.32509732e-01]
     [ 7.95669016e-01 -5.92373012e-01  4.78571135e-01  3.95774101e-01]
     [-1.73673948e-01 -5.92373012e-01  4.21733708e-01  1.32509732e-01]
     [ 5.53333275e-01  5.58610819e-01  5.35408562e-01  5.27406285e-01]
     [-1.14301691e+00 -1.51316008e+00 -2.60315415e-01 -2.62386821e-01]
     [ 9.16836886e-01 -3.62176246e-01  4.78571135e-01  1.32509732e-01]
     [-7.79513300e-01 -8.22569778e-01  8.07091462e-02  2.64141916e-01]
     [-1.02184904e+00 -2.43394714e+00 -1.46640561e-01 -2.62386821e-01]
     [ 6.86617933e-02 -1.31979479e-01  2.51221427e-01  3.95774101e-01]
     [ 1.89829664e-01 -1.97355361e+00  1.37546573e-01 -2.62386821e-01]
     [ 3.10997534e-01 -3.62176246e-01  5.35408562e-01  2.64141916e-01]
     [-2.94841818e-01 -3.62176246e-01 -8.98031345e-02  1.32509732e-01]
     [ 1.03800476e+00  9.82172869e-02  3.64896281e-01  2.64141916e-01]
     [-2.94841818e-01 -1.31979479e-01  4.21733708e-01  3.95774101e-01]
     [-5.25060772e-02 -8.22569778e-01  1.94384000e-01 -2.62386821e-01]
     [ 4.32165405e-01 -1.97355361e+00  4.21733708e-01  3.95774101e-01]
     [-2.94841818e-01 -1.28296331e+00  8.07091462e-02 -1.30754636e-01]
     [ 6.86617933e-02  3.28414053e-01  5.92245988e-01  7.90670654e-01]
     [ 3.10997534e-01 -5.92373012e-01  1.37546573e-01  1.32509732e-01]
     [ 5.53333275e-01 -1.28296331e+00  6.49083415e-01  3.95774101e-01]
     [ 3.10997534e-01 -5.92373012e-01  5.35408562e-01  8.77547895e-04]
     [ 6.74501145e-01 -3.62176246e-01  3.08058854e-01  1.32509732e-01]
     [ 9.16836886e-01 -1.31979479e-01  3.64896281e-01  2.64141916e-01]
     [ 1.15917263e+00 -5.92373012e-01  5.92245988e-01  2.64141916e-01]
     [ 1.03800476e+00 -1.31979479e-01  7.05920842e-01  6.59038469e-01]
     [ 1.89829664e-01 -3.62176246e-01  4.21733708e-01  3.95774101e-01]
     [-1.73673948e-01 -1.05276654e+00 -1.46640561e-01 -2.62386821e-01]
     [-4.16009689e-01 -1.51316008e+00  2.38717193e-02 -1.30754636e-01]
     [-4.16009689e-01 -1.51316008e+00 -3.29657076e-02 -2.62386821e-01]
     [-5.25060772e-02 -8.22569778e-01  8.07091462e-02  8.77547895e-04]
     [ 1.89829664e-01 -8.22569778e-01  7.62758269e-01  5.27406285e-01]
     [-5.37177559e-01 -1.31979479e-01  4.21733708e-01  3.95774101e-01]
     [ 1.89829664e-01  7.88807586e-01  4.21733708e-01  5.27406285e-01]
     [ 1.03800476e+00  9.82172869e-02  5.35408562e-01  3.95774101e-01]
     [ 5.53333275e-01 -1.74335684e+00  3.64896281e-01  1.32509732e-01]
     [-2.94841818e-01 -1.31979479e-01  1.94384000e-01  1.32509732e-01]
     [-4.16009689e-01 -1.28296331e+00  1.37546573e-01  1.32509732e-01]
     [-4.16009689e-01 -1.05276654e+00  3.64896281e-01  8.77547895e-04]
     [ 3.10997534e-01 -1.31979479e-01  4.78571135e-01  2.64141916e-01]
     [-5.25060772e-02 -1.05276654e+00  1.37546573e-01  8.77547895e-04]
     [-1.02184904e+00 -1.74335684e+00 -2.60315415e-01 -2.62386821e-01]
     [-2.94841818e-01 -8.22569778e-01  2.51221427e-01  1.32509732e-01]
     [-1.73673948e-01 -1.31979479e-01  2.51221427e-01  8.77547895e-04]
     [-1.73673948e-01 -3.62176246e-01  2.51221427e-01  1.32509732e-01]
     [ 4.32165405e-01 -3.62176246e-01  3.08058854e-01  1.32509732e-01]
     [-9.00681170e-01 -1.28296331e+00 -4.30827696e-01 -1.30754636e-01]
     [-1.73673948e-01 -5.92373012e-01  1.94384000e-01  1.32509732e-01]
     [ 5.53333275e-01  5.58610819e-01  1.27429511e+00  1.71209594e+00]
     [-5.25060772e-02 -8.22569778e-01  7.62758269e-01  9.22302838e-01]
     [ 1.52267624e+00 -1.31979479e-01  1.21745768e+00  1.18556721e+00]
     [ 5.53333275e-01 -3.62176246e-01  1.04694540e+00  7.90670654e-01]
     [ 7.95669016e-01 -1.31979479e-01  1.16062026e+00  1.31719939e+00]
     [ 2.12851559e+00 -1.31979479e-01  1.61531967e+00  1.18556721e+00]
     [-1.14301691e+00 -1.28296331e+00  4.21733708e-01  6.59038469e-01]
     [ 1.76501198e+00 -3.62176246e-01  1.44480739e+00  7.90670654e-01]
     [ 1.03800476e+00 -1.28296331e+00  1.16062026e+00  7.90670654e-01]
     [ 1.64384411e+00  1.24920112e+00  1.33113254e+00  1.71209594e+00]
     [ 7.95669016e-01  3.28414053e-01  7.62758269e-01  1.05393502e+00]
     [ 6.74501145e-01 -8.22569778e-01  8.76433123e-01  9.22302838e-01]
     [ 1.15917263e+00 -1.31979479e-01  9.90107977e-01  1.18556721e+00]
     [-1.73673948e-01 -1.28296331e+00  7.05920842e-01  1.05393502e+00]
     [-5.25060772e-02 -5.92373012e-01  7.62758269e-01  1.58046376e+00]
     [ 6.74501145e-01  3.28414053e-01  8.76433123e-01  1.44883158e+00]
     [ 7.95669016e-01 -1.31979479e-01  9.90107977e-01  7.90670654e-01]
     [ 2.24968346e+00  1.70959465e+00  1.67215710e+00  1.31719939e+00]
     [ 2.24968346e+00 -1.05276654e+00  1.78583195e+00  1.44883158e+00]
     [ 1.89829664e-01 -1.97355361e+00  7.05920842e-01  3.95774101e-01]
     [ 1.28034050e+00  3.28414053e-01  1.10378283e+00  1.44883158e+00]
     [-2.94841818e-01 -5.92373012e-01  6.49083415e-01  1.05393502e+00]
     [ 2.24968346e+00 -5.92373012e-01  1.67215710e+00  1.05393502e+00]
     [ 5.53333275e-01 -8.22569778e-01  6.49083415e-01  7.90670654e-01]
     [ 1.03800476e+00  5.58610819e-01  1.10378283e+00  1.18556721e+00]
     [ 1.64384411e+00  3.28414053e-01  1.27429511e+00  7.90670654e-01]
     [ 4.32165405e-01 -5.92373012e-01  5.92245988e-01  7.90670654e-01]
     [ 3.10997534e-01 -1.31979479e-01  6.49083415e-01  7.90670654e-01]
     [ 6.74501145e-01 -5.92373012e-01  1.04694540e+00  1.18556721e+00]
     [ 1.64384411e+00 -1.31979479e-01  1.16062026e+00  5.27406285e-01]
     [ 1.88617985e+00 -5.92373012e-01  1.33113254e+00  9.22302838e-01]
     [ 2.49201920e+00  1.70959465e+00  1.50164482e+00  1.05393502e+00]
     [ 6.74501145e-01 -5.92373012e-01  1.04694540e+00  1.31719939e+00]
     [ 5.53333275e-01 -5.92373012e-01  7.62758269e-01  3.95774101e-01]
     [ 3.10997534e-01 -1.05276654e+00  1.04694540e+00  2.64141916e-01]
     [ 2.24968346e+00 -1.31979479e-01  1.33113254e+00  1.44883158e+00]
     [ 5.53333275e-01  7.88807586e-01  1.04694540e+00  1.58046376e+00]
     [ 6.74501145e-01  9.82172869e-02  9.90107977e-01  7.90670654e-01]
     [ 1.89829664e-01 -1.31979479e-01  5.92245988e-01  7.90670654e-01]
     [ 1.28034050e+00  9.82172869e-02  9.33270550e-01  1.18556721e+00]
     [ 1.03800476e+00  9.82172869e-02  1.04694540e+00  1.58046376e+00]
     [ 1.28034050e+00  9.82172869e-02  7.62758269e-01  1.44883158e+00]
     [-5.25060772e-02 -8.22569778e-01  7.62758269e-01  9.22302838e-01]
     [ 1.15917263e+00  3.28414053e-01  1.21745768e+00  1.44883158e+00]
     [ 1.03800476e+00  5.58610819e-01  1.10378283e+00  1.71209594e+00]
     [ 1.03800476e+00 -1.31979479e-01  8.19595696e-01  1.44883158e+00]
     [ 5.53333275e-01 -1.28296331e+00  7.05920842e-01  9.22302838e-01]
     [ 7.95669016e-01 -1.31979479e-01  8.19595696e-01  1.05393502e+00]
     [ 4.32165405e-01  7.88807586e-01  9.33270550e-01  1.44883158e+00]
     [ 6.86617933e-02 -1.31979479e-01  7.62758269e-01  7.90670654e-01]] (150, 4)
    


```python
print(type(iris_data))
```

    <class 'pandas.core.frame.DataFrame'>
    

## Data Visualization

If you are trying to discuss or illustrate something to your Colleges,Co Worker, Your managers or etc. you need to SHOW them what you mean! so although we know **Data Talks Everywhere!**, without Data visualization you are just using 30-40% of Data potential. it also helps you to understand relation between datasets better (not in all case I believe!)

So let's dig deeper.


```python
# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=(12,6))
# =============
# First subplot
# =============
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

x1 = Xsk[:,0]
x2 = Xsk[:,1]

ax.scatter(x1, x2, ysk, marker='o')
ax.set_xlabel('Sepal L')
ax.set_ylabel('Sepal W"')
ax.set_zlabel('Category')
# ==============
# Second subplot
# ==============
# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')

x3 = Xsk[:,2]
x4 = Xsk[:,3]

ax.scatter(x3, x4, ysk, marker='x')
ax.set_xlabel('Petal L')
ax.set_ylabel('Petal W"')
ax.set_zlabel('Category')
plt.show()
```


    
![png](/assets/output_11_0.png)
    



```python
#compare any feature with respect to all features
sn.pairplot(iris_data)
```




    <seaborn.axisgrid.PairGrid at 0x263e366f8b0>




    
![png](/assets/output_12_1.png)
    



```python
plt.hist(ysk, 25)
#plt.show()

plt.title("Data Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()
```


    
![png](/assets/output_13_0.png)
    



<table id="T_19297">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_19297_level0_col0" class="col_heading level0 col0" >sepal.length</th>
      <th id="T_19297_level0_col1" class="col_heading level0 col1" >sepal.width</th>
      <th id="T_19297_level0_col2" class="col_heading level0 col2" >petal.length</th>
      <th id="T_19297_level0_col3" class="col_heading level0 col3" >petal.width</th>
      <th id="T_19297_level0_col4" class="col_heading level0 col4" >variety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_19297_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_19297_row0_col0" class="data row0 col0" >5.100000</td>
      <td id="T_19297_row0_col1" class="data row0 col1" >3.500000</td>
      <td id="T_19297_row0_col2" class="data row0 col2" >1.400000</td>
      <td id="T_19297_row0_col3" class="data row0 col3" >0.200000</td>
      <td id="T_19297_row0_col4" class="data row0 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_19297_row1_col0" class="data row1 col0" >4.900000</td>
      <td id="T_19297_row1_col1" class="data row1 col1" >3.000000</td>
      <td id="T_19297_row1_col2" class="data row1 col2" >1.400000</td>
      <td id="T_19297_row1_col3" class="data row1 col3" >0.200000</td>
      <td id="T_19297_row1_col4" class="data row1 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_19297_row2_col0" class="data row2 col0" >4.700000</td>
      <td id="T_19297_row2_col1" class="data row2 col1" >3.200000</td>
      <td id="T_19297_row2_col2" class="data row2 col2" >1.300000</td>
      <td id="T_19297_row2_col3" class="data row2 col3" >0.200000</td>
      <td id="T_19297_row2_col4" class="data row2 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_19297_row3_col0" class="data row3 col0" >4.600000</td>
      <td id="T_19297_row3_col1" class="data row3 col1" >3.100000</td>
      <td id="T_19297_row3_col2" class="data row3 col2" >1.500000</td>
      <td id="T_19297_row3_col3" class="data row3 col3" >0.200000</td>
      <td id="T_19297_row3_col4" class="data row3 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_19297_row4_col0" class="data row4 col0" >5.000000</td>
      <td id="T_19297_row4_col1" class="data row4 col1" >3.600000</td>
      <td id="T_19297_row4_col2" class="data row4 col2" >1.400000</td>
      <td id="T_19297_row4_col3" class="data row4 col3" >0.200000</td>
      <td id="T_19297_row4_col4" class="data row4 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_19297_row5_col0" class="data row5 col0" >5.400000</td>
      <td id="T_19297_row5_col1" class="data row5 col1" >3.900000</td>
      <td id="T_19297_row5_col2" class="data row5 col2" >1.700000</td>
      <td id="T_19297_row5_col3" class="data row5 col3" >0.400000</td>
      <td id="T_19297_row5_col4" class="data row5 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_19297_row6_col0" class="data row6 col0" >4.600000</td>
      <td id="T_19297_row6_col1" class="data row6 col1" >3.400000</td>
      <td id="T_19297_row6_col2" class="data row6 col2" >1.400000</td>
      <td id="T_19297_row6_col3" class="data row6 col3" >0.300000</td>
      <td id="T_19297_row6_col4" class="data row6 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_19297_row7_col0" class="data row7 col0" >5.000000</td>
      <td id="T_19297_row7_col1" class="data row7 col1" >3.400000</td>
      <td id="T_19297_row7_col2" class="data row7 col2" >1.500000</td>
      <td id="T_19297_row7_col3" class="data row7 col3" >0.200000</td>
      <td id="T_19297_row7_col4" class="data row7 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_19297_row8_col0" class="data row8 col0" >4.400000</td>
      <td id="T_19297_row8_col1" class="data row8 col1" >2.900000</td>
      <td id="T_19297_row8_col2" class="data row8 col2" >1.400000</td>
      <td id="T_19297_row8_col3" class="data row8 col3" >0.200000</td>
      <td id="T_19297_row8_col4" class="data row8 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_19297_row9_col0" class="data row9 col0" >4.900000</td>
      <td id="T_19297_row9_col1" class="data row9 col1" >3.100000</td>
      <td id="T_19297_row9_col2" class="data row9 col2" >1.500000</td>
      <td id="T_19297_row9_col3" class="data row9 col3" >0.100000</td>
      <td id="T_19297_row9_col4" class="data row9 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_19297_row10_col0" class="data row10 col0" >5.400000</td>
      <td id="T_19297_row10_col1" class="data row10 col1" >3.700000</td>
      <td id="T_19297_row10_col2" class="data row10 col2" >1.500000</td>
      <td id="T_19297_row10_col3" class="data row10 col3" >0.200000</td>
      <td id="T_19297_row10_col4" class="data row10 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_19297_row11_col0" class="data row11 col0" >4.800000</td>
      <td id="T_19297_row11_col1" class="data row11 col1" >3.400000</td>
      <td id="T_19297_row11_col2" class="data row11 col2" >1.600000</td>
      <td id="T_19297_row11_col3" class="data row11 col3" >0.200000</td>
      <td id="T_19297_row11_col4" class="data row11 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_19297_row12_col0" class="data row12 col0" >4.800000</td>
      <td id="T_19297_row12_col1" class="data row12 col1" >3.000000</td>
      <td id="T_19297_row12_col2" class="data row12 col2" >1.400000</td>
      <td id="T_19297_row12_col3" class="data row12 col3" >0.100000</td>
      <td id="T_19297_row12_col4" class="data row12 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_19297_row13_col0" class="data row13 col0" >4.300000</td>
      <td id="T_19297_row13_col1" class="data row13 col1" >3.000000</td>
      <td id="T_19297_row13_col2" class="data row13 col2" >1.100000</td>
      <td id="T_19297_row13_col3" class="data row13 col3" >0.100000</td>
      <td id="T_19297_row13_col4" class="data row13 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_19297_row14_col0" class="data row14 col0" >5.800000</td>
      <td id="T_19297_row14_col1" class="data row14 col1" >4.000000</td>
      <td id="T_19297_row14_col2" class="data row14 col2" >1.200000</td>
      <td id="T_19297_row14_col3" class="data row14 col3" >0.200000</td>
      <td id="T_19297_row14_col4" class="data row14 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_19297_row15_col0" class="data row15 col0" >5.700000</td>
      <td id="T_19297_row15_col1" class="data row15 col1" >4.400000</td>
      <td id="T_19297_row15_col2" class="data row15 col2" >1.500000</td>
      <td id="T_19297_row15_col3" class="data row15 col3" >0.400000</td>
      <td id="T_19297_row15_col4" class="data row15 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_19297_row16_col0" class="data row16 col0" >5.400000</td>
      <td id="T_19297_row16_col1" class="data row16 col1" >3.900000</td>
      <td id="T_19297_row16_col2" class="data row16 col2" >1.300000</td>
      <td id="T_19297_row16_col3" class="data row16 col3" >0.400000</td>
      <td id="T_19297_row16_col4" class="data row16 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_19297_row17_col0" class="data row17 col0" >5.100000</td>
      <td id="T_19297_row17_col1" class="data row17 col1" >3.500000</td>
      <td id="T_19297_row17_col2" class="data row17 col2" >1.400000</td>
      <td id="T_19297_row17_col3" class="data row17 col3" >0.300000</td>
      <td id="T_19297_row17_col4" class="data row17 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_19297_row18_col0" class="data row18 col0" >5.700000</td>
      <td id="T_19297_row18_col1" class="data row18 col1" >3.800000</td>
      <td id="T_19297_row18_col2" class="data row18 col2" >1.700000</td>
      <td id="T_19297_row18_col3" class="data row18 col3" >0.300000</td>
      <td id="T_19297_row18_col4" class="data row18 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_19297_row19_col0" class="data row19 col0" >5.100000</td>
      <td id="T_19297_row19_col1" class="data row19 col1" >3.800000</td>
      <td id="T_19297_row19_col2" class="data row19 col2" >1.500000</td>
      <td id="T_19297_row19_col3" class="data row19 col3" >0.300000</td>
      <td id="T_19297_row19_col4" class="data row19 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_19297_row20_col0" class="data row20 col0" >5.400000</td>
      <td id="T_19297_row20_col1" class="data row20 col1" >3.400000</td>
      <td id="T_19297_row20_col2" class="data row20 col2" >1.700000</td>
      <td id="T_19297_row20_col3" class="data row20 col3" >0.200000</td>
      <td id="T_19297_row20_col4" class="data row20 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_19297_row21_col0" class="data row21 col0" >5.100000</td>
      <td id="T_19297_row21_col1" class="data row21 col1" >3.700000</td>
      <td id="T_19297_row21_col2" class="data row21 col2" >1.500000</td>
      <td id="T_19297_row21_col3" class="data row21 col3" >0.400000</td>
      <td id="T_19297_row21_col4" class="data row21 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_19297_row22_col0" class="data row22 col0" >4.600000</td>
      <td id="T_19297_row22_col1" class="data row22 col1" >3.600000</td>
      <td id="T_19297_row22_col2" class="data row22 col2" >1.000000</td>
      <td id="T_19297_row22_col3" class="data row22 col3" >0.200000</td>
      <td id="T_19297_row22_col4" class="data row22 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_19297_row23_col0" class="data row23 col0" >5.100000</td>
      <td id="T_19297_row23_col1" class="data row23 col1" >3.300000</td>
      <td id="T_19297_row23_col2" class="data row23 col2" >1.700000</td>
      <td id="T_19297_row23_col3" class="data row23 col3" >0.500000</td>
      <td id="T_19297_row23_col4" class="data row23 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_19297_row24_col0" class="data row24 col0" >4.800000</td>
      <td id="T_19297_row24_col1" class="data row24 col1" >3.400000</td>
      <td id="T_19297_row24_col2" class="data row24 col2" >1.900000</td>
      <td id="T_19297_row24_col3" class="data row24 col3" >0.200000</td>
      <td id="T_19297_row24_col4" class="data row24 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_19297_row25_col0" class="data row25 col0" >5.000000</td>
      <td id="T_19297_row25_col1" class="data row25 col1" >3.000000</td>
      <td id="T_19297_row25_col2" class="data row25 col2" >1.600000</td>
      <td id="T_19297_row25_col3" class="data row25 col3" >0.200000</td>
      <td id="T_19297_row25_col4" class="data row25 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_19297_row26_col0" class="data row26 col0" >5.000000</td>
      <td id="T_19297_row26_col1" class="data row26 col1" >3.400000</td>
      <td id="T_19297_row26_col2" class="data row26 col2" >1.600000</td>
      <td id="T_19297_row26_col3" class="data row26 col3" >0.400000</td>
      <td id="T_19297_row26_col4" class="data row26 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row27" class="row_heading level0 row27" >27</th>
      <td id="T_19297_row27_col0" class="data row27 col0" >5.200000</td>
      <td id="T_19297_row27_col1" class="data row27 col1" >3.500000</td>
      <td id="T_19297_row27_col2" class="data row27 col2" >1.500000</td>
      <td id="T_19297_row27_col3" class="data row27 col3" >0.200000</td>
      <td id="T_19297_row27_col4" class="data row27 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row28" class="row_heading level0 row28" >28</th>
      <td id="T_19297_row28_col0" class="data row28 col0" >5.200000</td>
      <td id="T_19297_row28_col1" class="data row28 col1" >3.400000</td>
      <td id="T_19297_row28_col2" class="data row28 col2" >1.400000</td>
      <td id="T_19297_row28_col3" class="data row28 col3" >0.200000</td>
      <td id="T_19297_row28_col4" class="data row28 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row29" class="row_heading level0 row29" >29</th>
      <td id="T_19297_row29_col0" class="data row29 col0" >4.700000</td>
      <td id="T_19297_row29_col1" class="data row29 col1" >3.200000</td>
      <td id="T_19297_row29_col2" class="data row29 col2" >1.600000</td>
      <td id="T_19297_row29_col3" class="data row29 col3" >0.200000</td>
      <td id="T_19297_row29_col4" class="data row29 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row30" class="row_heading level0 row30" >30</th>
      <td id="T_19297_row30_col0" class="data row30 col0" >4.800000</td>
      <td id="T_19297_row30_col1" class="data row30 col1" >3.100000</td>
      <td id="T_19297_row30_col2" class="data row30 col2" >1.600000</td>
      <td id="T_19297_row30_col3" class="data row30 col3" >0.200000</td>
      <td id="T_19297_row30_col4" class="data row30 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row31" class="row_heading level0 row31" >31</th>
      <td id="T_19297_row31_col0" class="data row31 col0" >5.400000</td>
      <td id="T_19297_row31_col1" class="data row31 col1" >3.400000</td>
      <td id="T_19297_row31_col2" class="data row31 col2" >1.500000</td>
      <td id="T_19297_row31_col3" class="data row31 col3" >0.400000</td>
      <td id="T_19297_row31_col4" class="data row31 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row32" class="row_heading level0 row32" >32</th>
      <td id="T_19297_row32_col0" class="data row32 col0" >5.200000</td>
      <td id="T_19297_row32_col1" class="data row32 col1" >4.100000</td>
      <td id="T_19297_row32_col2" class="data row32 col2" >1.500000</td>
      <td id="T_19297_row32_col3" class="data row32 col3" >0.100000</td>
      <td id="T_19297_row32_col4" class="data row32 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row33" class="row_heading level0 row33" >33</th>
      <td id="T_19297_row33_col0" class="data row33 col0" >5.500000</td>
      <td id="T_19297_row33_col1" class="data row33 col1" >4.200000</td>
      <td id="T_19297_row33_col2" class="data row33 col2" >1.400000</td>
      <td id="T_19297_row33_col3" class="data row33 col3" >0.200000</td>
      <td id="T_19297_row33_col4" class="data row33 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row34" class="row_heading level0 row34" >34</th>
      <td id="T_19297_row34_col0" class="data row34 col0" >4.900000</td>
      <td id="T_19297_row34_col1" class="data row34 col1" >3.100000</td>
      <td id="T_19297_row34_col2" class="data row34 col2" >1.500000</td>
      <td id="T_19297_row34_col3" class="data row34 col3" >0.200000</td>
      <td id="T_19297_row34_col4" class="data row34 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row35" class="row_heading level0 row35" >35</th>
      <td id="T_19297_row35_col0" class="data row35 col0" >5.000000</td>
      <td id="T_19297_row35_col1" class="data row35 col1" >3.200000</td>
      <td id="T_19297_row35_col2" class="data row35 col2" >1.200000</td>
      <td id="T_19297_row35_col3" class="data row35 col3" >0.200000</td>
      <td id="T_19297_row35_col4" class="data row35 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row36" class="row_heading level0 row36" >36</th>
      <td id="T_19297_row36_col0" class="data row36 col0" >5.500000</td>
      <td id="T_19297_row36_col1" class="data row36 col1" >3.500000</td>
      <td id="T_19297_row36_col2" class="data row36 col2" >1.300000</td>
      <td id="T_19297_row36_col3" class="data row36 col3" >0.200000</td>
      <td id="T_19297_row36_col4" class="data row36 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row37" class="row_heading level0 row37" >37</th>
      <td id="T_19297_row37_col0" class="data row37 col0" >4.900000</td>
      <td id="T_19297_row37_col1" class="data row37 col1" >3.600000</td>
      <td id="T_19297_row37_col2" class="data row37 col2" >1.400000</td>
      <td id="T_19297_row37_col3" class="data row37 col3" >0.100000</td>
      <td id="T_19297_row37_col4" class="data row37 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row38" class="row_heading level0 row38" >38</th>
      <td id="T_19297_row38_col0" class="data row38 col0" >4.400000</td>
      <td id="T_19297_row38_col1" class="data row38 col1" >3.000000</td>
      <td id="T_19297_row38_col2" class="data row38 col2" >1.300000</td>
      <td id="T_19297_row38_col3" class="data row38 col3" >0.200000</td>
      <td id="T_19297_row38_col4" class="data row38 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row39" class="row_heading level0 row39" >39</th>
      <td id="T_19297_row39_col0" class="data row39 col0" >5.100000</td>
      <td id="T_19297_row39_col1" class="data row39 col1" >3.400000</td>
      <td id="T_19297_row39_col2" class="data row39 col2" >1.500000</td>
      <td id="T_19297_row39_col3" class="data row39 col3" >0.200000</td>
      <td id="T_19297_row39_col4" class="data row39 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row40" class="row_heading level0 row40" >40</th>
      <td id="T_19297_row40_col0" class="data row40 col0" >5.000000</td>
      <td id="T_19297_row40_col1" class="data row40 col1" >3.500000</td>
      <td id="T_19297_row40_col2" class="data row40 col2" >1.300000</td>
      <td id="T_19297_row40_col3" class="data row40 col3" >0.300000</td>
      <td id="T_19297_row40_col4" class="data row40 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row41" class="row_heading level0 row41" >41</th>
      <td id="T_19297_row41_col0" class="data row41 col0" >4.500000</td>
      <td id="T_19297_row41_col1" class="data row41 col1" >2.300000</td>
      <td id="T_19297_row41_col2" class="data row41 col2" >1.300000</td>
      <td id="T_19297_row41_col3" class="data row41 col3" >0.300000</td>
      <td id="T_19297_row41_col4" class="data row41 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row42" class="row_heading level0 row42" >42</th>
      <td id="T_19297_row42_col0" class="data row42 col0" >4.400000</td>
      <td id="T_19297_row42_col1" class="data row42 col1" >3.200000</td>
      <td id="T_19297_row42_col2" class="data row42 col2" >1.300000</td>
      <td id="T_19297_row42_col3" class="data row42 col3" >0.200000</td>
      <td id="T_19297_row42_col4" class="data row42 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row43" class="row_heading level0 row43" >43</th>
      <td id="T_19297_row43_col0" class="data row43 col0" >5.000000</td>
      <td id="T_19297_row43_col1" class="data row43 col1" >3.500000</td>
      <td id="T_19297_row43_col2" class="data row43 col2" >1.600000</td>
      <td id="T_19297_row43_col3" class="data row43 col3" >0.600000</td>
      <td id="T_19297_row43_col4" class="data row43 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row44" class="row_heading level0 row44" >44</th>
      <td id="T_19297_row44_col0" class="data row44 col0" >5.100000</td>
      <td id="T_19297_row44_col1" class="data row44 col1" >3.800000</td>
      <td id="T_19297_row44_col2" class="data row44 col2" >1.900000</td>
      <td id="T_19297_row44_col3" class="data row44 col3" >0.400000</td>
      <td id="T_19297_row44_col4" class="data row44 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row45" class="row_heading level0 row45" >45</th>
      <td id="T_19297_row45_col0" class="data row45 col0" >4.800000</td>
      <td id="T_19297_row45_col1" class="data row45 col1" >3.000000</td>
      <td id="T_19297_row45_col2" class="data row45 col2" >1.400000</td>
      <td id="T_19297_row45_col3" class="data row45 col3" >0.300000</td>
      <td id="T_19297_row45_col4" class="data row45 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row46" class="row_heading level0 row46" >46</th>
      <td id="T_19297_row46_col0" class="data row46 col0" >5.100000</td>
      <td id="T_19297_row46_col1" class="data row46 col1" >3.800000</td>
      <td id="T_19297_row46_col2" class="data row46 col2" >1.600000</td>
      <td id="T_19297_row46_col3" class="data row46 col3" >0.200000</td>
      <td id="T_19297_row46_col4" class="data row46 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row47" class="row_heading level0 row47" >47</th>
      <td id="T_19297_row47_col0" class="data row47 col0" >4.600000</td>
      <td id="T_19297_row47_col1" class="data row47 col1" >3.200000</td>
      <td id="T_19297_row47_col2" class="data row47 col2" >1.400000</td>
      <td id="T_19297_row47_col3" class="data row47 col3" >0.200000</td>
      <td id="T_19297_row47_col4" class="data row47 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row48" class="row_heading level0 row48" >48</th>
      <td id="T_19297_row48_col0" class="data row48 col0" >5.300000</td>
      <td id="T_19297_row48_col1" class="data row48 col1" >3.700000</td>
      <td id="T_19297_row48_col2" class="data row48 col2" >1.500000</td>
      <td id="T_19297_row48_col3" class="data row48 col3" >0.200000</td>
      <td id="T_19297_row48_col4" class="data row48 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row49" class="row_heading level0 row49" >49</th>
      <td id="T_19297_row49_col0" class="data row49 col0" >5.000000</td>
      <td id="T_19297_row49_col1" class="data row49 col1" >3.300000</td>
      <td id="T_19297_row49_col2" class="data row49 col2" >1.400000</td>
      <td id="T_19297_row49_col3" class="data row49 col3" >0.200000</td>
      <td id="T_19297_row49_col4" class="data row49 col4" >Setosa</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row50" class="row_heading level0 row50" >50</th>
      <td id="T_19297_row50_col0" class="data row50 col0" >7.000000</td>
      <td id="T_19297_row50_col1" class="data row50 col1" >3.200000</td>
      <td id="T_19297_row50_col2" class="data row50 col2" >4.700000</td>
      <td id="T_19297_row50_col3" class="data row50 col3" >1.400000</td>
      <td id="T_19297_row50_col4" class="data row50 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row51" class="row_heading level0 row51" >51</th>
      <td id="T_19297_row51_col0" class="data row51 col0" >6.400000</td>
      <td id="T_19297_row51_col1" class="data row51 col1" >3.200000</td>
      <td id="T_19297_row51_col2" class="data row51 col2" >4.500000</td>
      <td id="T_19297_row51_col3" class="data row51 col3" >1.500000</td>
      <td id="T_19297_row51_col4" class="data row51 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row52" class="row_heading level0 row52" >52</th>
      <td id="T_19297_row52_col0" class="data row52 col0" >6.900000</td>
      <td id="T_19297_row52_col1" class="data row52 col1" >3.100000</td>
      <td id="T_19297_row52_col2" class="data row52 col2" >4.900000</td>
      <td id="T_19297_row52_col3" class="data row52 col3" >1.500000</td>
      <td id="T_19297_row52_col4" class="data row52 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row53" class="row_heading level0 row53" >53</th>
      <td id="T_19297_row53_col0" class="data row53 col0" >5.500000</td>
      <td id="T_19297_row53_col1" class="data row53 col1" >2.300000</td>
      <td id="T_19297_row53_col2" class="data row53 col2" >4.000000</td>
      <td id="T_19297_row53_col3" class="data row53 col3" >1.300000</td>
      <td id="T_19297_row53_col4" class="data row53 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row54" class="row_heading level0 row54" >54</th>
      <td id="T_19297_row54_col0" class="data row54 col0" >6.500000</td>
      <td id="T_19297_row54_col1" class="data row54 col1" >2.800000</td>
      <td id="T_19297_row54_col2" class="data row54 col2" >4.600000</td>
      <td id="T_19297_row54_col3" class="data row54 col3" >1.500000</td>
      <td id="T_19297_row54_col4" class="data row54 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row55" class="row_heading level0 row55" >55</th>
      <td id="T_19297_row55_col0" class="data row55 col0" >5.700000</td>
      <td id="T_19297_row55_col1" class="data row55 col1" >2.800000</td>
      <td id="T_19297_row55_col2" class="data row55 col2" >4.500000</td>
      <td id="T_19297_row55_col3" class="data row55 col3" >1.300000</td>
      <td id="T_19297_row55_col4" class="data row55 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row56" class="row_heading level0 row56" >56</th>
      <td id="T_19297_row56_col0" class="data row56 col0" >6.300000</td>
      <td id="T_19297_row56_col1" class="data row56 col1" >3.300000</td>
      <td id="T_19297_row56_col2" class="data row56 col2" >4.700000</td>
      <td id="T_19297_row56_col3" class="data row56 col3" >1.600000</td>
      <td id="T_19297_row56_col4" class="data row56 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row57" class="row_heading level0 row57" >57</th>
      <td id="T_19297_row57_col0" class="data row57 col0" >4.900000</td>
      <td id="T_19297_row57_col1" class="data row57 col1" >2.400000</td>
      <td id="T_19297_row57_col2" class="data row57 col2" >3.300000</td>
      <td id="T_19297_row57_col3" class="data row57 col3" >1.000000</td>
      <td id="T_19297_row57_col4" class="data row57 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row58" class="row_heading level0 row58" >58</th>
      <td id="T_19297_row58_col0" class="data row58 col0" >6.600000</td>
      <td id="T_19297_row58_col1" class="data row58 col1" >2.900000</td>
      <td id="T_19297_row58_col2" class="data row58 col2" >4.600000</td>
      <td id="T_19297_row58_col3" class="data row58 col3" >1.300000</td>
      <td id="T_19297_row58_col4" class="data row58 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row59" class="row_heading level0 row59" >59</th>
      <td id="T_19297_row59_col0" class="data row59 col0" >5.200000</td>
      <td id="T_19297_row59_col1" class="data row59 col1" >2.700000</td>
      <td id="T_19297_row59_col2" class="data row59 col2" >3.900000</td>
      <td id="T_19297_row59_col3" class="data row59 col3" >1.400000</td>
      <td id="T_19297_row59_col4" class="data row59 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row60" class="row_heading level0 row60" >60</th>
      <td id="T_19297_row60_col0" class="data row60 col0" >5.000000</td>
      <td id="T_19297_row60_col1" class="data row60 col1" >2.000000</td>
      <td id="T_19297_row60_col2" class="data row60 col2" >3.500000</td>
      <td id="T_19297_row60_col3" class="data row60 col3" >1.000000</td>
      <td id="T_19297_row60_col4" class="data row60 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row61" class="row_heading level0 row61" >61</th>
      <td id="T_19297_row61_col0" class="data row61 col0" >5.900000</td>
      <td id="T_19297_row61_col1" class="data row61 col1" >3.000000</td>
      <td id="T_19297_row61_col2" class="data row61 col2" >4.200000</td>
      <td id="T_19297_row61_col3" class="data row61 col3" >1.500000</td>
      <td id="T_19297_row61_col4" class="data row61 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row62" class="row_heading level0 row62" >62</th>
      <td id="T_19297_row62_col0" class="data row62 col0" >6.000000</td>
      <td id="T_19297_row62_col1" class="data row62 col1" >2.200000</td>
      <td id="T_19297_row62_col2" class="data row62 col2" >4.000000</td>
      <td id="T_19297_row62_col3" class="data row62 col3" >1.000000</td>
      <td id="T_19297_row62_col4" class="data row62 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row63" class="row_heading level0 row63" >63</th>
      <td id="T_19297_row63_col0" class="data row63 col0" >6.100000</td>
      <td id="T_19297_row63_col1" class="data row63 col1" >2.900000</td>
      <td id="T_19297_row63_col2" class="data row63 col2" >4.700000</td>
      <td id="T_19297_row63_col3" class="data row63 col3" >1.400000</td>
      <td id="T_19297_row63_col4" class="data row63 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row64" class="row_heading level0 row64" >64</th>
      <td id="T_19297_row64_col0" class="data row64 col0" >5.600000</td>
      <td id="T_19297_row64_col1" class="data row64 col1" >2.900000</td>
      <td id="T_19297_row64_col2" class="data row64 col2" >3.600000</td>
      <td id="T_19297_row64_col3" class="data row64 col3" >1.300000</td>
      <td id="T_19297_row64_col4" class="data row64 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row65" class="row_heading level0 row65" >65</th>
      <td id="T_19297_row65_col0" class="data row65 col0" >6.700000</td>
      <td id="T_19297_row65_col1" class="data row65 col1" >3.100000</td>
      <td id="T_19297_row65_col2" class="data row65 col2" >4.400000</td>
      <td id="T_19297_row65_col3" class="data row65 col3" >1.400000</td>
      <td id="T_19297_row65_col4" class="data row65 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row66" class="row_heading level0 row66" >66</th>
      <td id="T_19297_row66_col0" class="data row66 col0" >5.600000</td>
      <td id="T_19297_row66_col1" class="data row66 col1" >3.000000</td>
      <td id="T_19297_row66_col2" class="data row66 col2" >4.500000</td>
      <td id="T_19297_row66_col3" class="data row66 col3" >1.500000</td>
      <td id="T_19297_row66_col4" class="data row66 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row67" class="row_heading level0 row67" >67</th>
      <td id="T_19297_row67_col0" class="data row67 col0" >5.800000</td>
      <td id="T_19297_row67_col1" class="data row67 col1" >2.700000</td>
      <td id="T_19297_row67_col2" class="data row67 col2" >4.100000</td>
      <td id="T_19297_row67_col3" class="data row67 col3" >1.000000</td>
      <td id="T_19297_row67_col4" class="data row67 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row68" class="row_heading level0 row68" >68</th>
      <td id="T_19297_row68_col0" class="data row68 col0" >6.200000</td>
      <td id="T_19297_row68_col1" class="data row68 col1" >2.200000</td>
      <td id="T_19297_row68_col2" class="data row68 col2" >4.500000</td>
      <td id="T_19297_row68_col3" class="data row68 col3" >1.500000</td>
      <td id="T_19297_row68_col4" class="data row68 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row69" class="row_heading level0 row69" >69</th>
      <td id="T_19297_row69_col0" class="data row69 col0" >5.600000</td>
      <td id="T_19297_row69_col1" class="data row69 col1" >2.500000</td>
      <td id="T_19297_row69_col2" class="data row69 col2" >3.900000</td>
      <td id="T_19297_row69_col3" class="data row69 col3" >1.100000</td>
      <td id="T_19297_row69_col4" class="data row69 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row70" class="row_heading level0 row70" >70</th>
      <td id="T_19297_row70_col0" class="data row70 col0" >5.900000</td>
      <td id="T_19297_row70_col1" class="data row70 col1" >3.200000</td>
      <td id="T_19297_row70_col2" class="data row70 col2" >4.800000</td>
      <td id="T_19297_row70_col3" class="data row70 col3" >1.800000</td>
      <td id="T_19297_row70_col4" class="data row70 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row71" class="row_heading level0 row71" >71</th>
      <td id="T_19297_row71_col0" class="data row71 col0" >6.100000</td>
      <td id="T_19297_row71_col1" class="data row71 col1" >2.800000</td>
      <td id="T_19297_row71_col2" class="data row71 col2" >4.000000</td>
      <td id="T_19297_row71_col3" class="data row71 col3" >1.300000</td>
      <td id="T_19297_row71_col4" class="data row71 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row72" class="row_heading level0 row72" >72</th>
      <td id="T_19297_row72_col0" class="data row72 col0" >6.300000</td>
      <td id="T_19297_row72_col1" class="data row72 col1" >2.500000</td>
      <td id="T_19297_row72_col2" class="data row72 col2" >4.900000</td>
      <td id="T_19297_row72_col3" class="data row72 col3" >1.500000</td>
      <td id="T_19297_row72_col4" class="data row72 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row73" class="row_heading level0 row73" >73</th>
      <td id="T_19297_row73_col0" class="data row73 col0" >6.100000</td>
      <td id="T_19297_row73_col1" class="data row73 col1" >2.800000</td>
      <td id="T_19297_row73_col2" class="data row73 col2" >4.700000</td>
      <td id="T_19297_row73_col3" class="data row73 col3" >1.200000</td>
      <td id="T_19297_row73_col4" class="data row73 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row74" class="row_heading level0 row74" >74</th>
      <td id="T_19297_row74_col0" class="data row74 col0" >6.400000</td>
      <td id="T_19297_row74_col1" class="data row74 col1" >2.900000</td>
      <td id="T_19297_row74_col2" class="data row74 col2" >4.300000</td>
      <td id="T_19297_row74_col3" class="data row74 col3" >1.300000</td>
      <td id="T_19297_row74_col4" class="data row74 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row75" class="row_heading level0 row75" >75</th>
      <td id="T_19297_row75_col0" class="data row75 col0" >6.600000</td>
      <td id="T_19297_row75_col1" class="data row75 col1" >3.000000</td>
      <td id="T_19297_row75_col2" class="data row75 col2" >4.400000</td>
      <td id="T_19297_row75_col3" class="data row75 col3" >1.400000</td>
      <td id="T_19297_row75_col4" class="data row75 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row76" class="row_heading level0 row76" >76</th>
      <td id="T_19297_row76_col0" class="data row76 col0" >6.800000</td>
      <td id="T_19297_row76_col1" class="data row76 col1" >2.800000</td>
      <td id="T_19297_row76_col2" class="data row76 col2" >4.800000</td>
      <td id="T_19297_row76_col3" class="data row76 col3" >1.400000</td>
      <td id="T_19297_row76_col4" class="data row76 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row77" class="row_heading level0 row77" >77</th>
      <td id="T_19297_row77_col0" class="data row77 col0" >6.700000</td>
      <td id="T_19297_row77_col1" class="data row77 col1" >3.000000</td>
      <td id="T_19297_row77_col2" class="data row77 col2" >5.000000</td>
      <td id="T_19297_row77_col3" class="data row77 col3" >1.700000</td>
      <td id="T_19297_row77_col4" class="data row77 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row78" class="row_heading level0 row78" >78</th>
      <td id="T_19297_row78_col0" class="data row78 col0" >6.000000</td>
      <td id="T_19297_row78_col1" class="data row78 col1" >2.900000</td>
      <td id="T_19297_row78_col2" class="data row78 col2" >4.500000</td>
      <td id="T_19297_row78_col3" class="data row78 col3" >1.500000</td>
      <td id="T_19297_row78_col4" class="data row78 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row79" class="row_heading level0 row79" >79</th>
      <td id="T_19297_row79_col0" class="data row79 col0" >5.700000</td>
      <td id="T_19297_row79_col1" class="data row79 col1" >2.600000</td>
      <td id="T_19297_row79_col2" class="data row79 col2" >3.500000</td>
      <td id="T_19297_row79_col3" class="data row79 col3" >1.000000</td>
      <td id="T_19297_row79_col4" class="data row79 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row80" class="row_heading level0 row80" >80</th>
      <td id="T_19297_row80_col0" class="data row80 col0" >5.500000</td>
      <td id="T_19297_row80_col1" class="data row80 col1" >2.400000</td>
      <td id="T_19297_row80_col2" class="data row80 col2" >3.800000</td>
      <td id="T_19297_row80_col3" class="data row80 col3" >1.100000</td>
      <td id="T_19297_row80_col4" class="data row80 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row81" class="row_heading level0 row81" >81</th>
      <td id="T_19297_row81_col0" class="data row81 col0" >5.500000</td>
      <td id="T_19297_row81_col1" class="data row81 col1" >2.400000</td>
      <td id="T_19297_row81_col2" class="data row81 col2" >3.700000</td>
      <td id="T_19297_row81_col3" class="data row81 col3" >1.000000</td>
      <td id="T_19297_row81_col4" class="data row81 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row82" class="row_heading level0 row82" >82</th>
      <td id="T_19297_row82_col0" class="data row82 col0" >5.800000</td>
      <td id="T_19297_row82_col1" class="data row82 col1" >2.700000</td>
      <td id="T_19297_row82_col2" class="data row82 col2" >3.900000</td>
      <td id="T_19297_row82_col3" class="data row82 col3" >1.200000</td>
      <td id="T_19297_row82_col4" class="data row82 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row83" class="row_heading level0 row83" >83</th>
      <td id="T_19297_row83_col0" class="data row83 col0" >6.000000</td>
      <td id="T_19297_row83_col1" class="data row83 col1" >2.700000</td>
      <td id="T_19297_row83_col2" class="data row83 col2" >5.100000</td>
      <td id="T_19297_row83_col3" class="data row83 col3" >1.600000</td>
      <td id="T_19297_row83_col4" class="data row83 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row84" class="row_heading level0 row84" >84</th>
      <td id="T_19297_row84_col0" class="data row84 col0" >5.400000</td>
      <td id="T_19297_row84_col1" class="data row84 col1" >3.000000</td>
      <td id="T_19297_row84_col2" class="data row84 col2" >4.500000</td>
      <td id="T_19297_row84_col3" class="data row84 col3" >1.500000</td>
      <td id="T_19297_row84_col4" class="data row84 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row85" class="row_heading level0 row85" >85</th>
      <td id="T_19297_row85_col0" class="data row85 col0" >6.000000</td>
      <td id="T_19297_row85_col1" class="data row85 col1" >3.400000</td>
      <td id="T_19297_row85_col2" class="data row85 col2" >4.500000</td>
      <td id="T_19297_row85_col3" class="data row85 col3" >1.600000</td>
      <td id="T_19297_row85_col4" class="data row85 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row86" class="row_heading level0 row86" >86</th>
      <td id="T_19297_row86_col0" class="data row86 col0" >6.700000</td>
      <td id="T_19297_row86_col1" class="data row86 col1" >3.100000</td>
      <td id="T_19297_row86_col2" class="data row86 col2" >4.700000</td>
      <td id="T_19297_row86_col3" class="data row86 col3" >1.500000</td>
      <td id="T_19297_row86_col4" class="data row86 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row87" class="row_heading level0 row87" >87</th>
      <td id="T_19297_row87_col0" class="data row87 col0" >6.300000</td>
      <td id="T_19297_row87_col1" class="data row87 col1" >2.300000</td>
      <td id="T_19297_row87_col2" class="data row87 col2" >4.400000</td>
      <td id="T_19297_row87_col3" class="data row87 col3" >1.300000</td>
      <td id="T_19297_row87_col4" class="data row87 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row88" class="row_heading level0 row88" >88</th>
      <td id="T_19297_row88_col0" class="data row88 col0" >5.600000</td>
      <td id="T_19297_row88_col1" class="data row88 col1" >3.000000</td>
      <td id="T_19297_row88_col2" class="data row88 col2" >4.100000</td>
      <td id="T_19297_row88_col3" class="data row88 col3" >1.300000</td>
      <td id="T_19297_row88_col4" class="data row88 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row89" class="row_heading level0 row89" >89</th>
      <td id="T_19297_row89_col0" class="data row89 col0" >5.500000</td>
      <td id="T_19297_row89_col1" class="data row89 col1" >2.500000</td>
      <td id="T_19297_row89_col2" class="data row89 col2" >4.000000</td>
      <td id="T_19297_row89_col3" class="data row89 col3" >1.300000</td>
      <td id="T_19297_row89_col4" class="data row89 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row90" class="row_heading level0 row90" >90</th>
      <td id="T_19297_row90_col0" class="data row90 col0" >5.500000</td>
      <td id="T_19297_row90_col1" class="data row90 col1" >2.600000</td>
      <td id="T_19297_row90_col2" class="data row90 col2" >4.400000</td>
      <td id="T_19297_row90_col3" class="data row90 col3" >1.200000</td>
      <td id="T_19297_row90_col4" class="data row90 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row91" class="row_heading level0 row91" >91</th>
      <td id="T_19297_row91_col0" class="data row91 col0" >6.100000</td>
      <td id="T_19297_row91_col1" class="data row91 col1" >3.000000</td>
      <td id="T_19297_row91_col2" class="data row91 col2" >4.600000</td>
      <td id="T_19297_row91_col3" class="data row91 col3" >1.400000</td>
      <td id="T_19297_row91_col4" class="data row91 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row92" class="row_heading level0 row92" >92</th>
      <td id="T_19297_row92_col0" class="data row92 col0" >5.800000</td>
      <td id="T_19297_row92_col1" class="data row92 col1" >2.600000</td>
      <td id="T_19297_row92_col2" class="data row92 col2" >4.000000</td>
      <td id="T_19297_row92_col3" class="data row92 col3" >1.200000</td>
      <td id="T_19297_row92_col4" class="data row92 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row93" class="row_heading level0 row93" >93</th>
      <td id="T_19297_row93_col0" class="data row93 col0" >5.000000</td>
      <td id="T_19297_row93_col1" class="data row93 col1" >2.300000</td>
      <td id="T_19297_row93_col2" class="data row93 col2" >3.300000</td>
      <td id="T_19297_row93_col3" class="data row93 col3" >1.000000</td>
      <td id="T_19297_row93_col4" class="data row93 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row94" class="row_heading level0 row94" >94</th>
      <td id="T_19297_row94_col0" class="data row94 col0" >5.600000</td>
      <td id="T_19297_row94_col1" class="data row94 col1" >2.700000</td>
      <td id="T_19297_row94_col2" class="data row94 col2" >4.200000</td>
      <td id="T_19297_row94_col3" class="data row94 col3" >1.300000</td>
      <td id="T_19297_row94_col4" class="data row94 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row95" class="row_heading level0 row95" >95</th>
      <td id="T_19297_row95_col0" class="data row95 col0" >5.700000</td>
      <td id="T_19297_row95_col1" class="data row95 col1" >3.000000</td>
      <td id="T_19297_row95_col2" class="data row95 col2" >4.200000</td>
      <td id="T_19297_row95_col3" class="data row95 col3" >1.200000</td>
      <td id="T_19297_row95_col4" class="data row95 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row96" class="row_heading level0 row96" >96</th>
      <td id="T_19297_row96_col0" class="data row96 col0" >5.700000</td>
      <td id="T_19297_row96_col1" class="data row96 col1" >2.900000</td>
      <td id="T_19297_row96_col2" class="data row96 col2" >4.200000</td>
      <td id="T_19297_row96_col3" class="data row96 col3" >1.300000</td>
      <td id="T_19297_row96_col4" class="data row96 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row97" class="row_heading level0 row97" >97</th>
      <td id="T_19297_row97_col0" class="data row97 col0" >6.200000</td>
      <td id="T_19297_row97_col1" class="data row97 col1" >2.900000</td>
      <td id="T_19297_row97_col2" class="data row97 col2" >4.300000</td>
      <td id="T_19297_row97_col3" class="data row97 col3" >1.300000</td>
      <td id="T_19297_row97_col4" class="data row97 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row98" class="row_heading level0 row98" >98</th>
      <td id="T_19297_row98_col0" class="data row98 col0" >5.100000</td>
      <td id="T_19297_row98_col1" class="data row98 col1" >2.500000</td>
      <td id="T_19297_row98_col2" class="data row98 col2" >3.000000</td>
      <td id="T_19297_row98_col3" class="data row98 col3" >1.100000</td>
      <td id="T_19297_row98_col4" class="data row98 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row99" class="row_heading level0 row99" >99</th>
      <td id="T_19297_row99_col0" class="data row99 col0" >5.700000</td>
      <td id="T_19297_row99_col1" class="data row99 col1" >2.800000</td>
      <td id="T_19297_row99_col2" class="data row99 col2" >4.100000</td>
      <td id="T_19297_row99_col3" class="data row99 col3" >1.300000</td>
      <td id="T_19297_row99_col4" class="data row99 col4" >Versicolor</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row100" class="row_heading level0 row100" >100</th>
      <td id="T_19297_row100_col0" class="data row100 col0" >6.300000</td>
      <td id="T_19297_row100_col1" class="data row100 col1" >3.300000</td>
      <td id="T_19297_row100_col2" class="data row100 col2" >6.000000</td>
      <td id="T_19297_row100_col3" class="data row100 col3" >2.500000</td>
      <td id="T_19297_row100_col4" class="data row100 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row101" class="row_heading level0 row101" >101</th>
      <td id="T_19297_row101_col0" class="data row101 col0" >5.800000</td>
      <td id="T_19297_row101_col1" class="data row101 col1" >2.700000</td>
      <td id="T_19297_row101_col2" class="data row101 col2" >5.100000</td>
      <td id="T_19297_row101_col3" class="data row101 col3" >1.900000</td>
      <td id="T_19297_row101_col4" class="data row101 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row102" class="row_heading level0 row102" >102</th>
      <td id="T_19297_row102_col0" class="data row102 col0" >7.100000</td>
      <td id="T_19297_row102_col1" class="data row102 col1" >3.000000</td>
      <td id="T_19297_row102_col2" class="data row102 col2" >5.900000</td>
      <td id="T_19297_row102_col3" class="data row102 col3" >2.100000</td>
      <td id="T_19297_row102_col4" class="data row102 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row103" class="row_heading level0 row103" >103</th>
      <td id="T_19297_row103_col0" class="data row103 col0" >6.300000</td>
      <td id="T_19297_row103_col1" class="data row103 col1" >2.900000</td>
      <td id="T_19297_row103_col2" class="data row103 col2" >5.600000</td>
      <td id="T_19297_row103_col3" class="data row103 col3" >1.800000</td>
      <td id="T_19297_row103_col4" class="data row103 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row104" class="row_heading level0 row104" >104</th>
      <td id="T_19297_row104_col0" class="data row104 col0" >6.500000</td>
      <td id="T_19297_row104_col1" class="data row104 col1" >3.000000</td>
      <td id="T_19297_row104_col2" class="data row104 col2" >5.800000</td>
      <td id="T_19297_row104_col3" class="data row104 col3" >2.200000</td>
      <td id="T_19297_row104_col4" class="data row104 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row105" class="row_heading level0 row105" >105</th>
      <td id="T_19297_row105_col0" class="data row105 col0" >7.600000</td>
      <td id="T_19297_row105_col1" class="data row105 col1" >3.000000</td>
      <td id="T_19297_row105_col2" class="data row105 col2" >6.600000</td>
      <td id="T_19297_row105_col3" class="data row105 col3" >2.100000</td>
      <td id="T_19297_row105_col4" class="data row105 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row106" class="row_heading level0 row106" >106</th>
      <td id="T_19297_row106_col0" class="data row106 col0" >4.900000</td>
      <td id="T_19297_row106_col1" class="data row106 col1" >2.500000</td>
      <td id="T_19297_row106_col2" class="data row106 col2" >4.500000</td>
      <td id="T_19297_row106_col3" class="data row106 col3" >1.700000</td>
      <td id="T_19297_row106_col4" class="data row106 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row107" class="row_heading level0 row107" >107</th>
      <td id="T_19297_row107_col0" class="data row107 col0" >7.300000</td>
      <td id="T_19297_row107_col1" class="data row107 col1" >2.900000</td>
      <td id="T_19297_row107_col2" class="data row107 col2" >6.300000</td>
      <td id="T_19297_row107_col3" class="data row107 col3" >1.800000</td>
      <td id="T_19297_row107_col4" class="data row107 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row108" class="row_heading level0 row108" >108</th>
      <td id="T_19297_row108_col0" class="data row108 col0" >6.700000</td>
      <td id="T_19297_row108_col1" class="data row108 col1" >2.500000</td>
      <td id="T_19297_row108_col2" class="data row108 col2" >5.800000</td>
      <td id="T_19297_row108_col3" class="data row108 col3" >1.800000</td>
      <td id="T_19297_row108_col4" class="data row108 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row109" class="row_heading level0 row109" >109</th>
      <td id="T_19297_row109_col0" class="data row109 col0" >7.200000</td>
      <td id="T_19297_row109_col1" class="data row109 col1" >3.600000</td>
      <td id="T_19297_row109_col2" class="data row109 col2" >6.100000</td>
      <td id="T_19297_row109_col3" class="data row109 col3" >2.500000</td>
      <td id="T_19297_row109_col4" class="data row109 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row110" class="row_heading level0 row110" >110</th>
      <td id="T_19297_row110_col0" class="data row110 col0" >6.500000</td>
      <td id="T_19297_row110_col1" class="data row110 col1" >3.200000</td>
      <td id="T_19297_row110_col2" class="data row110 col2" >5.100000</td>
      <td id="T_19297_row110_col3" class="data row110 col3" >2.000000</td>
      <td id="T_19297_row110_col4" class="data row110 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row111" class="row_heading level0 row111" >111</th>
      <td id="T_19297_row111_col0" class="data row111 col0" >6.400000</td>
      <td id="T_19297_row111_col1" class="data row111 col1" >2.700000</td>
      <td id="T_19297_row111_col2" class="data row111 col2" >5.300000</td>
      <td id="T_19297_row111_col3" class="data row111 col3" >1.900000</td>
      <td id="T_19297_row111_col4" class="data row111 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row112" class="row_heading level0 row112" >112</th>
      <td id="T_19297_row112_col0" class="data row112 col0" >6.800000</td>
      <td id="T_19297_row112_col1" class="data row112 col1" >3.000000</td>
      <td id="T_19297_row112_col2" class="data row112 col2" >5.500000</td>
      <td id="T_19297_row112_col3" class="data row112 col3" >2.100000</td>
      <td id="T_19297_row112_col4" class="data row112 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row113" class="row_heading level0 row113" >113</th>
      <td id="T_19297_row113_col0" class="data row113 col0" >5.700000</td>
      <td id="T_19297_row113_col1" class="data row113 col1" >2.500000</td>
      <td id="T_19297_row113_col2" class="data row113 col2" >5.000000</td>
      <td id="T_19297_row113_col3" class="data row113 col3" >2.000000</td>
      <td id="T_19297_row113_col4" class="data row113 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row114" class="row_heading level0 row114" >114</th>
      <td id="T_19297_row114_col0" class="data row114 col0" >5.800000</td>
      <td id="T_19297_row114_col1" class="data row114 col1" >2.800000</td>
      <td id="T_19297_row114_col2" class="data row114 col2" >5.100000</td>
      <td id="T_19297_row114_col3" class="data row114 col3" >2.400000</td>
      <td id="T_19297_row114_col4" class="data row114 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row115" class="row_heading level0 row115" >115</th>
      <td id="T_19297_row115_col0" class="data row115 col0" >6.400000</td>
      <td id="T_19297_row115_col1" class="data row115 col1" >3.200000</td>
      <td id="T_19297_row115_col2" class="data row115 col2" >5.300000</td>
      <td id="T_19297_row115_col3" class="data row115 col3" >2.300000</td>
      <td id="T_19297_row115_col4" class="data row115 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row116" class="row_heading level0 row116" >116</th>
      <td id="T_19297_row116_col0" class="data row116 col0" >6.500000</td>
      <td id="T_19297_row116_col1" class="data row116 col1" >3.000000</td>
      <td id="T_19297_row116_col2" class="data row116 col2" >5.500000</td>
      <td id="T_19297_row116_col3" class="data row116 col3" >1.800000</td>
      <td id="T_19297_row116_col4" class="data row116 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row117" class="row_heading level0 row117" >117</th>
      <td id="T_19297_row117_col0" class="data row117 col0" >7.700000</td>
      <td id="T_19297_row117_col1" class="data row117 col1" >3.800000</td>
      <td id="T_19297_row117_col2" class="data row117 col2" >6.700000</td>
      <td id="T_19297_row117_col3" class="data row117 col3" >2.200000</td>
      <td id="T_19297_row117_col4" class="data row117 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row118" class="row_heading level0 row118" >118</th>
      <td id="T_19297_row118_col0" class="data row118 col0" >7.700000</td>
      <td id="T_19297_row118_col1" class="data row118 col1" >2.600000</td>
      <td id="T_19297_row118_col2" class="data row118 col2" >6.900000</td>
      <td id="T_19297_row118_col3" class="data row118 col3" >2.300000</td>
      <td id="T_19297_row118_col4" class="data row118 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row119" class="row_heading level0 row119" >119</th>
      <td id="T_19297_row119_col0" class="data row119 col0" >6.000000</td>
      <td id="T_19297_row119_col1" class="data row119 col1" >2.200000</td>
      <td id="T_19297_row119_col2" class="data row119 col2" >5.000000</td>
      <td id="T_19297_row119_col3" class="data row119 col3" >1.500000</td>
      <td id="T_19297_row119_col4" class="data row119 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row120" class="row_heading level0 row120" >120</th>
      <td id="T_19297_row120_col0" class="data row120 col0" >6.900000</td>
      <td id="T_19297_row120_col1" class="data row120 col1" >3.200000</td>
      <td id="T_19297_row120_col2" class="data row120 col2" >5.700000</td>
      <td id="T_19297_row120_col3" class="data row120 col3" >2.300000</td>
      <td id="T_19297_row120_col4" class="data row120 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row121" class="row_heading level0 row121" >121</th>
      <td id="T_19297_row121_col0" class="data row121 col0" >5.600000</td>
      <td id="T_19297_row121_col1" class="data row121 col1" >2.800000</td>
      <td id="T_19297_row121_col2" class="data row121 col2" >4.900000</td>
      <td id="T_19297_row121_col3" class="data row121 col3" >2.000000</td>
      <td id="T_19297_row121_col4" class="data row121 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row122" class="row_heading level0 row122" >122</th>
      <td id="T_19297_row122_col0" class="data row122 col0" >7.700000</td>
      <td id="T_19297_row122_col1" class="data row122 col1" >2.800000</td>
      <td id="T_19297_row122_col2" class="data row122 col2" >6.700000</td>
      <td id="T_19297_row122_col3" class="data row122 col3" >2.000000</td>
      <td id="T_19297_row122_col4" class="data row122 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row123" class="row_heading level0 row123" >123</th>
      <td id="T_19297_row123_col0" class="data row123 col0" >6.300000</td>
      <td id="T_19297_row123_col1" class="data row123 col1" >2.700000</td>
      <td id="T_19297_row123_col2" class="data row123 col2" >4.900000</td>
      <td id="T_19297_row123_col3" class="data row123 col3" >1.800000</td>
      <td id="T_19297_row123_col4" class="data row123 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row124" class="row_heading level0 row124" >124</th>
      <td id="T_19297_row124_col0" class="data row124 col0" >6.700000</td>
      <td id="T_19297_row124_col1" class="data row124 col1" >3.300000</td>
      <td id="T_19297_row124_col2" class="data row124 col2" >5.700000</td>
      <td id="T_19297_row124_col3" class="data row124 col3" >2.100000</td>
      <td id="T_19297_row124_col4" class="data row124 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row125" class="row_heading level0 row125" >125</th>
      <td id="T_19297_row125_col0" class="data row125 col0" >7.200000</td>
      <td id="T_19297_row125_col1" class="data row125 col1" >3.200000</td>
      <td id="T_19297_row125_col2" class="data row125 col2" >6.000000</td>
      <td id="T_19297_row125_col3" class="data row125 col3" >1.800000</td>
      <td id="T_19297_row125_col4" class="data row125 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row126" class="row_heading level0 row126" >126</th>
      <td id="T_19297_row126_col0" class="data row126 col0" >6.200000</td>
      <td id="T_19297_row126_col1" class="data row126 col1" >2.800000</td>
      <td id="T_19297_row126_col2" class="data row126 col2" >4.800000</td>
      <td id="T_19297_row126_col3" class="data row126 col3" >1.800000</td>
      <td id="T_19297_row126_col4" class="data row126 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row127" class="row_heading level0 row127" >127</th>
      <td id="T_19297_row127_col0" class="data row127 col0" >6.100000</td>
      <td id="T_19297_row127_col1" class="data row127 col1" >3.000000</td>
      <td id="T_19297_row127_col2" class="data row127 col2" >4.900000</td>
      <td id="T_19297_row127_col3" class="data row127 col3" >1.800000</td>
      <td id="T_19297_row127_col4" class="data row127 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row128" class="row_heading level0 row128" >128</th>
      <td id="T_19297_row128_col0" class="data row128 col0" >6.400000</td>
      <td id="T_19297_row128_col1" class="data row128 col1" >2.800000</td>
      <td id="T_19297_row128_col2" class="data row128 col2" >5.600000</td>
      <td id="T_19297_row128_col3" class="data row128 col3" >2.100000</td>
      <td id="T_19297_row128_col4" class="data row128 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row129" class="row_heading level0 row129" >129</th>
      <td id="T_19297_row129_col0" class="data row129 col0" >7.200000</td>
      <td id="T_19297_row129_col1" class="data row129 col1" >3.000000</td>
      <td id="T_19297_row129_col2" class="data row129 col2" >5.800000</td>
      <td id="T_19297_row129_col3" class="data row129 col3" >1.600000</td>
      <td id="T_19297_row129_col4" class="data row129 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row130" class="row_heading level0 row130" >130</th>
      <td id="T_19297_row130_col0" class="data row130 col0" >7.400000</td>
      <td id="T_19297_row130_col1" class="data row130 col1" >2.800000</td>
      <td id="T_19297_row130_col2" class="data row130 col2" >6.100000</td>
      <td id="T_19297_row130_col3" class="data row130 col3" >1.900000</td>
      <td id="T_19297_row130_col4" class="data row130 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row131" class="row_heading level0 row131" >131</th>
      <td id="T_19297_row131_col0" class="data row131 col0" >7.900000</td>
      <td id="T_19297_row131_col1" class="data row131 col1" >3.800000</td>
      <td id="T_19297_row131_col2" class="data row131 col2" >6.400000</td>
      <td id="T_19297_row131_col3" class="data row131 col3" >2.000000</td>
      <td id="T_19297_row131_col4" class="data row131 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row132" class="row_heading level0 row132" >132</th>
      <td id="T_19297_row132_col0" class="data row132 col0" >6.400000</td>
      <td id="T_19297_row132_col1" class="data row132 col1" >2.800000</td>
      <td id="T_19297_row132_col2" class="data row132 col2" >5.600000</td>
      <td id="T_19297_row132_col3" class="data row132 col3" >2.200000</td>
      <td id="T_19297_row132_col4" class="data row132 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row133" class="row_heading level0 row133" >133</th>
      <td id="T_19297_row133_col0" class="data row133 col0" >6.300000</td>
      <td id="T_19297_row133_col1" class="data row133 col1" >2.800000</td>
      <td id="T_19297_row133_col2" class="data row133 col2" >5.100000</td>
      <td id="T_19297_row133_col3" class="data row133 col3" >1.500000</td>
      <td id="T_19297_row133_col4" class="data row133 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row134" class="row_heading level0 row134" >134</th>
      <td id="T_19297_row134_col0" class="data row134 col0" >6.100000</td>
      <td id="T_19297_row134_col1" class="data row134 col1" >2.600000</td>
      <td id="T_19297_row134_col2" class="data row134 col2" >5.600000</td>
      <td id="T_19297_row134_col3" class="data row134 col3" >1.400000</td>
      <td id="T_19297_row134_col4" class="data row134 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row135" class="row_heading level0 row135" >135</th>
      <td id="T_19297_row135_col0" class="data row135 col0" >7.700000</td>
      <td id="T_19297_row135_col1" class="data row135 col1" >3.000000</td>
      <td id="T_19297_row135_col2" class="data row135 col2" >6.100000</td>
      <td id="T_19297_row135_col3" class="data row135 col3" >2.300000</td>
      <td id="T_19297_row135_col4" class="data row135 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row136" class="row_heading level0 row136" >136</th>
      <td id="T_19297_row136_col0" class="data row136 col0" >6.300000</td>
      <td id="T_19297_row136_col1" class="data row136 col1" >3.400000</td>
      <td id="T_19297_row136_col2" class="data row136 col2" >5.600000</td>
      <td id="T_19297_row136_col3" class="data row136 col3" >2.400000</td>
      <td id="T_19297_row136_col4" class="data row136 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row137" class="row_heading level0 row137" >137</th>
      <td id="T_19297_row137_col0" class="data row137 col0" >6.400000</td>
      <td id="T_19297_row137_col1" class="data row137 col1" >3.100000</td>
      <td id="T_19297_row137_col2" class="data row137 col2" >5.500000</td>
      <td id="T_19297_row137_col3" class="data row137 col3" >1.800000</td>
      <td id="T_19297_row137_col4" class="data row137 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row138" class="row_heading level0 row138" >138</th>
      <td id="T_19297_row138_col0" class="data row138 col0" >6.000000</td>
      <td id="T_19297_row138_col1" class="data row138 col1" >3.000000</td>
      <td id="T_19297_row138_col2" class="data row138 col2" >4.800000</td>
      <td id="T_19297_row138_col3" class="data row138 col3" >1.800000</td>
      <td id="T_19297_row138_col4" class="data row138 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row139" class="row_heading level0 row139" >139</th>
      <td id="T_19297_row139_col0" class="data row139 col0" >6.900000</td>
      <td id="T_19297_row139_col1" class="data row139 col1" >3.100000</td>
      <td id="T_19297_row139_col2" class="data row139 col2" >5.400000</td>
      <td id="T_19297_row139_col3" class="data row139 col3" >2.100000</td>
      <td id="T_19297_row139_col4" class="data row139 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row140" class="row_heading level0 row140" >140</th>
      <td id="T_19297_row140_col0" class="data row140 col0" >6.700000</td>
      <td id="T_19297_row140_col1" class="data row140 col1" >3.100000</td>
      <td id="T_19297_row140_col2" class="data row140 col2" >5.600000</td>
      <td id="T_19297_row140_col3" class="data row140 col3" >2.400000</td>
      <td id="T_19297_row140_col4" class="data row140 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row141" class="row_heading level0 row141" >141</th>
      <td id="T_19297_row141_col0" class="data row141 col0" >6.900000</td>
      <td id="T_19297_row141_col1" class="data row141 col1" >3.100000</td>
      <td id="T_19297_row141_col2" class="data row141 col2" >5.100000</td>
      <td id="T_19297_row141_col3" class="data row141 col3" >2.300000</td>
      <td id="T_19297_row141_col4" class="data row141 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row142" class="row_heading level0 row142" >142</th>
      <td id="T_19297_row142_col0" class="data row142 col0" >5.800000</td>
      <td id="T_19297_row142_col1" class="data row142 col1" >2.700000</td>
      <td id="T_19297_row142_col2" class="data row142 col2" >5.100000</td>
      <td id="T_19297_row142_col3" class="data row142 col3" >1.900000</td>
      <td id="T_19297_row142_col4" class="data row142 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row143" class="row_heading level0 row143" >143</th>
      <td id="T_19297_row143_col0" class="data row143 col0" >6.800000</td>
      <td id="T_19297_row143_col1" class="data row143 col1" >3.200000</td>
      <td id="T_19297_row143_col2" class="data row143 col2" >5.900000</td>
      <td id="T_19297_row143_col3" class="data row143 col3" >2.300000</td>
      <td id="T_19297_row143_col4" class="data row143 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row144" class="row_heading level0 row144" >144</th>
      <td id="T_19297_row144_col0" class="data row144 col0" >6.700000</td>
      <td id="T_19297_row144_col1" class="data row144 col1" >3.300000</td>
      <td id="T_19297_row144_col2" class="data row144 col2" >5.700000</td>
      <td id="T_19297_row144_col3" class="data row144 col3" >2.500000</td>
      <td id="T_19297_row144_col4" class="data row144 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row145" class="row_heading level0 row145" >145</th>
      <td id="T_19297_row145_col0" class="data row145 col0" >6.700000</td>
      <td id="T_19297_row145_col1" class="data row145 col1" >3.000000</td>
      <td id="T_19297_row145_col2" class="data row145 col2" >5.200000</td>
      <td id="T_19297_row145_col3" class="data row145 col3" >2.300000</td>
      <td id="T_19297_row145_col4" class="data row145 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row146" class="row_heading level0 row146" >146</th>
      <td id="T_19297_row146_col0" class="data row146 col0" >6.300000</td>
      <td id="T_19297_row146_col1" class="data row146 col1" >2.500000</td>
      <td id="T_19297_row146_col2" class="data row146 col2" >5.000000</td>
      <td id="T_19297_row146_col3" class="data row146 col3" >1.900000</td>
      <td id="T_19297_row146_col4" class="data row146 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row147" class="row_heading level0 row147" >147</th>
      <td id="T_19297_row147_col0" class="data row147 col0" >6.500000</td>
      <td id="T_19297_row147_col1" class="data row147 col1" >3.000000</td>
      <td id="T_19297_row147_col2" class="data row147 col2" >5.200000</td>
      <td id="T_19297_row147_col3" class="data row147 col3" >2.000000</td>
      <td id="T_19297_row147_col4" class="data row147 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row148" class="row_heading level0 row148" >148</th>
      <td id="T_19297_row148_col0" class="data row148 col0" >6.200000</td>
      <td id="T_19297_row148_col1" class="data row148 col1" >3.400000</td>
      <td id="T_19297_row148_col2" class="data row148 col2" >5.400000</td>
      <td id="T_19297_row148_col3" class="data row148 col3" >2.300000</td>
      <td id="T_19297_row148_col4" class="data row148 col4" >Virginica</td>
    </tr>
    <tr>
      <th id="T_19297_level0_row149" class="row_heading level0 row149" >149</th>
      <td id="T_19297_row149_col0" class="data row149 col0" >5.900000</td>
      <td id="T_19297_row149_col1" class="data row149 col1" >3.000000</td>
      <td id="T_19297_row149_col2" class="data row149 col2" >5.100000</td>
      <td id="T_19297_row149_col3" class="data row149 col3" >1.800000</td>
      <td id="T_19297_row149_col4" class="data row149 col4" >Virginica</td>
    </tr>
  </tbody>
</table>





```python
iris_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal.length</th>
      <th>sepal.width</th>
      <th>petal.length</th>
      <th>petal.width</th>
      <th>variety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal.length</th>
      <th>sepal.width</th>
      <th>petal.length</th>
      <th>petal.width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.057333</td>
      <td>3.758000</td>
      <td>1.199333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.435866</td>
      <td>1.765298</td>
      <td>0.762238</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris_data.shape
```




    (150, 5)



# Classification
Classification in machine learning is the process of recognition, understanding, and grouping of objects and ideas into preset categories. It requires the use of machine learning algorithms that learn how to assign a class label to examples from the problem domain. There are many different types of classification tasks that you may encounter in machine learning and specialized approaches to modeling that may be used for each.


## Decision Tree

A Decision Tree is a Flow Chart, and can help you make decisions based on previous experience. we will use **Confusion Matrix** in order to evaluate the accuracy of our model.


```python
d = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
features=['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
Xtree = iris_data[features]
ytree = iris_data['variety'].map(d)

dfStyler = iris_data.style.set_properties(**{'text-align': 'left'})
dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
print(iris_data)
```

         sepal.length  sepal.width  petal.length  petal.width    variety
    0             5.1          3.5           1.4          0.2     Setosa
    1             4.9          3.0           1.4          0.2     Setosa
    2             4.7          3.2           1.3          0.2     Setosa
    3             4.6          3.1           1.5          0.2     Setosa
    4             5.0          3.6           1.4          0.2     Setosa
    ..            ...          ...           ...          ...        ...
    145           6.7          3.0           5.2          2.3  Virginica
    146           6.3          2.5           5.0          1.9  Virginica
    147           6.5          3.0           5.2          2.0  Virginica
    148           6.2          3.4           5.4          2.3  Virginica
    149           5.9          3.0           5.1          1.8  Virginica
    
    [150 rows x 5 columns]
    


```python
dtree = tree.DecisionTreeClassifier()
dtree.fit(Xtree, ytree)

#Plot the tree
plt.figure(figsize=(15,10))
tree.plot_tree(dtree, feature_names=features, fontsize=10)
plt.show()
```


    
![png](/assets/output_21_0.png)
    



```python
print(dtree.predict([[5.5, 4, 4, 1.5]]))
```

    [1]
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py:450: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names
      warnings.warn(
    

### Confusion Matrix
It is a table that is used in classification problems to assess where errors in the model were made.

The rows represent the actual classes the outcomes should have been. While the columns represent the predictions we have made. Using this table it is easy to see which predictions are wrong.


```python
clf = tree.DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(Xtree, ytree, test_size=0.33, random_state=0)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)

df_cfm = pd.DataFrame(cm, index = clf.classes_, columns = clf.classes_)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True)
plt.title("Decision Tree", fontsize = 22, fontweight="bold")
plt.xlabel("Predicted Label", fontsize = 22)
plt.ylabel("True Label", fontsize = 22)
sn.set(font_scale=1.4)
plt.show()
```


    
![png](/assets/output_24_0.png)
    


### AUC - ROC Curve

In classification, there are many different evaluation metrics. The most popular is accuracy, which measures how often the model is correct. This is a great metric because it is easy to understand and getting the most correct guesses is often desired. There are some cases where you might consider using another evaluation metric.

Another common metric is AUC, area under the receiver operating characteristic (ROC) curve. The Reciever operating characteristic curve plots the true positive (TP) rate versus the false positive (FP) rate at different classification thresholds. The thresholds are different probability cutoffs that separate the two classes in binary classification. It uses probability to tell us how well a model separates the classes.


```python
clf = tree.DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(Xsk, ysk, test_size=0.33, random_state=0)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)

df_cfm = pd.DataFrame(cm, index = clf.classes_, columns = clf.classes_)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True)
plt.title("Decision Tree", fontsize = 22, fontweight="bold")
plt.xlabel("Predicted Label", fontsize = 22)
plt.ylabel("True Label", fontsize = 22)
sn.set(font_scale=1.4)
plt.show()
```


    
![png](/assets/output_26_0.png)
    



```python
# Binarize the /assets/output
y = label_binarize(ysk, classes = clf.classes_)
n_classes = y.shape[1]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(Xsk, y, test_size=0.33, random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(
    clf
)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
```


```python
plt.figure()
lw = 2
plt.plot(
    fpr[2],
    tpr[2],
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc[2],
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
#plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()
#fig.savefig('curves.png')
```


    
![png](/assets/output_28_0.png)
    


### Cross Validation

When adjusting models we are aiming to increase overall model performance on unseen data. Hyperparameter tuning can lead to much better performance on test sets. However, optimizing parameters to the test set can lead information leakage causing the model to preform worse on unseen data. To correct for this we can perform cross validation.

To better understand CV, we will be performing different methods on the iris dataset.


```python
# K-Fold Cross Validation 

clf = DecisionTreeClassifier(random_state=42)

k_folds = KFold(n_splits = 5)

scores = cross_val_score(clf, Xsk, ysk, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
```

    Cross Validation Scores:  [1.         1.         0.83333333 0.93333333 0.8       ]
    Average CV Score:  0.9133333333333333
    Number of CV Scores used in Average:  5
    


```python
# Stratified K-Fold

clf = DecisionTreeClassifier(random_state=42)

sk_folds = StratifiedKFold(n_splits = 5)

scores = cross_val_score(clf, Xsk, ysk, cv = sk_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
```

    Cross Validation Scores:  [0.96666667 0.96666667 0.9        0.93333333 1.        ]
    Average CV Score:  0.9533333333333334
    Number of CV Scores used in Average:  5
    


```python
#Leave One Out
X, y = datasets.load_iris(return_X_y=True)

clf = DecisionTreeClassifier(random_state=42)

loo = LeaveOneOut()

scores = cross_val_score(clf, X, y, cv = loo)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
```

    Cross Validation Scores:  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1.
     1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 0. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1.]
    Average CV Score:  0.94
    Number of CV Scores used in Average:  150
    


```python
clf = DecisionTreeClassifier(random_state=42)

lpo = LeavePOut(p=2)

scores = cross_val_score(clf, Xsk, ysk, cv = lpo)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
```

    Cross Validation Scores:  [1. 1. 1. ... 1. 1. 1.]
    Average CV Score:  0.9382997762863534
    Number of CV Scores used in Average:  11175
    

### Ensemble


```python
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = train_test_split(Xtree, ytree, test_size = 0.25, random_state = 22)

dtree = DecisionTreeClassifier(random_state = 22)
dtree.fit(X_train,y_train)

y_pred = dtree.predict(X_test)

print("Train data accuracy:",accuracy_score(y_true = y_train, y_pred = dtree.predict(X_train)))
print("Test data accuracy:",accuracy_score(y_true = y_test, y_pred = y_pred))
```

    Train data accuracy: 1.0
    Test data accuracy: 0.9210526315789473
    


```python
from sklearn.ensemble import BaggingClassifier

X_train, X_test, y_train, y_test = train_test_split(Xtree, ytree, test_size = 0.25, random_state = 22)

estimator_range = [2,4,6,8,10,12,14,16,18,20]

models = []
scores = []

for n_estimators in estimator_range:

    # Create bagging classifier
    clf = BaggingClassifier(n_estimators = n_estimators, random_state = 22)

    # Fit the model
    clf.fit(X_train, y_train)

    # Append the model and score to their respective list
    models.append(clf)
    scores.append(accuracy_score(y_true = y_test, y_pred = clf.predict(X_test)))

# Generate the plot of scores against number of estimators
plt.figure(figsize=(9,6))
plt.plot(estimator_range, scores)

# Adjust labels and font (to make visable)
plt.xlabel("n_estimators", fontsize = 18)
plt.ylabel("score", fontsize = 18)
plt.tick_params(labelsize = 16)

# Visualize plot
plt.show()
```


    
![png](/assets/output_36_0.png)
    



```python
X_train, X_test, y_train, y_test = train_test_split(Xtree, ytree, test_size = 0.25, random_state = 22)

clf = BaggingClassifier(n_estimators = 12, oob_score = True,random_state = 22)

clf.fit(X_train, y_train)

plt.figure(figsize=(15, 10))

plot_tree(clf.estimators_[0], feature_names = features, fontsize=14)
```




    [Text(0.375, 0.9, 'petal.length <= 2.45\ngini = 0.661\nsamples = 71\nvalue = [35, 44, 33]'),
     Text(0.25, 0.7, 'gini = 0.0\nsamples = 23\nvalue = [35, 0, 0]'),
     Text(0.5, 0.7, 'petal.width <= 1.7\ngini = 0.49\nsamples = 48\nvalue = [0, 44, 33]'),
     Text(0.25, 0.5, 'petal.length <= 5.0\ngini = 0.044\nsamples = 26\nvalue = [0, 43, 1]'),
     Text(0.125, 0.3, 'gini = 0.0\nsamples = 24\nvalue = [0, 42, 0]'),
     Text(0.375, 0.3, 'sepal.length <= 6.15\ngini = 0.5\nsamples = 2\nvalue = [0, 1, 1]'),
     Text(0.25, 0.1, 'gini = 0.0\nsamples = 1\nvalue = [0, 1, 0]'),
     Text(0.5, 0.1, 'gini = 0.0\nsamples = 1\nvalue = [0, 0, 1]'),
     Text(0.75, 0.5, 'petal.length <= 4.85\ngini = 0.059\nsamples = 22\nvalue = [0, 1, 32]'),
     Text(0.625, 0.3, 'gini = 0.0\nsamples = 1\nvalue = [0, 1, 0]'),
     Text(0.875, 0.3, 'gini = 0.0\nsamples = 21\nvalue = [0, 0, 32]')]




    
![png](/assets/output_37_1.png)
    


## SVM


```python
clf = svm.LinearSVC(max_iter=3080)
X_train, X_test, y_train, y_test = train_test_split(Xtree, ytree, test_size = 0.33, random_state=0)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)

df_cfm = pd.DataFrame(cm, index = clf.classes_, columns = clf.classes_)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True)
plt.title("SVM", fontsize = 22, fontweight="bold")
plt.xlabel("Predicted Label", fontsize = 22)
plt.ylabel("True Label", fontsize = 22)
sn.set(font_scale=1.4)
plt.show()
```


    
![png](/assets/output_39_0.png)
    



```python
#report = classification_report(y_test, predictions, target_names = clf.classes_, labels=clf.classes_, zero_division = 0, /assets/output_dict=True)
#df = pd.DataFrame(report).transpose()
#df.to_csv("Report_SVM.csv")
```

## Random Forest


```python
clf = RandomForestClassifier()
```


```python
X_train, X_test, y_train, y_test = train_test_split(Xtree, ytree, test_size=0.33, random_state=0)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)

df_cfm = pd.DataFrame(cm, index = clf.classes_, columns = clf.classes_)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True)
plt.title("Random Forest", fontsize = 22, fontweight="bold")
plt.xlabel("Predicted Label", fontsize = 22)
plt.ylabel("True Label", fontsize = 22)
sn.set(font_scale=1.4)
plt.show()
```


    
![png](/assets/output_43_0.png)
    



```python
#report = classification_report(y_test, predictions, target_names = clf.classes_, labels=clf.classes_, zero_division = 0, /assets/output_dict=True)
#df = pd.DataFrame(report).transpose()
#df.to_csv("Report_RF.csv") 
```

## Logistic Regression 

### Grid Search


```python
logit = LogisticRegression(max_iter = 10000)

C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

scores = []

for choice in C:
  logit.set_params(C=choice)
  logit.fit(Xsk, ysk)
  scores.append(logit.score(Xsk, ysk))

print(scores)
```

    [0.9666666666666667, 0.9666666666666667, 0.9733333333333334, 0.9733333333333334, 0.98, 0.98, 0.9866666666666667, 0.9866666666666667]
    


```python
clf = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(Xtree, ytree, test_size=0.33, random_state=0)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)

df_cfm = pd.DataFrame(cm, index = clf.classes_, columns = clf.classes_)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True)
plt.title("Logistic Regression", fontsize = 22, fontweight="bold")
plt.xlabel("Predicted Label", fontsize = 22)
plt.ylabel("True Label", fontsize = 22)
sn.set(font_scale=1.4)
plt.show()
```


    
![png](/assets/output_48_0.png)
    



```python
#report = classification_report(y_test, predictions, target_names = clf.classes_, labels=clf.classes_, zero_division = 0, /assets/output_dict=True)
#df = pd.DataFrame(report).transpose()
#df.to_csv("Report_LR.csv") 
```

## Gaussian Naïve Bays


```python
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(Xtree, ytree, test_size=0.33, random_state=0)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)

df_cfm = pd.DataFrame(cm, index = clf.classes_, columns = clf.classes_)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True)
plt.title("Gaussian Naïve Bays", fontsize = 22, fontweight="bold")
plt.xlabel("Predicted Label", fontsize = 22)
plt.ylabel("True Label", fontsize = 22)
sn.set(font_scale=1.4)
plt.show()
```


    
![png](/assets/output_51_0.png)
    



```python
#report = classification_report(y_test, predictions, target_names = clf.classes_, labels=clf.classes_, zero_division = 0, /assets/output_dict=True)
#df = pd.DataFrame(report).transpose()
#df.to_csv("Report_GNB.csv") 
```

## KNN


```python
clf = KNeighborsClassifier(n_neighbors=1,)
X_train, X_test, y_train, y_test = train_test_split(Xtree, ytree, test_size =0.33, random_state=0)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)

df_cfm = pd.DataFrame(cm, index = clf.classes_, columns = clf.classes_)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True)
plt.title("K-NN", fontsize = 22, fontweight="bold")
plt.xlabel("Predicted Label", fontsize = 22)
plt.ylabel("True Label", fontsize = 22)
sn.set(font_scale=1.4)
plt.show()
```


    
![png](/assets/output_54_0.png)
    



```python
#report = classification_report(y_test, predictions, target_names = clf.classes_, labels=clf.classes_, zero_division = 0, /assets/output_dict=True)
#df = pd.DataFrame(report).transpose()
#df.to_csv("Report_KNN.csv") 
```

# Hierarchical Clustering

Hierarchical clustering is an unsupervised learning method for clustering data points. The algorithm builds clusters by measuring the dissimilarities between data. Unsupervised learning means that a model does not have to be trained, and we do not need a "target" variable. This method can be used on any data to visualize and interpret the relationship between individual data points.


```python
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
fig = plt.figure(figsize=(15,5))


data_to_analyze = iris_data[['petal.length', 'petal.width']]

# =============
# First subplot
# =============

ax = fig.add_subplot(1, 2, 1)
groups = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
groups.fit_predict(data_to_analyze)
plt.scatter(iris_data['petal.length'] ,iris_data['petal.width'], c= groups.labels_, cmap='cool')


# =============
# Secound subplot
# =============

ax = fig.add_subplot(1, 2, 2)
data_to_analyze = list(zip(iris_data['petal.length'], iris_data['petal.width']))
linkage_data = linkage(data_to_analyze, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.show()
```


    
![png](/assets/output_57_0.png)
    


## K-means


```python
from sklearn.cluster import KMeans

inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data_to_analyze)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
```


    
![png](/assets/output_59_1.png)
    



```python
kmeans = KMeans(n_clusters=3)
kmeans.fit(data_to_analyze)

plt.scatter(iris_data['petal.length'], iris_data['petal.width'], c=kmeans.labels_, cmap='cool')
plt.show()
```


    
![png](/assets/output_60_0.png)
