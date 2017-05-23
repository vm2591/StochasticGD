import GradientDescent as gd
import Plotter as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa' , -1 , 1)

X = df.iloc[0:100 , [0,2]].values

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()

ada = gd.AdalineSGD(n_iter = 15, eta = 0.01)
ada.fit(X_std,y)

pl.plot_decision_regions(X_std,y,classifier=ada)
plt.title('Gradient Descent')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc = 'upper left')


plt.show()
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('iterations')
plt.ylabel('Squared Error Sum')
plt.show()
