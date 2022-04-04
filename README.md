# DATA-410-Project5
# Variable Selection and Sparsity Patterns Using SCAD, SQRTLasso, Lasso, Ridge, and ElasticNet Penalties 

## Variable Selection Theoretical Background

Variable (or feature) selection is a statistical technique to choose the most important features from a dataset that contribute meaningful information for prediction. Choosing the best features is critical to building a useful regression or classification model. If additional non-useful variables are part of a machine learning model, then the generalization capability or the accuracy of classification can worsen for that particular model. Variable selection is particularly important in high-dimensional modeling. This paper will examine regression on a simulated dataset with 1200 features and 200 observations with a specified sparsity pattern to gauge effectiveness of particular regularization methods, namely SCAD, SQRTLasso, Lasso, Ridge, and ElasticNet. Essentially, regularization reduces the freedom of the model and places a penality on different terms in the model. Specifically, the penalty is placed on the beta coefficients of the features. In the case of the L1 and L2 penalty, the constraint is placed on the weights of the features like a bound for the sum of the squared weights or the sum of the absolute value of the weights. The L1 penalty of Lasso can eliminate some of the coefficients altogether because they are shrunk down to zero. 

Here is a graphical example of the Ridge and Lasso penalties (Jones):

![image](https://user-images.githubusercontent.com/76021844/161326208-8efa2ec3-04d9-4a25-b98a-c15f8aa09729.png)

SCAD stands for the smoothly clipped absolute deviations penalty and it is a type of penalized least squares penalty. SCAD is useful for encouraging sparse solutions in variable selection. Furthermore, SCAD allows for beta coefficients to be high even while encouraging a sparse solution. SCAD differs from Lasso in that Lasso has a monotonically increasing penalty whereas higher values of beta beyond a threshold are not penalized more under SCAD. Local quadratic approximations can be used to fit a model using SCAD. Below, a graphical representation of the SCAD penalty is shown (Jones):

![image](https://user-images.githubusercontent.com/76021844/161326849-8081d2f5-8b1d-4bf5-a7af-eae74b74069a.png)

Square Root Lasso is a modification of the original LASSO algorithm with the aim of encouraging sparse solutions. SQRTLasso is hypothesized to be particularly useful when the number of features, p, exceeds the number of observations, n. SQRTLasso was developed as a solution to convex conic programming problems and research has demonstrated that it has reached effectiveness levels of Lasso when sigma is known. 

## Import Necessary Libraries 
```
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import warnings
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from numba import jit, prange
```

## Writing SKLearn Compliant Functions

### SQRTLasso 
```
class SQRTLasso(BaseEstimator):
    def __init__(self, maxiter=50000, alpha=0.01):
        self.maxiter, self.alpha = maxiter, alpha
    
    def fit(self, x, y):
        alpha=self.alpha
        def f_obj(x,y,beta,alpha):
          n =len(x)
          output = np.linalg.norm((1/n)*(y-x.dot(beta)),ord=2) + alpha * np.linalg.norm(beta,ord=1)
          return output
        
        def f_grad(x,y,beta,alpha):
          n=x.shape[0]
          p=x.shape[1]
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output1 = (-1/np.sqrt(n))*np.transpose(x).dot((y-x.dot(beta)))/np.linalg.norm(y-x.dot(beta))
          output2 = (alpha)*np.sign(beta)
          return output1 + output2
        
        def objective(beta):
          return(f_obj(x,y,beta,alpha))

        def gradient(beta):
          return(f_grad(x,y,beta,alpha))
        
        beta0 = np.random.uniform(size=(x.shape[1],1))
        output = minimize(objective, beta0, method='L-BFGS-B', jac=gradient,options={'gtol': 1e-8, 'maxiter': self.maxiter,'maxls': 25,'disp': True})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
    
    def get_params(self, deep=True):
    # suppose this estimator has parameters "alpha" and "recursive"
      return {"alpha": self.alpha}

    def set_params(self, **parameters):
      for parameter, value in parameters.items():
          setattr(self, parameter, value)
      return self
```

### SCAD

** Based on the work of Andy Jones. 
```
@jit
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part
    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
```
```
class SCAD(BaseEstimator):
    def __init__(self, maxiter=50000, lam = .001, a=1.55):
      self.maxiter, self.a, self.lam = maxiter, a, lam

    def fit(self, X, y): 
      # we add aan extra columns of 1 for the intercept
      #X = np.c_[np.ones((n,1)),X]
      n = X.shape[0]
      p = X.shape[1]
      def scad(beta):
        beta = beta.flatten()
        beta = beta.reshape(-1,1)
        n = len(y)
        return 1/n*np.sum((y-X.dot(beta))**2) + np.sum(scad_penalty(beta,self.lam,self.a))
      
      def dscad(beta):
        beta = beta.flatten()
        beta = beta.reshape(-1,1)
        n = len(y)
        return np.array(-2/n*np.transpose(X).dot(y-X.dot(beta))+scad_derivative(beta,self.lam,self.a)).flatten()
      b0 = np.ones((p,1))
      output = minimize(scad, b0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 1e7,'maxls': 25,'disp': True})
      self.coef = output.x
      return output.x
    
    def predict(self, x): 
      return x.dot(self.coef)

    def get_params(self, deep=True):
    # suppose this estimator has parameters "alpha" and "recursive"
      return {"lam": self.lam, 'a': self.a}

    def set_params(self, **parameters):
      for parameter, value in parameters.items():
          setattr(self, parameter, value)
      return self
```

## Simulating a High Dimensional Dataset
```
n = 200 
p = 1200 
beta_star = np.array([1] * 7 + [0] * 25 + [.25] * 5 + [0] * 50 + [.7] * 15 + [0] * 1098)
rho = .8

# we need toeplitz ([1, .8, .8**2. .8**3, .8**4, ... .8**1199])
corr_vector = []
for i in range(p): 
  corr_vector.append(rho**i)

r = toeplitz(corr_vector)
mu = [0] * p 
sigma = 3.5

# Generate the random samples.
X = np.random.multivariate_normal(mu, r, size=n)
y = np.matmul(X,beta_star) + sigma*np.random.normal(0,1,n)
y = y.reshape(200, 1)
```

The variable pos will be used to keep track of how closely the penalization methods align with the ideal sparsity pattern defined. 
```
pos = np.where(beta_star != 0)
```

## Testing Regularization Methods 

#### SQRTLasso
```
model = SQRTLasso()
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  sqrt_lasso_params = [{'alpha': np.linspace(0.001, 1, 50)}]

  grid = GridSearchCV(model, sqrt_lasso_params, cv = 5, scoring='neg_mean_absolute_error')

  grid.fit(X, y)
```

```
kf = KFold(n_splits=10, shuffle=True, random_state=2021)
coeffs = []
rmse = []
l2_dist = []
for train_index , test_index in kf.split(X):
    X = pd.DataFrame(X)
    X_train , X_test = X.iloc[train_index,:].values,X.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    model = SQRTLasso(alpha = grid.best_params_.get('alpha'))
    model.fit(X_train, y_train)
    beta_hat = model.coef_
    pos_lasso = np.where(beta_hat != 0)
    coeffs.append(len(np.intersect1d(pos, pos_lasso)))
    yhat = model.predict(X_test)
    rmse.append(MSE(yhat, y_test)**.5)
    l2_dist.append(np.linalg.norm(model.coef_ - beta_star,ord = 2))
print("Average true non-zero coefficients:", np.mean(coeffs))
print("Average RMSE:", np.mean(rmse))
print("Average L2 Distance to Ideal:", np.mean(l2_dist))
```
Average true non-zero coefficients: 27.0

Average RMSE: 3.899887766593173

Average L2 Distance to Ideal: 1.5045835314990466

#### SCAD
```
model = SCAD()
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  scad_params = [{'lam': np.linspace(0.001, 1, 25), 'a': np.linspace(.1, 3, 25)}]

  grid = GridSearchCV(model, scad_params, cv = 5, scoring='neg_mean_absolute_error')

  grid.fit(X, y)
```


```
kf = KFold(n_splits=10, shuffle=True, random_state=2021)
coeffs = []
rmse = []
l2_dist = []
for train_index , test_index in kf.split(X):
    X = pd.DataFrame(X)
    X_train , X_test = X.iloc[train_index,:].values,X.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    model = SCAD(lam = grid.best_params_.get('lam'), a = grid.best_params_.get('a'))
    model.fit(X_train, y_train)
    beta_hat = model.coef
    pos_lasso = np.where(beta_hat != 0)
    coeffs.append(len(np.intersect1d(pos, pos_lasso)))
    yhat = model.predict(X_test)
    rmse.append(MSE(yhat, y_test)**.5)
    l2_dist.append(np.linalg.norm(model.coef - beta_star,ord = 2))
print("Average true non-zero coefficients:", np.mean(coeffs))
print("Average RMSE:", np.mean(rmse))
print("Average L2 Distance to Ideal:", np.mean(l2_dist))
```
Average true non-zero coefficients: 27.0

Average RMSE: 9.069864242200328

Average L2 Distance to Ideal: 6.343565875213969

#### Ridge
```
model = Ridge()
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  ridge_params = [{'alpha': np.linspace(0.001, 2, 50)}]

  grid = GridSearchCV(model, ridge_params, cv = 5, scoring='neg_mean_absolute_error')

  grid.fit(X, y)
```

```
kf = KFold(n_splits=10, shuffle=True, random_state=2021)
coeffs = []
rmse = []
l2_dist = []
for train_index , test_index in kf.split(X):
    X = pd.DataFrame(X)
    X_train , X_test = X.iloc[train_index,:].values,X.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    model = Ridge(alpha = grid.best_params_.get('alpha'))
    model.fit(X_train, y_train)
    beta_hat = model.coef_
    pos_lasso = np.where(beta_hat != 0)
    coeffs.append(len(np.intersect1d(pos, pos_lasso)))
    yhat = model.predict(X_test)
    rmse.append(MSE(yhat, y_test)**.5)
    l2_dist.append(np.linalg.norm(model.coef_ - beta_star,ord = 2))
print("Average true non-zero coefficients:", np.mean(coeffs))
print("Average RMSE:", np.mean(rmse))
print("Average L2 Distance to Ideal:", np.mean(l2_dist))
```
Average true non-zero coefficients: 27.0

Average RMSE: 6.370267910386121

Average L2 Distance to Ideal: 3.173100262633921

#### Lasso 
```
model = Lasso()
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  lasso_params = [{'alpha': np.linspace(0.001, 2, 50)}]

  grid = GridSearchCV(model, lasso_params, cv = 5, scoring='neg_mean_absolute_error')

  grid.fit(X, y)
```

```
kf = KFold(n_splits=10, shuffle=True, random_state=2021)
coeffs = []
rmse = []
l2_dist = []
for train_index , test_index in kf.split(X):
    X = pd.DataFrame(X)
    X_train , X_test = X.iloc[train_index,:].values,X.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    model = Lasso(alpha = grid.best_params_.get('alpha'))
    model.fit(X_train, y_train)
    beta_hat = model.coef_
    pos_lasso = np.where(beta_hat != 0)
    coeffs.append(len(np.intersect1d(pos, pos_lasso)))
    yhat = model.predict(X_test)
    rmse.append(MSE(yhat, y_test)**.5)
    l2_dist.append(np.linalg.norm(model.coef_ - beta_star,ord = 2))
print("Average true non-zero coefficients:", np.mean(coeffs))
print("Average RMSE:", np.mean(rmse))
print("Average L2 Distance to Ideal:", np.mean(l2_dist))
```
Average true non-zero coefficients: 20.7

Average RMSE: 3.86589927535748

Average L2 Distance to Ideal: 2.411194331698895

#### ElasticNet
```
model = ElasticNet()
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  elastic_params = [{'alpha': np.linspace(0.001, 2, 25), 'l1_ratio': np.linspace(.01, 1.5, 25)}]

  grid = GridSearchCV(model, elastic_params, cv = 5, scoring='neg_mean_absolute_error')

  grid.fit(X, y)
```
    
```
kf = KFold(n_splits=10, shuffle=True, random_state=2021)
coeffs = []
rmse = []
l2_dist = []
for train_index , test_index in kf.split(X):
    X = pd.DataFrame(X)
    X_train , X_test = X.iloc[train_index,:].values,X.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    model = ElasticNet(alpha = grid.best_params_.get('alpha'), l1_ratio = grid.best_params_.get('l1_ratio'))
    model.fit(X_train, y_train)
    beta_hat = model.coef_
    pos_lasso = np.where(beta_hat != 0)
    coeffs.append(len(np.intersect1d(pos, pos_lasso)))
    yhat = model.predict(X_test)
    rmse.append(MSE(yhat, y_test)**.5)
    l2_dist.append(np.linalg.norm(model.coef_ - beta_star,ord = 2))
print("Average true non-zero coefficients:", np.mean(coeffs))
print("Average RMSE:", np.mean(rmse))
print("Average L2 Distance to Ideal:", np.mean(l2_dist))
```
Average true non-zero coefficients: 21.1

Average RMSE: 3.863633381774405

Average L2 Distance to Ideal: 2.0912826537192526

    

#### References 

Belloni, A., Chernozhukov, V., & Wang, L. (2011). Square-root lasso: pivotal recovery of sparse signals via conic programming. Biometrika, 98(4), 791–806. http://www.jstor.org/stable/23076172

Gupta, A. (2020, December 2). Feature selection techniques in machine learning. Analytics Vidhya. Retrieved April 1, 2022, from https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/ 

Jones, A. (2020, March 27). The smoothly clipped absolute deviation (SCAD) penalty. Andy Jones. Retrieved April 1, 2022, from https://andrewcharlesjones.github.io/journal/scad.html 

Shaikh, R. (2018, October 28). Feature selection techniques in machine learning with python. Medium. Retrieved April 1, 2022, from https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e 
