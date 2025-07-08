# imports
import numpy as np
import matplotlib.pyplot as plt
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
import seaborn as sns

# 3d figures
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# creating animations
import matplotlib.animation
from IPython.display import HTML

# styling additions
from IPython.display import HTML
style = '''
    <style>
        div.info{
            padding: 15px;
            border: 1px solid transparent;
            border-left: 5px solid #dfb5b4;
            border-color: transparent;
            margin-bottom: 10px;
            border-radius: 4px;
            background-color: #fcf8e3;
            border-color: #faebcc;
        }
        hr{
            border: 1px solid;
            border-radius: 5px;
        }
    </style>'''
HTML(style)

"""# Logistic Regression

## Load Dataset
"""

from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
iris_df = iris.data
iris_df

iris_df['class'] = iris.target
label_name_dict = {val:key for key,val in zip(iris.target_names,range(3))}
iris_df['species'] = iris_df['class'].map(label_name_dict)

iris_df = iris_df[iris_df['class'].isin([0,1])]
iris_df

"""## 1D version"""

sns.scatterplot(data=iris_df,x='sepal length (cm)',y='class',hue='species');

X = iris_df.iloc[:,0].values.reshape(-1,1)
y = iris_df['class'].values

setosa_X = iris_df.loc[iris_df["class"]==0]
setosa_X = setosa_X.iloc[:,0].values.reshape(-1,1)
versicolor_X = iris_df.loc[iris_df["class"]==1]
versicolor_X = versicolor_X.iloc[:,0].values.reshape(-1,1)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty='none')
log_reg.fit(X,y)

log_reg.coef_,log_reg.intercept_

x_pl = np.linspace(iris_df.iloc[:,0].min(),iris_df.iloc[:,0].max())
import math
def sig_curve(x):
    curve = []
    for i in range(len(x)):
      curve.append(1/(1 + math.exp(-(log_reg.intercept_ + (log_reg.coef_ * x[i])))))
    return curve

plt.plot(x_pl,sig_curve(x_pl));

plt.scatter(setosa_X, np.zeros_like(setosa_X))
plt.scatter(versicolor_X, np.ones_like(versicolor_X))
plt.plot(x_pl, sig_curve(x_pl));

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))

for ax in [ax1,ax2]:
    ax.plot(x_pl,sig_curve(x_pl), alpha=0.3);
    ax.plot(x_pl,0.5*np.ones_like(x_pl),'k--')
    ax.fill_between(x_pl,0.5,1,color=colors[1],alpha=0.2)
    ax.fill_between(x_pl,0.5,color=colors[0],alpha=0.2)

ax1.scatter(setosa_X,sig_curve(setosa_X));
ax2.scatter(versicolor_X,sig_curve(versicolor_X),color=colors[1]);

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))

for ax in [ax1,ax2]:
    ax.plot(x_pl,sig_curve(x_pl), alpha=0.3);
    ax.plot(x_pl,0.5*np.ones_like(x_pl),'k--')
    ax.fill_between(x_pl,0.5,1,color=colors[1],alpha=0.2)
    ax.fill_between(x_pl,0.5,color=colors[0],alpha=0.2)

ax1.scatter(setosa_X,sig_curve(setosa_X));
errors = log_reg.predict(setosa_X) != np.zeros_like(setosa_X).ravel()
ax1.scatter(setosa_X[errors],sig_curve(setosa_X[errors]),marker='X',s=50,c='r');

ax2.scatter(versicolor_X,sig_curve(versicolor_X),color=colors[1]);
errors = log_reg.predict(versicolor_X) != np.ones_like(versicolor_X).ravel()
ax2.scatter(versicolor_X[errors],sig_curve(versicolor_X[errors]),marker='X',s=50,c='r');

# make predictions using the model and get the errors
errors = log_reg.predict(X) != y

# plot them!
plt.plot(x_pl,sig_curve(x_pl));
plt.scatter(setosa_X,np.zeros_like(setosa_X));
plt.scatter(versicolor_X,np.ones_like(versicolor_X));
plt.scatter(X[errors],y[errors],marker='X',s=50,c='r');

"""## 2D version"""

X = iris_df.iloc[:,:2].values
y = iris_df['class'].values

X.shape, y.shape

setosa_X = iris_df.loc[iris_df["class"]==0]
setosa_X = setosa_X.iloc[:,:2].values
versicolor_X = iris_df.loc[iris_df["class"]==1]
versicolor_X = versicolor_X.iloc[:,:2].values

# uncomment and run this cell to install plotly
# !pip install plotly
import plotly.express as px

sns.scatterplot(data=iris_df,x='sepal length (cm)',y='sepal width (cm)',hue='species');

fig = px.scatter_3d(iris_df,
                    x='sepal length (cm)',
                    y='sepal width (cm)',
                    z='class',
                    color='species')
fig.update_traces(marker={'size': 4})

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty='none')
log_reg.fit(X,y)

log_reg.coef_, log_reg.intercept_

def sig_curve(data):
  e_array = math.e*(np.ones_like(len(data)))
  return 1/(1+np.power(e_array, -(log_reg.intercept_ + (log_reg.coef_[0][0])*data[0:len(data), 0]+(log_reg.coef_[0][1]*data[0:len(data), 1]))))

# setup plot
x_pl = np.linspace(4.3,7,100)
y_pl = np.linspace(2,4.5,100)
X_pl,Y_pl = np.meshgrid(x_pl,y_pl)
XY_pl = np.vstack((X_pl.ravel(),Y_pl.ravel())).T
XY_pl.shape
Z = sig_curve(XY_pl).reshape(100,100)

# draw plot
import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(z=Z,
                                 x=x_pl,
                                 y=y_pl,
                                 opacity=0.2,
                                 colorscale='Turbo',
                                 showscale=False),
                      go.Scatter3d(z=sig_curve(setosa_X),
                                   x=setosa_X[:,0],
                                   y=setosa_X[:,1],
                                   mode='markers',
                                   marker=dict(
                                       size=4,
                                       color=colors[0],
                                       opacity=0.8),
                                   name='setosa'
                                  ),
                      go.Scatter3d(z=sig_curve(versicolor_X),
                                   x=versicolor_X[:,0],
                                   y=versicolor_X[:,1],
                                   mode='markers',
                                   marker=dict(
                                       size=4,
                                       color=colors[1],
                                       opacity=0.8),
                                   name='versicolor'
                      )])
fig.update_coloraxes(showscale=False)
fig.update_layout(autosize=False,
                  width=500,
                  height=500,
                  scene = dict(
                    xaxis_title='sepal length (cm)',
                    yaxis_title='sepal width (cm)'),
                    margin=dict(l=0, r=0, b=0, t=0)
                 )
