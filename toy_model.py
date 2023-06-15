import pandas as pd
import numpy as np
from numpy.random import random_sample
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


data_x = np.linspace(10,20,100)
data = pd.DataFrame({'x': data_x,#'2x': 2*data_x, 'x2': data_x**2, 'exp(x)': np.exp(data_x), 'sin(x)': np.sin(data_x),
                     #'rand+x': data_x+random_sample(len(data_x)), 'rand+x_1': data_x+random_sample(len(data_x)),
                     'rand+x_2': data_x+random_sample(len(data_x)),'rand+x_3': data_x+random_sample(len(data_x)),
                     'rand*x': data_x*random_sample(len(data_x)),'rand': random_sample(len(data_x))})
gauss = GaussianProcessRegressor()
pipe = Pipeline([('Scale',StandardScaler()),('gauss',GaussianProcessRegressor())])

OBJECTIVE = 'rand+x_3'

cols = data.columns.drop(OBJECTIVE)
for i in range(10):
    mod_cols = []
    fig, ax = plt.subplots(len(cols),1)
    for i, col in enumerate(cols):

        # Split the data into features and target variables
        mod_cols.append(col)
        x = data[mod_cols]#.drop('2x',axis=1)
        y = data[OBJECTIVE]

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        gauss.fit(x_train,y_train)
        pipe.fit(x_train, y_train)

        # Make predictions on the test set
        y_pred = gauss.predict(x_test)
        y_pred_s = pipe.predict(x_test)

        # Calculate mean squared error and R-squared on the test set
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_s = mean_squared_error(y_test, y_pred_s)
        r2_s = r2_score(y_test, y_pred_s)

        print(f'For data with columns {mod_cols} and {len(data["x"])} data-------------------')
        print(f'Mean Squared Error: {mse:.4f}\n MSE Scaled: {mse_s:.4f}')
        print(f'R-squared: {r2:.4f}\n R^2 Scaled: {r2_s:.4f}')
        
        #Plotting        
        ax[i].set_title(f'Predictions and values for {OBJECTIVE} w/ features {mod_cols} and {len(y_test)} tests', wrap=True)
        ax[i].scatter(x_test['x'], y_test, marker='+', label='Real values')
        ax[i].scatter(x_test['x'], y_pred, marker='.', label='Predictions')
        ax[i].scatter(x_test['x'], y_pred_s, marker='.', label='Predictions scaled')
        ax[i].legend()
        
    plt.show()
    eliminate = list({np.random.randint(0,len(data['x'])) for _ in range(len(data['x'])//5)})

    data.drop(eliminate, inplace=True)
    data.reset_index(drop=True, inplace=True)

