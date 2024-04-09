## Project Title: Modeling Neuronal Responses to Active and Passive Movements in the Visual Thalamus
## Our dataset contains three variables: speed_active, speed_passive, and speed_time.
## We select two models for analysis: Generalized Linear Models (GLM) and XGBoost (XGB) models.
## The scripts generate two scatter plots and nine heatmap figures.
## Author: Fangfei Lin
## Time:2024-04

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pyglmnet import GLM
import xgboost as xgb
import copy
import os
import pdb

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from operator import itemgetter
from scipy import io
import scipy.io as sio
import warnings


warnings.filterwarnings("ignore")


##################### Define the file paths for input files ################
"""
Dataset contains three variables: speed_active, speed_passive, speed_time.
"""

expt_path = r'/Users/tuotuofei/Desktop/fangfei_real_data_24_07_2024_expt.mat'
speed_time_path = r'/Users/tuotuofei/Desktop/fangfei_speed_time.mat'


#################### machine learning trianing #################
"""
Here, we select the GML and XGB models from parameters derived from the dataset3
XGB: Best Learning Rate: 0.1; Best Depth : 1; Best Num Round: 5011.
GLM: Best Learning Rate: 0.025; Best Reg Lambda: 0.0215.
"""

# Set the random seed for reproducibility
rs = 1

# Read the dataset from the specified experiment file path
mat_contents = io.loadmat(expt_path)
mat_contents_speed_time = sio.loadmat(speed_time_path)


# Reshape the dataset and display the reshaped matrix
"""
Combine all trials into a single 2D array where rows correspond to observations and columns to time points
"""

Y_matrix = mat_contents['fr'].reshape((323*78, 368))
Y_matrix 

# Filter the zero data
non_zeros_index = []
for idx in list(range(mat_contents['fr'].shape[2])):
    if sum(sum(mat_contents['fr'][:,:,idx])) != 0:
        non_zeros_index .append(idx)

Y_matrix = Y_matrix[:,non_zeros_index]


def predict_data(X,y,rs):
    
    #init misc
    N = np.size(y,0)
    index = np.array(range(0,N))
    yhat_glm = np.zeros(N)
    yhat_xgb = np.zeros(N)
    
    # init xgboost
    xgb_params = {'objective': "count:poisson", #for poisson output
                'eval_metric': "logloss", #loglikelihood loss
                'learning_rate': 0.1,
                'subsample': 1,
                'max_depth': 1,
                'gamma': 1,
                #'tree_method': 'gpu_hist'
                 }
    num_round = 5011
    
    
    #stratify variable
    str_var = y > np.median(y)
        
    # generate test and training set
    X1, X2, y1, y2, index1, index2 = train_test_split(X, y, index,
                                                    test_size=0.5,
                                                    random_state=rs,
                                                    shuffle=True,
                                                    stratify=None)
        
    #########set1#############
    
    # create an instance of the GLM class
    glm = GLM(distr='poisson', alpha=0.2,
              learning_rate=0.025,
              reg_lambda=0.0215,
              score_metric='pseudo_R2')

    # fit the model on the training data
    glm.fit(X1, y1)
    
    # predict using fitted glm model on the test data
    yhat_glm[index2] = glm.predict(X2)
    
    # train xgboost
    xgb1 = xgb.DMatrix(X1, label=y1)
    model = xgb.train(xgb_params, xgb1, num_round)
    
    # predict using fitted xgboost model on the test data
    xgb2 = xgb.DMatrix(X2)
    yhat_xgb[index2] = model.predict(xgb2)
    
    #pdb.set_trace()
    
    #########set2#############

    # create an instance of the GLM class
    glm = GLM(distr='poisson', alpha=0.2,
              learning_rate=0.025,
              reg_lambda=0.0215,
              score_metric='pseudo_R2')

    # fit the model on the training data
    glm.fit(X2, y2)
    
    # predict using fitted glm model on the test data
    yhat_glm[index1] = glm.predict(X1)
    
    # train xgboost
    xgb2 = xgb.DMatrix(X2, label=y2)
    model = xgb.train(xgb_params, xgb2, num_round)
    
    # predict using fitted xgboost model on the test data
    xgb1 = xgb.DMatrix(X1)
    yhat_xgb[index1] = model.predict(xgb1)
    
    # return results
    return yhat_glm, yhat_xgb


###################################### speed_active #############################################
# Prepare input data (each time a feature is modified)
X = np.array(mat_contents['speed_active'].flatten())


glm_r2_list = []
xgb_r2_list = []

for y_index in range(Y_matrix.shape[1]):

    y = Y_matrix[:,y_index]
    X = X.reshape((X.shape[0],1))
    
    #init Yhats 
    N = np.size(y,0)
    Yhat_glm = np.zeros((N)) # ,dimX[2]
    Yhat_xgb = np.zeros((N)) # ,dimX[2]


    Yhat_glm, Yhat_xgb = predict_data(X, y, rs)

    # Store prediction results in a dictionary. Includes input feature 'X', actual values 'y',
    # predictions from GLM 'Yhat_glm', predictions from XGB 'Yhat_xgb', and the random seed 'rs'
    
    results = {'X':X[:,0],'y':y,'Yhat_glm': Yhat_glm, 'Yhat_xgb': Yhat_xgb , 'rs': rs}  
    print(results)
    df_predict = pd.DataFrame({key: value for key, value in results.items() if key != 'rs'})
    df_predict['rs'] = results['rs']
    
    # save models prediction reults to a CSV file: GLM and XGB 
    glm_r2_list.append(r2_score(y, Yhat_glm))
    xgb_r2_list.append(r2_score(y, Yhat_xgb))


# Create a DataFrame to store the R^2 scores for both the GLM and XGB models

df_speed_active_R2 = pd.DataFrame(
    {'glm_r2': glm_r2_list, 'xgb_r2': xgb_r2_list},
    index = ['Neuron_'+ str(n) for n in range(Y_matrix.shape[1])])
    
################################################speed_active_speed_passive_speed_time ##################################################
# Prepare input data (each time a feature is modified)
X = np.array(list(zip(mat_contents['speed_passive'].flatten(), mat_contents['speed_active'].flatten(), mat_contents_speed_time['speed_time'].flatten())))


glm_r2_list = []
xgb_r2_list = []

for y_index in range(Y_matrix.shape[1]):

    y = Y_matrix[:,y_index]

    #init Yhats 
    N = np.size(y,0)
    Yhat_glm = np.zeros((N)) # ,dimX[2]
    Yhat_xgb = np.zeros((N)) # ,dimX[2]


    Yhat_glm, Yhat_xgb = predict_data(X, y, rs)

    # Store prediction results in a dictionary. Includes input feature 'X', actual values 'y',
    # predictions from GLM 'Yhat_glm', predictions from XGB 'Yhat_xgb', and the random seed 'rs'
    
    results = {'X':X[:,0],'y':y,'Yhat_glm': Yhat_glm, 'Yhat_xgb': Yhat_xgb , 'rs': rs}  
    print(results)
    df_predict = pd.DataFrame({key: value for key, value in results.items() if key != 'rs'})
    df_predict['rs'] = results['rs']



######################## save R^2 and models prediction results in CSV files##################

# save models prediction reults to a CSV file: GLM and XGB 

    df_predict.to_csv(f'/Users/tuotuofei/Desktop/result/Neuron_{str(y_index)}_speed_active_speed_passive_speed_time_models prediction results.csv')

    glm_r2_list.append(r2_score(y, Yhat_glm))
    xgb_r2_list.append(r2_score(y, Yhat_xgb))


# Create a DataFrame to store the R^2 scores for both the GLM and XGB models

df_R2 = pd.DataFrame(
    {'glm_r2': glm_r2_list, 'xgb_r2': xgb_r2_list},
    index = ['Neuron_'+ str(n) for n in range(Y_matrix.shape[1])])

# Save the DataFrame containing the R^2 scores for both GLM and XGB models to a CSV file. 'rs' is ramdom seed
df_R2.to_csv('/Users/tuotuofei/Desktop/result/speed_active, speed_passive and speed_time_R2.csv')


############################## save the figures ################################

"""
Here,
we will generate two scatter plots:
1. Comparing the performance of GLM and XGB models in terms of their R^2 values.
2. Comparing the performance of different variables in XGB models with their R^2 values.

Next, we will generate three heatmap figures, including:
1. The neuron with the highest R^2 value from the real data.
2. The neuron with the highest R^2 value from XGB prediction results without Poisson distribution.
3. The neuron with the highest R^2 value from XGB prediction results with Poisson distribution.
"""


############## Compare 2 modle performance of GLM and XGB with their R2 ################
"""
Compare the performance of GLM and XGB models in terms of their R^2 values, representing the models' ability to explain the variance of the response variable. 
Then, it visualizes this comparison using scatter plots for each model type, highlighting the top-performing neurons.
The goal is to identify which model performs better on average and to pinpoint the specific neurons that are best explained by each model type.
"""


# Sort the DataFrame based on 'glm_r2' and 'xgb_r2' in descending order
df_sorted_glm = df_R2.sort_values(by='glm_r2', ascending=False)
df_sorted_xgb = df_R2.sort_values(by='xgb_r2', ascending=False)

# Create scatter plots
fig, ax = plt.subplots(1, 1, figsize=(12, 8),dpi=300)

# GLM R2 Scatter Plot
ax.scatter(
    x = range(len(df_sorted_glm)), 
    y = df_sorted_glm['glm_r2'], label='GLM_$R^2$')
for i, txt in enumerate(df_sorted_glm.index[:5]):
    ax.annotate(txt, (i, df_sorted_glm['glm_r2'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

ax.set_xlabel('Neuron Index', fontsize=14, labelpad=10)
ax.set_ylabel('Coefficient of Determination ($R^2$)', fontsize=14, labelpad=10)
ax.set_ylim(0, 0.3)

# XGB R2 Scatter Plot
ax.scatter(
    x = range(len(df_sorted_xgb)), 
    y = df_sorted_xgb['xgb_r2'], color='orange', label='XGB_$R^2$')
for i, txt in enumerate(df_sorted_xgb.index[:5]):
    ax.annotate(txt, (i, df_sorted_xgb['xgb_r2'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')


plt.legend(title='Model', title_fontsize='large', fontsize='large')
plt.tight_layout()
plt.savefig('/Users/tuotuofei/Desktop/result/xgb_glm_R2_scatter_plot.png',dpi=300)

    


###################### Compare the performance of different variables in XGB models with their R2 ##################
"""
Compare analysis of XGB model predictions using different variable sets on a dataset of neurons. 
The goal is to evaluate how the inclusion of various features (active speed, passive speed, and time) affects the predictive performance of the models by the R2.

1. models considering only "speed_active" versus models incorporating "speed_active, speed_passive, and speed_time"
2. Visualizing the comparison with a scatter plot, highlighting the difference in R^2 values for each model type. 
3. Plotting a reference line (y = x) to identify improvements or declines in model performance with the addition of speed_passive, and speed_time as features.
"""

df_speed_active_passive_time_R2 = df_R2
df_merge_R2 = pd.merge(df_speed_active_R2, df_speed_active_passive_time_R2, left_index=True,right_index=True)

plt.figure(dpi=300)
plt.axis('equal')
sns.scatterplot(
    x = df_merge_R2['xgb_r2_x'], # active_R2
    y= df_merge_R2['xgb_r2_y'] # active_passive_time_R2
)

# Define the reference line for "x = y"
x_ref_line = [0, 0.35]  
y_ref_line = [0, 0.35] 

plt.plot(x_ref_line, y_ref_line, label="y = x", color='gray', linestyle='--')

plt.text(0.250609, 0.272641, 'Neuron_25',fontsize=6)
plt.text(0.009490, 0.064496, 'Neuron_44',fontsize=6)
plt.text(0.028991, 0.080296, 'Neuron_49',fontsize=6)

plt.xlabel('Active Variable $R^2$', fontsize=12)
plt.ylabel('Combined Variables $R^2$', fontsize=12)
         
plt.ylim(0, 0.35)
plt.xlim(0, 0.35)
plt.tight_layout()
plt.grid(True)
plt.legend(loc='upper left', frameon=True)

plt.savefig('/Users/tuotuofei/Desktop/result/active_vs_speed_active_passive_time_R2.png',dpi=300)



########################## a heatmap figure: the best Neuron with real data ###############################
"""
First, determining the neuron with the highest R2 value from XGBoost models. 
Next, visualize this neuron's data through a heatmap based on its maximum R2 value.
The best one is Neuron_25
"""


# According to the R2, determining the neuron with the highest R2 value from XGBoost models
best_neuron_idx = df_R2['xgb_r2'].idxmax()
best_r2_score = df_R2['xgb_r2'].max().round(3)

print(f"Best Neuron Index: {best_neuron_idx}, $R^2$ Score: {best_r2_score}")

# Plotting the heatmap for the best neuron
Y_best = Y_matrix[:,int(best_neuron_idx.split('_')[1])]
plt.figure(figsize=(8,4),dpi=300)
sns.heatmap(Y_best.reshape(323, 78), vmin=0, vmax=10)
plt.xticks(ticks=[0, 15, 30, 45, 60, 75], labels=['0', '15', '30', '45', '60','75'])
plt.yticks(ticks=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 323], labels=['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '323'])
plt.ylabel('Trial Number', fontsize=12)
plt.xlabel('Time (s)', fontsize=12)
plt.title('SPKCOUNT for Neuron_25', fontsize=14)
plt.savefig('/Users/tuotuofei/Desktop/result/heatmap_the_best_neuron_with_real_data.png',dpi=300)




########################## two heatmap figures: Comparing model predictions with and without Poisson distribution of XGB in the heatmap generation ######################

# Load the prediction results for the best Neuron according to the maximum R2 value from a CSV file.
df_predict_Neuron_25 = pd.read_csv(f'../result/{best_neuron_idx}_speed_active_speed_passive_speed_time_models prediction results.csv')
df_predict_Neuron_25

# Generate a heatmap figure of model predictions without Poisson distribution.
plt.figure(figsize=(8,4),dpi=300)
sns.heatmap(np.array(df_predict_Neuron_25['Yhat_xgb']).reshape(323, 78), vmin=0, vmax=10)
plt.xticks(ticks=[0, 15, 30, 45, 60, 75], labels=['0', '15', '30', '45', '60','75'])
plt.yticks(ticks=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 323], labels=['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '323'])
plt.ylabel('Trial Number', fontsize=12)
plt.xlabel('Time (s)', fontsize=12)
plt.title('XGB Predictions Heatmap for Neuron 25 (Non-Poisson)', fontsize=14)
plt.savefig('/Users/tuotuofei/Desktop/result/heatmap_XGB_the best_Neuron_without_Poisson distribution.png',dpi=300)

# Generate a heatmap figure of model predictions with Poisson distribution.
lambda_t = np.array(df_predict_Neuron_25['Yhat_xgb'])
poisson_data_variable = [np.random.poisson(rate) for rate in lambda_t]
reshaped_data = np.array(poisson_data_variable).reshape(323, 78) 

plt.figure(figsize=(8, 4), dpi=300)
sns.heatmap(reshaped_data, vmin=0, vmax=10)
plt.xticks(ticks=[0, 15, 30, 45, 60, 75], labels=['0', '15', '30', '45', '60','75'])
plt.yticks(ticks=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 323], labels=['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '323'])
plt.ylabel('Trial Number', fontsize=12)
plt.xlabel('Time (s)', fontsize=12)
plt.title('XGB Predictions Heatmap for Neuron 25 (Poisson)', fontsize=14)
plt.savefig('/Users/tuotuofei/Desktop/result/heatmap_XGB_the best_Neuron_with_Poisson distribution.png',dpi=300)


######### Neuron_44 ##########
"""
Generate three heatmaps: real data, XGB prediction results without Poisson distribution, and XGB prediction results with Poisson distribution.
"""


# heatmap of Neuron_44 with the real data
Y_Neuron_44= Y_matrix[:,44]
plt.figure(figsize=(8,4),dpi=300)
sns.heatmap(Y_Neuron_44.reshape(323, 78), vmin=0, vmax=2)
plt.xticks(ticks=[0, 15, 30, 45, 60, 75], labels=['0', '15', '30', '45', '60','75'])
plt.yticks(ticks=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 323], labels=['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '323'])
plt.ylabel('Trial Number', fontsize=12)
plt.xlabel('Time (s)', fontsize=12)
plt.title('SPKCOUNT for Neuron_44', fontsize=14)
plt.savefig('/Users/tuotuofei/Desktop/result/heatmap_the_neuron_44_with_real_data.png',dpi=300)

# heatmap of Neuron_44_without_Poisson distribution
# Load the prediction results for the Neuron_44 according to the maximum R2 value from a CSV file.
df_predict_Neuron_44 = pd.read_csv(r'/Users/tuotuofei/Desktop/result/Neuron_44_speed_active_speed_passive_speed_time_models prediction results.csv')


# Generate a heatmap figure of model predictions without Poisson distribution.
plt.figure(figsize=(8,4),dpi=300)
sns.heatmap(np.array(df_predict_Neuron_44['Yhat_xgb']).reshape(323, 78), vmin=0, vmax=2)
plt.xticks(ticks=[0, 15, 30, 45, 60, 75], labels=['0', '15', '30', '45', '60','75'])
plt.yticks(ticks=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 323], labels=['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '323'])
plt.ylabel('Trial Number', fontsize=12)
plt.xlabel('Time (s)', fontsize=12)
plt.title('XGB Predictions Heatmap for Neuron 44 (Non-Poisson)', fontsize=14)
plt.savefig('/Users/tuotuofei/Desktop/result/heatmap_XGB_the Neuron_44_without_Poisson distribution.png',dpi=300)

# Generate a heatmap figure of model predictions with Poisson distribution.
lambda_t = np.array(df_predict_Neuron_44['Yhat_xgb'])
poisson_data_variable = [np.random.poisson(rate) for rate in lambda_t]
reshaped_data = np.array(poisson_data_variable).reshape(323, 78) 

plt.figure(figsize=(8, 4), dpi=300)
sns.heatmap(reshaped_data, vmin=0, vmax=2)
plt.xticks(ticks=[0, 15, 30, 45, 60, 75], labels=['0', '15', '30', '45', '60','75'])
plt.yticks(ticks=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 323], labels=['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '323'])
plt.ylabel('Trial Number', fontsize=12)
plt.xlabel('Time (s)', fontsize=12)
plt.title('XGB Predictions Heatmap for Neuron 44 (Poisson)', fontsize=14)
plt.savefig('/Users/tuotuofei/Desktop/result/heatmap_XGB_the Neuron_44_with_Poisson distribution.png',dpi=300)




######### Neuron_49 ##########
"""
Generate three heatmaps: real data, XGB prediction results without Poisson distribution, and XGB prediction results with Poisson distribution.
"""

# heatmap of Neuron_49 with the real data
Y_Neuron_49= Y_matrix[:,49]
plt.figure(figsize=(8,4),dpi=300)
sns.heatmap(Y_Neuron_49.reshape(323, 78), vmin=0, vmax=10)
plt.xticks(ticks=[0, 15, 30, 45, 60, 75], labels=['0', '15', '30', '45', '60','75'])
plt.yticks(ticks=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 323], labels=['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '323'])
plt.ylabel('Trial Number', fontsize=12)
plt.xlabel('Time (s)', fontsize=12)
plt.title('SPKCOUNT for Neuron_49', fontsize=14)
plt.savefig('/Users/tuotuofei/Desktop/result/heatmap_the_neuron_49_with_real_data.png',dpi=300)

# heatmap of Neuron_49_without_Poisson distribution
# Load the prediction results for the Neuron_49 according to the maximum R2 value from a CSV file.
df_predict_Neuron_49 = pd.read_csv(r'/Users/tuotuofei/Desktop/result/Neuron_49_speed_active_speed_passive_speed_time_models prediction results.csv')


# Generate a heatmap figure of model predictions without Poisson distribution.
plt.figure(figsize=(8,4),dpi=300)
sns.heatmap(np.array(df_predict_Neuron_49['Yhat_xgb']).reshape(323, 78), vmin=0, vmax=10)
plt.xticks(ticks=[0, 15, 30, 45, 60, 75], labels=['0', '15', '30', '45', '60','75'])
plt.yticks(ticks=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 323], labels=['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '323'])
plt.ylabel('Trial Number', fontsize=12)
plt.xlabel('Time (s)', fontsize=12)
plt.title('XGB Predictions Heatmap for Neuron 49 (Non-Poisson)', fontsize=14)
plt.savefig('/Users/tuotuofei/Desktop/result/heatmap_XGB_the Neuron_49_without_Poisson distribution.png',dpi=300)

# Generate a heatmap figure of model predictions with Poisson distribution.
lambda_t = np.array(df_predict_Neuron_49['Yhat_xgb'])
poisson_data_variable = [np.random.poisson(rate) for rate in lambda_t]
reshaped_data = np.array(poisson_data_variable).reshape(323, 78) 

plt.figure(figsize=(8, 4), dpi=300)
sns.heatmap(reshaped_data, vmin=0, vmax=10)
plt.xticks(ticks=[0, 15, 30, 45, 60, 75], labels=['0', '15', '30', '45', '60','75'])
plt.yticks(ticks=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 323], labels=['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '323'])
plt.ylabel('Trial Number', fontsize=12)
plt.xlabel('Time (s)', fontsize=12)
plt.title('XGB Predictions Heatmap for Neuron 49 (Poisson)', fontsize=14)
plt.savefig('/Users/tuotuofei/Desktop/result/heatmap_XGB_the Neuron_49_with_Poisson distribution.png',dpi=300)
