import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, shapiro
from sklearn.model_selection import GridSearchCV
import gc
import itertools
from sklearn.utils import resample
import ast
import json
import re
import statsmodels.api as sm
from scipy.stats import entropy

from constants import *

def data_prep(df):

    # Convert columns with object dtype to category dtype
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype('category')

    # Add a unique identifier for each record
    df['id'] = np.arange(len(df))   
   
    return df

def encode_categorical_from_file(df):
    
    # Apply one-hot encoding to categorical columns
    df_encoded = pd.get_dummies(df, drop_first=True)  # drop_first=True avoids dummy variable trap
    
    return df_encoded

def track_errors(X_test, y_test, y_pred):
    # Create new columns for Type I and Type II errors and initialize them to 0
    X_test['TypeI_Error'] = 0  # False Positive: model predicted 1, actual is 0
    X_test['TypeII_Error'] = 0  # False Negative: model predicted 0, actual is 1
    
    # Only update the rows corresponding to the test set
    # Type I error: model predicted 1 but true value is 0
    X_test.loc[X_test.index, 'TypeI_Error'] = ((y_pred == 1) & (y_test == 0)).astype(int)
    
    # Type II error: model predicted 0 but true value is 1
    X_test.loc[X_test.index, 'TypeII_Error'] = ((y_pred == 0) & (y_test == 1)).astype(int)
    
    return X_test



def calculate_denominator(df, NQIs, CQIs):

    # For NQIs, use NumPy for vectorized calculation of squared deviations from the mean
    denominator_nqi = np.sum((df[NQIs].values - df[NQIs].mean(axis=0).values) ** 2)
    
    # For CQIs, use NumPy to calculate how many different values exist compared to the mode
    denominator_cqi = sum(np.sum(df[CQIs].ne(df[CQIs].mode().iloc[0])))
    
    # Total denominator is the sum of the variances for NQIs and deviations for CQIs
    denominator = denominator_nqi + denominator_cqi
    return denominator



def get_cqi_levels(df, CQIs):

    levels_dict = {}
    for cqi in CQIs:
        if cqi in df.columns:
            # Map each unique category in the CQI column to a unique index
            levels_dict[cqi] = {category: index for index, category in enumerate(df[cqi].unique())}
    
    return levels_dict




def get_nqi_bounds(df, NQIs):

    bounds = {}
    
    for nqi in NQIs:
        if nqi in df.columns:
            lower_bound = df[nqi].min()
            upper_bound = df[nqi].max()
            bounds[nqi] = {'lower_bound': lower_bound, 'upper_bound': upper_bound}
    
    return bounds



def calculate_information_loss(original_df,anonymized_df, NQIs, CQIs):

    # Sort original data and anonymized data by unique identifier (id)
    original_df = original_df.sort_values('id')
    anonymized_df = anonymized_df.sort_values('id')
    
    # Compute squared differences for numerical quasi-identifiers
    num_loss = np.sum((original_df[NQIs].values - anonymized_df[NQIs].values) ** 2)

    # Compute categorical mismatch (0 if same, 1 if different)
    cat_loss = np.sum(original_df[CQIs].values != anonymized_df[CQIs].values)

    # Total information loss per record
    total_loss = num_loss + cat_loss

    # Calculate denominator for normalization
    denominator = calculate_denominator(original_df, NQIs, CQIs)

    infoloss=total_loss/denominator

    return infoloss

def calculate_k_constraint(anonymized_df, k, n_cluster):

    # Count the number of records per cluster
    num_records_per_cluster = anonymized_df['cluster'].value_counts()

    # Identify clusters that violate the k constraint
    violating_clusters = num_records_per_cluster[num_records_per_cluster < k]

    # Calculate the total number of k-violations (sum the deficits)
    total_k_violation = np.sum(k - violating_clusters)

    return {
        "k violation": total_k_violation,
        "violating clusters": violating_clusters
    }


def calculate_entropy_of_SA(anonymized_data, SA_columns):
    # Step 1: Extract the relevant data (assuming SA_columns are the columns to compute entropy on)
    SA_data = anonymized_data[SA_columns + ['cluster']]
    
    # Step 2: Separate the data by cluster
    SA_by_cluster = SA_data.groupby('cluster')
    
    # Step 3: Create a function to calculate entropy for each column
    def calculate_column_entropy(cluster_data, sa_col):
        value_counts = cluster_data[sa_col].value_counts(normalize=True)
        return entropy(value_counts, base=2)

    # Step 4: Apply the entropy calculation for each SA column and cluster
    entropy_matrix = SA_by_cluster[SA_columns].apply(lambda cluster: cluster.apply(lambda col: calculate_column_entropy(cluster, col.name)))
    
    # Step 5: Compute the minimum entropy for each SA across clusters
    entropy_SA = np.sum(entropy_matrix, axis=1).min()
    
    return -entropy_SA


def normalize_data(value, min_value, max_value):

    return (value - min_value) / (max_value - min_value + 1e-6)



def convert_results_df(results_df):
    results = []

    for row_idx in range(results_df.shape[0]):  # Iterate over rows
        row_results = []
        
        for col_idx in range(results_df.shape[1]):  # Iterate over columns
            cell = results_df.iloc[row_idx, col_idx]
            
            # Step 1: Split and clean the data
            mappings = cell.replace('{', '').replace('}', '').split(',')

            # Step 2: Fix split confusion matrix values
            fixed_mappings = []
            inside_conf_matrix = False
            conf_matrix_parts = []

            for item in mappings:
                item = item.strip()
                
                if "Confusion matrix" in item:  
                    # Start collecting confusion matrix values
                    inside_conf_matrix = True
                    conf_matrix_parts.append(item)
                elif inside_conf_matrix:
                    # Collect the remaining parts of the confusion matrix
                    conf_matrix_parts.append(item)
                    if "]]" in item:  # End of confusion matrix
                        inside_conf_matrix = False
                        fixed_mappings.append(" ".join(conf_matrix_parts))  # Join into a single entry
                        conf_matrix_parts = []
                else:
                    fixed_mappings.append(item)  # Regular key-value pairs

            # Step 3: Convert fixed mappings into a dictionary
            parsed_dict = {}
            for pair in fixed_mappings:
                key, value = pair.split(":", 1)
                key = key.strip().strip("'")  # Remove any extra quotes/spaces
                value = value.strip()

                # Convert numerical values
                if value.startswith("np.float64("):
                    value = float(value.replace("np.float64(", "").replace(")", ""))
                elif "array([" in value:
                    # Extract confusion matrix values
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", value)
                    numbers = list(map(float, numbers))
                    size = int(len(numbers) / 2)  # Assuming a 2x2 matrix
                    value = np.array(numbers).reshape(size, 2)
                
                parsed_dict[key] = value

            row_results.append(parsed_dict)
        
        results.append(row_results)

    return results


def plot_metric_trend_with_mean(results_df, metric_column, y_label, smooth_method='moving_avg', window_size=5, y_range=None):
    results = convert_results_df(results_df)  # Convert results

    plt.figure(figsize=(50, 20))  # Adjust figure size

    all_res = []  # Store all particle values per iteration

    for j in range(len(results[0])):  # Iterate over particles
        res = [results[i][j][metric_column] for i in range(len(results))]  # Extract metric values
        all_res.append(res)
        plt.plot(res, alpha=0.6, linewidth=0.5)  # Plot all particles

    # Convert to NumPy array for efficient calculations
    all_res = np.array(all_res)
    mean_trend = np.mean(all_res, axis=0)
    std_dev = np.std(all_res, axis=0)  # Compute standard deviation for shading

    # **Trend Smoothing**
    if smooth_method == 'moving_avg':
        smooth_trend = pd.Series(mean_trend).rolling(window=window_size, center=True).mean()
    elif smooth_method == 'lowess':
        smooth_trend = sm.nonparametric.lowess(mean_trend, np.arange(len(mean_trend)), frac=0.1)[:, 1]
    else:
        smooth_trend = mean_trend  # Default to mean if no smoothing is applied

    # **Plot Mean with Smoothed Trend**
    plt.plot(smooth_trend, color='red', linewidth=10, linestyle='dashed', label="Smoothed Trend")

    # **Shaded Region for Variability**
    #plt.fill_between(np.arange(len(mean_trend)), mean_trend - std_dev, mean_trend + std_dev, color='red', alpha=0.2)

    # **Apply Y-axis range if provided**
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])

    plt.ylabel(y_label, fontsize=50)  
    plt.xlabel('Iteration', fontsize=50)  
    plt.tick_params(axis='y', labelsize=50)
    plt.tick_params(axis='x', labelsize=50)
    plt.legend(fontsize=50, loc='upper right')

    plt.show()

def plot_global_best_trend(results_df, metric_column, y_label):
    results = convert_results_df(results_df)  # Convert results

    plt.figure(figsize=(50, 20))  # Adjust figure size

    best_res = [min([results[i][j][metric_column] for j in range(len(results[i]))]) for i in range(len(results))]  # Extract best metric values
    
    # smooth_trend = pd.Series(best_res).rolling(window=5, center=True).mean()

    plt.plot(best_res, linewidth=10)  # Plot all particles
    # plt.plot(smooth_trend, color='red', linewidth=10, linestyle='dashed', label="Smoothed Trend")

    plt.ylabel(y_label, fontsize=50)  
    plt.xlabel('Iteration', fontsize=50)  
    plt.tick_params(axis='y', labelsize=50)
    plt.tick_params(axis='x', labelsize=50)

    plt.show()

def plot_global_best_so_far(results_df, metric_column, y_label):
    results = convert_results_df(results_df)  # Convert results

    plt.figure(figsize=(50, 20))  # Adjust figure size

    best_res = [min([results[i][j][metric_column] for j in range(len(results[i]))]) for i in range(len(results))]  # Extract best metric values

    # Compute the global best found so far
    global_best_so_far = [best_res[0]]  # Initialize with the first value
    for i in range(1, len(best_res)):
        global_best_so_far.append(min(global_best_so_far[-1], best_res[i]))  # Update with minimum so far

    plt.plot(global_best_so_far, linewidth=10)  # Plot the global best trend

    plt.ylabel(y_label, fontsize=50)  
    plt.xlabel('Iteration', fontsize=50)  
    plt.tick_params(axis='y', labelsize=50)
    plt.tick_params(axis='x', labelsize=50)

    plt.show()