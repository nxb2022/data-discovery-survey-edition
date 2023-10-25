#!/usr/bin/env python

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime as dt

def transform_data(df):
    """
    Take the DataFrame and apply some transformations for further implementations

    Args:
    df: DataFrame.

    Returns:
    refined/transformed DataFrame
    """

    # Renaming columns
    df.columns.values[0] = "timestamp"
    df.columns.values[1] = "shoe_size"
    df.columns.values[2] = "study_program"
    df.columns.values[3] = "height"
    df.columns.values[4] = "course_reason"
    df.columns.values[5] = "letters_seattle"

    # Generate a date column based on timestamp
    df['date'] = pd.to_datetime(df['timestamp']).dt.date

    # Fix wrong shoe_size value - 2 users used the US metric instead of European
    df.loc[(df['shoe_size'] == 6.0), 'shoe_size'] = 37.0
    df.loc[(df['shoe_size'] == 7.5), 'shoe_size'] = 39.0

    # Fix wrong height value - Some users used meters instead of inches
    df.loc[(df['height'] > 100), 'height'] = df['height'] / 2.54

    # One user entered a completely wrong number as height, probably missing some digits
    df.loc[(df['height'] == 5.2), 'height'] = df["height"].mean()

    return df

def clean_data(df):
    """
    Cleaning the DataFrame regarding the K-means implementation

    Args:
    df: DataFrame.

    Returns:
    cleaned DataFrame
    """

    # Remove NaN from letters_seattle
    df = df.dropna()

    # Remove data from old survey
    df = df[(df['date'] >= dt.strptime('2023-07-09', '%Y-%m-%d').date())]

    # Keep only relevant columns, in this case, only numerical ones
    df = df.drop(columns=['timestamp', 'date', 'study_program', 'course_reason', 'letters_seattle'])

    return df

def initializeMeans(df):
    """
    Intialize the means by randomly taking two points from the dataframe

    Args:
    df: DataFrame.

    Returns:
    two pandas dataframes with mean1 and mean2
    """
    rand = random.sample(range(len(df)), 2)

    return df.iloc[rand[0]], df.iloc[rand[1]]

def euclideanDist(df, pointIDX, mean1, mean2):
    """
    Take the index of the point in the dataframe you want to calculate the
    distance from and calculate the euclidean distance to both means.

    Args:
    df: DataFrame.
    pointIDX: Index of the point the DataFrame
    mean1: DataFrame with a single row
    mean2: DataFrame with a single row

    Returns:
    a pandas dataframe with the closest mean assigned to column 'class' given the pointIDX
    """
    r_1 = (df.loc[pointIDX]['shoe_size'] - mean1['shoe_size'])**2 + (df.loc[pointIDX]['height'] - mean1['height'])**2
    r_2 = (df.loc[pointIDX]['shoe_size'] - mean2['shoe_size'])**2 + (df.loc[pointIDX]['height'] - mean2['height'])**2

    distance_1 = np.sqrt(r_1)
    distance_2 = np.sqrt(r_2)

    if distance_1 < distance_2:
        df.at[pointIDX, 'class'] = int(mean1.name)
    else:
        df.at[pointIDX, 'class'] = int(mean2.name)

    return df

def updateMean(df):
    """
    Update the mean values based on the groups from last iteration

    Args:
    df: DataFrame of points containing the assigned classes.

    Returns:
    return updated mean1 and mean2 based on the value in column 'class'.
    """
    gb = df.groupby('class')
    df_class_split = pd.Series([gb.get_group(group) for group in gb.groups])

    class_1 = df_class_split[0]
    class_2 = df_class_split[1]

    rand_class_1 = np.random.randint(len(class_1))
    rand_class_2 = np.random.randint(len(class_2))

    return class_1.iloc[rand_class_1], class_2.iloc[rand_class_2]

def kMeans(df, iterations):
    """
    Implements the K-means clustering method

    Args:
    df: DataFrame of points containing the assigned classes.
    iterations: Number of iterations

    Returns:
    df: Resulted DataFrame from the clustering method with the assigned classes
    mean1: DataFrame with a single value containing the mean point of the first class
    mean2: DataFrame with a single value containing the mean point of the second class
    """
    mean1, mean2 = initializeMeans(df)

    for iteration in range(iterations):

        print("Iteration {}/{}".format(iteration,iteration), mean1.name, mean2.name)

        for i in df.index:
            df = euclideanDist(df, i, mean1, mean2)

        mean1, mean2 = updateMean(df)

    return df, mean1, mean2

def generatePlot(df, mean1, mean2):
    """
    Generates and shows a pyplot with the distribution of the features and classes

    Args:
    df: DataFrame.
    pointIDX: Index of the point the DataFrame
    mean1: DataFrame with a single row
    mean2: DataFrame with a single row

    Returns:
    None
    """

    plt.clf()

    colors = ["limegreen", "royalblue"]
    classes = df['class'].unique()

    for i, el in enumerate(classes):
        plt.scatter(df.loc[df['class'] == el]['shoe_size'], df.loc[df['class'] == el]['height'], color=colors[i], label="Class "+str(el))

    plt.scatter(mean1['shoe_size'], mean1['height'], label='Mean '+str(classes[0]), color='purple')
    plt.scatter(mean2['shoe_size'], mean2['height'], label='Mean '+str(classes[1]), color='orange')

    plt.xlabel('shoe_size')
    plt.ylabel('height')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Loading data
    filepath = "Dataminers 2023H2.tsv"
    df = pd.read_table(filepath)

    # Transforming and cleaning data
    df = transform_data(df)
    df = clean_data(df)

    # Assign all points to class 1
    df['class'] = 1

    # Normalize shoe_size
    df['shoe_size'] = df['shoe_size'] / max(df['shoe_size'])

    # Normalize height
    df['height'] = df['height'] / max(df['height'])

    # Visualisation of the Distribution
    # plt.title('Scatter plot of the Distribution of data')
    # plt.scatter(df['shoe_size'], df['height'])
    # plt.show()

    # Calling the implementation of K-Means
    df, mean1, mean2 = kMeans(df, 5)

    # Generating viualisation
    generatePlot(df, mean1, mean2)