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


    ####### Supervised Method using NBayes #######

def feature_extraction(data):
    # We want to try categorize the shoe size and height into my generalized categories (the feature of it).
    # The way we do that is to take the sum of the data points and rounding them into nearest 10.

    # sum of data[0] + data[1] rounded to nearest 10
    return round((data[0] + data[1]) / 10) * 10


def train(data):
    # declare our model
    model = {}

    # go through all the data that we have
    for example in data:
        # We want to predict the study line based on features (in it's set).
        # take out study line as the "label"
        studyLine = example[1]

        # find the feature between shoe and height using our feature extraction
        f_ShoeHeight = feature_extraction((example[0], example[2]))

        # this is basic declaration, for python syntax
        # if the studyLine and shoe height is not declared in the dictionary then we want to declare them first
        # to avoid crashes (when trying to set them)
        if studyLine not in model:
            model[studyLine] = {}
        if f_ShoeHeight not in model[studyLine]:
            model[studyLine][f_ShoeHeight] = 0
        if 'observed' not in model[studyLine]:
            model[studyLine]['observed'] = 0

        # here we increment the number of observations of that given feature to the study line
        model[studyLine][f_ShoeHeight] = model[studyLine][f_ShoeHeight] + 1

        # keep count of observations in total on this study line
        model[studyLine]['observed'] = model[studyLine]['observed'] + 1

    # Give back the model
    return model

# This method can predict an outcome based on a trained model (parameters)
def predict(model, data):
    # The features we want to predict based on is (data).
    # Extract the feature set from our prediction data
	f_x = feature_extraction(data)

    # declare our maximum likelihood estimation
	mle = {}
     
    # go through all the labels in our model
	for label in model:
		mle[label] = 0

		if f_x in model[label]:
            # if the feature is in the model then we want to calculate the mle
			mle[label] = model[label][f_x] / model[label]['observed']

    # find the best estimate based on the maximum likelihood estimation
	best_estimate = ['', 0]
	for label in mle:
		if mle[label] > best_estimate[1]: # just local maximum of MLE's
			best_estimate = [label, mle[label]]

	return best_estimate


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

    data = [
        ( 44, "MSc Software Design, ITU", 75.59 ), 
        ( 38, "MSc Software Design, ITU", 167 ), 
        ( 44, "MSc Software Design, ITU", 192 ), 
        ( 38, "MSc Software Design, ITU", 66 ), 
        ( 44, "MSc Software Design, ITU", 184 ), 
        ( 38, "MSc Software Design, ITU", 64 ), 
        ( 45, "MSc Software Design, ITU", 72 ), 
        ( 42, "MSc Software Design, ITU", 71.65 ), 
        ( 37.5, "MSc Software Design, ITU", 70 ), 
        ( 42.5, "MSc Software Design, ITU", 70.0787402 ), 
        ( 39, "MSc Software Design, ITU", 5.6 ), 
        ( 44, "MSc Software Design, ITU", 75.984 ), 
        ( 39, "MSc Software Design, ITU", 65 ), 
        ( 46, "MSc Software Design, ITU", 72 ), 
        ( 44, "MSc Software Design, ITU", 74 ), 
        ( 37, "MSc Software Design, ITU", 63.4 ), 
        ( 42.5, "MSc Software Design, ITU", 71.3 ), 
        ( 45, "MSc Computer Science, ITU", 187 ), 
        ( 39, "MSc Software Design, ITU", 174 ), 
        ( 39, "MSc Software Design, ITU", 69 ), 
        ( 44, "MSc Software Design, ITU", 73 ), 
        ( 44, "MSc Software Design, ITU", 6.1 ), 
        ( 39, "MSc Software Design, ITU", 68 ), 
        ( 43, "MSc Software Design, ITU", 73.23 ), 
        ( 38.5, "MSc Software Design, ITU", 68 ), 
        ( 39, "MSc Software Design, ITU", 178 ), 
        ( 38, "MSc elsewhere", 70.07 ), 
        ( 44, "MSc Software Design, ITU", 71 ), 
        ( 37, "MSc Software Design, ITU", 65 ), 
        ( 45, "MSc Software Design, ITU", 73 ), 
        ( 38, "MSc Software Design, ITU", 64 ), 
        ( 37, "MSc elsewhere", 63.78 ), 
        ( 38, "MSc Software Design, ITU", 64.56692913 ), 
        ( 36, "MSc Software Design, ITU", 63 ), 
        ( 40, "MSc Software Design, ITU", 68 ), 
        ( 43, "MSc Software Design, ITU", 71 ), 
        ( 39, "MSc Software Design, ITU", 67.7 ), 
        ( 42.5, "MSc elsewhere", 68.9 ), 
        ( 43.5, "MSc Software Design, ITU", 71.2 ), 
        ( 39, "MSc Software Design, ITU", 68 ), 
        ( 45, "MSc Software Design, ITU", 73 ), 
        ( 41, "MSc elsewhere", 70.5 ), 
        ( 43, "MSc elsewhere", 68 ), 
        ( 40, "MSc Software Design, ITU", 70.8 ), 
        ( 9, "MSc Software Design, ITU", 55 ), 
        ( 46, "MSc elsewhere", 70.9 ), 
        ( 38, "MSc Software Design, ITU", 173 ), 
        ( 40, "MSc Software Design, ITU", 183 ), 
        ( 45, "MSc Software Design, ITU", 74.8 ), 
        ( 42, "MSc Software Design, ITU", 72.44 ), 
        ( 37, "MSc Software Design, ITU", 63 ), 
        ( 39, "MSc Software Design, ITU", 66 ), 
        ( 45, "MSc Software Design, ITU", 70 ), 
        ( 38, "MSc elsewhere", 176 ), 
        ( 41, "MSc Software Design, ITU", 67.71654 ), 
        ( 39, "MSc elsewhere", 62 ), 
        ( 39, "MSc Software Design, ITU", 66 ), 
        ( 41, "MSc Software Design, ITU", 71.26 ), 
        ( 44, "MSc Software Design, ITU", 72 ), 
        ( 42, "MSc Software Design, ITU", 70.8 ), 
        ( 40, "MSc Software Design, ITU", 72 ), 
        ( 44, "MSc Software Design, ITU", 185 ), 
        ( 6, "MSc Software Design, ITU", 158 ), 
        ( 42, "MSc elsewhere", 72 ), 
        ( 41, "MSc elsewhere", 66.5354331 ), 
        ( 44, "MSc elsewhere", 75.6 ), 
        ( 42, "MSc Software Design, ITU", 70 ), 
        ( 7.5, "MSc Games ITU", 69 ), 
        ( 39, "MSc elsewhere", 68.89764 ), 
        ( 44.5, "MSc Software Design, ITU", 75 ), 
        ( 39, "MSc Software Design, ITU", 168 ), 
        ( 43, "MSc elsewhere", 75.19 ), 
        ( 40, "MSc elsewhere", 68 ), 
        ( 44, "MSc elsewhere", 69 ), 
        ( 43.5, "MSc Software Design, ITU", 73.62 ), 
        ( 36, "MSc elsewhere", 5.2 ), 
        ( 44, "MSc Software Design, ITU", 70 ), 
        ( 43, "MSc elsewhere", 188 ), 
        ( 45, "MSc elsewhere", 74 ), 
        ( 42.5, "MSc Software Design, ITU", 70.1 ), 
        ( 41.5, "MSc Software Design, ITU", 70.07 ), 
        ( 41, "MSc Software Design, ITU", 68 ), 
        ( 37, "MSc Software Design, ITU", 63.77 ), 
        ( 41.5, "MSc Software Design, ITU", 172 ), 
        ( 41.5, "MSc elsewhere", 67 ), 
        ( 38, "MSc Software Design, ITU", 64 ), 
        ( 38, "MSc Software Design, ITU", 62.59 ), 
        ( 37, "MSc Software Design, ITU", 150 ), 
        ( 45, "MSc Software Design, ITU", 186 ), 
        ( 44.5, "MSc Software Design, ITU", 73.622047 )
    ]


    # normalize data
    for i in range(len(data)):
        shoe_size, study_line, height = data[i]
        
        # normalize shoe size to european standard
        if shoe_size < 16:
            shoe_size = shoe_size * 5

        # normalize the height to cm
        if height <= 8:
            # feet to cm
            height = height * 30.48
        elif height < 100: # we don't account for dwarfs in normalization
            # inches to cm
            height = height * 2.54

        data[i] = (shoe_size, study_line, height)


    # Train the model and get the parameters (using above method)

    parameters = train(data)

    # Do a sample prediction with shoe size: 42 and height 185cm!
    test_value = (42, 185)
    prediction = predict(parameters, test_value) # best MLE

    print(prediction)