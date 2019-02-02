from flask import Flask, request, render_template


#recommender data



import numpy as np
import pandas as pd


header = ['user_id', 'item_id', 'rating', 'timestamp']
BookData = pd.read_csv('ml-100k/u.data', sep='\t', names=header)


#number of users and items
n_users = BookData.user_id.unique().shape[0]
n_items = BookData.item_id.unique().shape[0]



#creating training and testing data
from sklearn.model_selection import train_test_split
ratings_train, ratings_test = train_test_split(BookData, test_size=0.25)





#training and testing matrix
training_matrix = np.zeros((n_users+1, n_items+1))
for line in ratings_train.itertuples():
    training_matrix[line[1], line[2]] = line[3]

testing_matrix = np.zeros((n_users+1, n_items+1))
for line in ratings_test.itertuples():
    testing_matrix[line[1], line[2]] = line[3]





#similarity matrices
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(training_matrix, metric='cosine')
item_similarity = pairwise_distances(training_matrix.T, metric='cosine')





# prediction method

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


#prediction matrix
item_prediction = predict(training_matrix, item_similarity, type='item')
user_prediction = predict(training_matrix, user_similarity, type='user')





#generation of RMSE score
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


#select top ten items
















app = Flask(__name__)

@app.route('/')
def index():
  return render_template("testing.html")



if __name__ == "__main__":
    app.run(debug = True)

