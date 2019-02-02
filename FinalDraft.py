
# coding: utf-8


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






# # Comparision based model for improving predictions in item-item collaborative filtering model







# ## mess with ratings



from random import randint
randomItem1 = randint (0, 1683)


randomItem2 = randint(0, 1683)

# ### copy item prediction matrix for manipulation and comparison

copyItem_prediction = item_prediction






# ## TESTING BOXES HERE





# # COMPARISON MODEL TESTING CODE HERE


for i in BookData.itertuples():
    #for j in range(0,10):
        choice = randint(0,1)
        randomItem1 = randint (0, 1683)
        randomItem2 = randint(0, 1683)
        
        
        #testing if combination of userid and item exist, otherwise break
        test = BookData.loc[BookData['user_id'] == i.user_id]
        testArray = test['item_id'].values
        if (randomItem1 in testArray) & (randomItem2 in testArray):
            series1 = (BookData.loc[(BookData['user_id']==i.user_id) & (BookData['item_id'] == randomItem1)]).rating.values
            series2 = (BookData.loc[(BookData['user_id']==i.user_id) & (BookData['item_id'] == randomItem2)]).rating.values
            rating1 = series1[0]
            rating2 = series2[0]
            predictionScore1 = copyItem_prediction[i.user_id, randomItem1]
            predictionScore2 = copyItem_prediction[i.user_id, randomItem2]
        
            ratingDiff = rating1 - rating2
        
            update1 = 0.01
            update2 = 0.001
            update3 = 0.0001
        else: 
            break
            
        if(choice == 0):
            while (abs(ratingDiff) < 4) & (min(rating1, rating2) != 0) and (max(rating1, rating2)!= 5)  :
                if(ratingDiff>0):
                    if(ratingDiff == 3):
                        predictionScore1 = predictionScore1 + update3
                        predictionScore2 = predictionScore2 - update3 
                    elif(ratingDiff == 2):
                        predictionScore1 = predictionScore1 + update2
                        predictionScore2 = predictionScore2 - update2
                    elif(ratingDiff == 1):
                        predictionScore1 = predictionScore1 + update1
                        predictionScore2 = predictionScore2 - update1
                if(ratingDiff<0):
                    if(ratingDiff == -3):
                        predictionScore1 = predictionScore1 + update1
                        predictionScore2 = predictionScore2 - update1
                    elif(ratingDiff == -2):
                        predictionScore1 = predictionScore1 + update2
                        predictionScore2 = predictionScore2 - update2
                    elif(ratingDiff == -1):
                        predictionScore1 = predictionScore1 + update3 
                        predictionScore2 = predictionScore2 - update3
                        
        elif (choice == 1):
            while  (abs(ratingDiff) < 4) & (min(rating1, rating2) != 0) & (max(rating1, rating2)!= 5)  :
                if(ratingDiff>0):
                    if(ratingDiff == 3):
                        predictionScore1 = predictionScore1 - update3
                        predictionScore2 = predictionScore2 + update3 
                    elif(ratingDiff == 2):
                        predictionScore1 = predictionScore1 - update2
                        predictionScore2 = predictionScore2 + update2
                    elif(ratingDiff == 1):
                        predictionScore1 = predictionScore1 - update1
                        predictionScore2 = predictionScore2 + update1
                if(ratingDiff<0):
                    if(ratingDiff == -3):
                        predictionScore1 = predictionScore1 - update1
                        predictionScore2 = predictionScore2 + update1
                    elif(ratingDiff == -2):
                        predictionScore1 = predictionScore1 - update2
                        predictionScore2 = predictionScore2 + update2
                    elif(ratingDiff == -1):
                        predictionScore1 = predictionScore1 - update3 
                        predictionScore2 = predictionScore2 + update3
            
        
        copyItem_prediction[i.user_id, randomItem1] = predictionScore1
        copyItem_prediction[i.user_id, randomItem2] = predictionScore2


print("End of program")





