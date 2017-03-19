import argparse
import re
import os
import csv
import math
import collections as coll
import numpy as np

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def parse_file(filename):
    """
    Given a filename outputs user_ratings and movie_ratings dictionaries
    Input: filename
    Output: user_ratings, movie_ratings
        where:
            user_ratings[user_id] = {movie_id: rating}
            movie_ratings[movie_id] = {user_id: rating}
    """
    filename = np.loadtxt(filename, delimiter=',')
    user_ratings = coll.defaultdict(dict)
    movie_ratings = coll.defaultdict(dict)
    for row in range(len(filename)):
        user_ratings[filename[row, 1]][filename[row, 0]] = filename[row, 2]
        movie_ratings[filename[row, 0]][filename[row, 1]] = filename[row, 2]
    return user_ratings, movie_ratings


def compute_average_user_ratings(user_ratings):
    """ Given a the user_rating dict compute average user ratings
    Input: user_ratings (dictionary of user, movies, ratings)
    Output: ave_ratings (dictionary of user and ave_ratings)
    """
    ave_ratings = dict((index, np.mean(ratings.values())) for index, ratings in user_ratings.iteritems())
    return ave_ratings


def compute_user_similarity(d1, d2, ave_rat1, ave_rat2):
    """ Computes similarity between two users
        Input: d1, d2, (dictionary of user ratings per user)
            ave_rat1, ave_rat2 average rating per user (float)
        Ouput: user similarity (float)
    """
    key1 = d1.keys()
    key2 = d2.keys()
    common = set(key1) & set(key2)
    if not common:
        return 0.0
    else:
        num = 0
        bottom1 = 0
        bottom2 = 0
        for value in common:
            num += (d1[value] - ave_rat1) * (d2[value] - ave_rat2)
            bottom1 += (d1[value] - ave_rat1) ** 2
            bottom2 += (d2[value] - ave_rat2) ** 2
    if math.sqrt(bottom1 * bottom2) == 0:
        return 0.0
    else:
        return float(num / math.sqrt(bottom1 * bottom2))


def predict_rating(newuser, movie):
    user_rated = movie_dictionary[movie]
    denom = 0
    num = 0
    for user in user_rated:
        sim = compute_user_similarity(user_dictionary[newuser], user_dictionary[user], avg_dictionary[newuser],
                                      avg_dictionary[user])
        denom += abs(sim)
        num += sim * (user_dictionary[user][movie] - avg_dictionary[user])
    if denom == 0 or num == 0:
        pred = float(avg_dictionary[newuser])
    else:
        pred = float(avg_dictionary[newuser]) + (num / denom)
    return pred


def main():
    """
    This function is called from the command line via
    python cf.py --train [path to filename] --test [path to filename]
    """
    global train_file
    global test_file
    global user_dictionary
    global movie_dictionary
    global avg_dictionary
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    test_file = np.loadtxt(test_file, delimiter=',')

    user_dictionary, movie_dictionary = parse_file(train_file)
    avg_dictionary = compute_average_user_ratings(user_dictionary)

    mse = []
    with open('predictions.txt', 'w') as txtfile:
        for row in test_file:
            writer = csv.writer(txtfile, delimiter=' ')
            pred = predict_rating(int(row[1]), row[0])
            txtfile.write("{0} {1} {2} {3}\n".format(row[0], int(row[1]), row[2], pred))
            mse.append(row[2] - pred)

    mae = np.mean([abs(m) for m in mse])
    rmse = math.sqrt(np.mean([m ** 2 for m in mse]))
    print("MAE = {0:.5f}".format(mae))
    print("RMSE = {0:.5f}".format(rmse))


if __name__ == '__main__':
    main()
    