import pyrepseq.pwdistances as pwdist
import numpy as np

def test_lev():

    s = ['martina', 'martin']
    s1 = ['martina', 'martino']
    np.testing.assert_array_equal(pwdist.levenshtein_dist(s), np.array([[0,1], [1,0]]))
    np.testing.assert_array_equal(pwdist.levenshtein_dist(s, s1), np.array([[0,1], [1,1]]))

def test_lev_weighted():

    s = ['martina', 'martin']
    s1 = ['martina', 'martino']
    np.testing.assert_array_equal(pwdist.weighted_levenshtein_dist(s).round(2), np.array([[0,1+np.log(4)], [1+np.log(4),0]]).round(2))
    np.testing.assert_array_equal(pwdist.weighted_levenshtein_dist(s, s1).round(2), np.array([[0,1], [1+np.log(4),1+np.log(4)]]).round(2))

def test_triplet():
    s = ['martina', 'martin']
    s1 = ['martina', 'martino']

    X = 4/np.sqrt(6*5) # triplet sim between martina and martin - note this is a bit of a weird behaviour of R's stringdot function (i.e. one extra triplet counted in each)
    np.testing.assert_array_equal(pwdist.triplet_similarity(s).round(2), np.array([[1,X], [X,1]]).round(2))
    Y = 4/np.sqrt(6*6)
    np.testing.assert_array_equal(pwdist.triplet_similarity(s, s1).round(2), np.array([[1,Y], [X,X]]).round(2))

def test_triplet_diversity():
    s = ['martina', 'martin']
    s1 = ['martina', 'martino']

    X = 1-4/np.sqrt(6*5) # triplet sim between martina and martin - note this is a bit of a weird behaviour of R's stringdot function (i.e. one extra triplet counted in each)
    np.testing.assert_array_equal(pwdist.triplet_diversity(s).round(2), np.array([[0,X], [X,0]]).round(2))
    Y = 1-4/np.sqrt(6*6)
    np.testing.assert_array_equal(pwdist.triplet_diversity(s, s1).round(2), np.array([[0,Y], [X,X]]).round(2))