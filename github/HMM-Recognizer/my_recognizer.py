import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # loop through test set
    for i in range(test_set.num_items):

        # initialize
        guess = None
        best_prob = float('-inf')

        # get the items
        X, lengths = test_set.get_item_Xlengths(i)

        # initialize an empty item which will be stored in the
        # final probabilities list. It is a dict{WORD:LogN}
        probability_item = {}

        # loop through all trained models
        for word, model in models.items():
            try:
                logN = model.score(X, lengths)
                probability_item[word] = logN

                # update the values
                if best_prob < logN:
                    best_prob = logN
                    guess = word
            except:
                # if error give probability 0 to the word
                probability_item[word] = np.log(0)


        # store values
        probabilities.append(probability_item)
        guesses.append(guess)

    return probabilities, guesses
