import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # initialize
        best_bic = float('inf')
        best_model = None

        # SOL(Ricardo): test all number of states between a given
        # range
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                # define the model
                model_n = self.base_model(n)
                logL = model_n.score(self.X, self.lengths)
                # the number of params is defined as p = m^2 + km - 1 as
                # in MacDonald & Zucchinni (2009) where k is the number
                # of parameters of the underlying distribution (2 for
                # univariate normal dist [mean, std]), since we have
                # several variables, it needs to be multiplied by the
                # number of features. m is the number of states
                k = 2 * model_n.n_features
                params = n ** 2 + k * n - 1
                # BIC = -2 * logL + p * logN
                bic = -2 * logL + params * np.log(len(self.X))
                # update best config.
                if bic < best_bic:
                    best_bic = bic
                    best_model = model_n
            except:
                # as suggested, ignore those models which cannot be
                # trained by hmmlearn
                pass

        return best_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # initialize
        best_model = None
        best_dic = float('-inf')

        # SOL(Ricardo): test a range of number of states to compare
        # their scores
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model_n = self.base_model(n)
                # current model score log(P(X(i))
                logX = model_n.score(self.X, self.lengths)
                # we need the scores of the model excluding the current word
                logX_but_i = [model_n.score(x, l)
                              for w, (x, l) in self.hwords.items()
                              if w != self.this_word]
                # 1/(M-1)SUM(log(P(X(all but i))
                logX_but_i = sum(logX_but_i) / len(logX_but_i)
                dic = logX - logX_but_i

                # update thebest config
                if dic > best_dic:
                    best_dic = dic
                    best_model = model_n
            except:
                # as suggested, ignore those models which cannot be
                # trained by hmmlearn
                pass
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # initialize
        best_cv = float('-inf')
        best_model = None

        # SOL(Ricardo): test a range of number of states to compare
        # their scores
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model_n = self.base_model(n)
                split_method = KFold()

                # variables to store CV test sets results
                test_score = 0
                test_count = 0

                # iterate through the splits
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                    # get new model inputs
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)

                    # fit the new model on the training set
                    train = self.base_model(n)

                    # get the score for the test set_index
                    test_score += train.score(test_X, test_lengths)
                    test_count += 1

                # average the scores of the CV test sets
                if test_count == 0:
                    # avoid division by 0
                    test_score = float('-Inf')
                else:
                    test_score = test_score / test_count

                # update values
                if test_score > best_cv:
                    best_cv = test_score
                    best_model = model_n

            except:
                pass

        # if it has failed, return minimal model as default
        if best_model is None:
            best_model = self.base_model(self.min_n_components)

        return best_model
