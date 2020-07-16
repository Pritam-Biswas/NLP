import sys
from collections import defaultdict
import math
import random
import os
from collections import deque
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2019
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    # CREATE PADDING
    front_pad = []
    for i in range(1,n):
        front_pad.append('START')

    if n <=2:
        tmp_sequence = ['START'] + sequence + ['STOP']
    else:
        tmp_sequence = front_pad + sequence + ['STOP']
    n_gram_item  = []
    n_gram_tuple_list = []

    l = len(tmp_sequence)
    for i in range(0,l-n+1):
        n_gram_item = tmp_sequence[i:i+n]
        n_gram_tuple = tuple(n_gram_item)
        n_gram_tuple_list.append(n_gram_tuple)
    return n_gram_tuple_list  


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        self.total_words = 0
        self.total_sentences = 0
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int) 
        self.trigramcounts = defaultdict(int) 

        ##Your code here
        count = 0
        for sentence in corpus:
            u_gram = get_ngrams(sentence,1)
            b_gram = get_ngrams(sentence,2)
            t_gram = get_ngrams(sentence,3)

            self.total_words += len(u_gram)
            self.total_sentences += 1
            
            for gram in u_gram:
                self.unigramcounts[gram] += 1

            for gram in b_gram:
                self.bigramcounts[gram] += 1
            
            for gram in t_gram:
                self.trigramcounts[gram] += 1
                 
        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        trigram_list = list(trigram)
        trigram_list.pop(-1)
        bigram = tuple(trigram_list)
        trigram_count = self.trigramcounts[trigram]
        bigram_count = self.bigramcounts[bigram]

        try:
            prob = trigram_count / bigram_count
        except Exception as e:
            return 0

        return prob

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        bigram_list = list(bigram)
        bigram_list.pop(-1)
        unigram = tuple(bigram_list)
        bigram_count = self.bigramcounts[bigram]
        unigram_count = self.unigramcounts[unigram]
        try:
            prob = bigram_count / unigram_count
        except Exception as e:
            return 0

        return prob
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it. 
        unigram_count = self.unigramcounts[unigram]
        try:
            prob = unigram_count / self.total_words
        except Exception as e:
            return 0

        return prob

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        trigram_prob = self.raw_trigram_probability(trigram)
        
        trigram_list = list(trigram)
        bigram_list = trigram_list[1:]
        unigram_list = bigram_list[1:]

        bigram = tuple(bigram_list)
        unigram = tuple(unigram_list)

        bigram_prob = self.raw_bigram_probability(bigram)
        unigram_prob = self.raw_unigram_probability(unigram)

        smooth_prob = lambda1*trigram_prob + lambda2*bigram_prob + lambda3*unigram_prob
        return smooth_prob

        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigram_list = get_ngrams(sentence, 3)
        log_prob = 0.0
        for trigram in trigram_list:
            trigram_prob = math.log2(self.smoothed_trigram_probability(trigram))
            log_prob += trigram_prob
        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        total_log_prob = 0.0
        count = 0
        word_count = 0
        for sentence in corpus:
            count +=1
            word_count += len(sentence)
            total_log_prob += self.sentence_logprob(sentence)
        total_log_prob = total_log_prob / word_count

        return 2 ** (-total_log_prob)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
        acc = 0.0
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp < pp2:
                correct+=1
            total +=1
            # .. 
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp < pp2:
                correct += 1
            total +=1
            # .. 
        
        acc = correct/ total
        return acc

if __name__ == "__main__":

    ## test code 
    output = get_ngrams(["natural","language", "processing"], 1)
    print(output)
    output = get_ngrams(["natural","language", "processing"], 2)
    print(output)
    output = get_ngrams(["natural","language", "processing"], 3)
    print(output)

    model = TrigramModel(sys.argv[1]) 
    print (model.trigramcounts[('START','START','the')])
    print (model.bigramcounts[('START','the')])
    print (model.unigramcounts[('the',)]) 

    print (model.total_words)
    raw_uni_prob = model.raw_unigram_probability(('the',))
    print (raw_uni_prob)
    
    raw_bi_prob = model.raw_bigram_probability(('START', 'the'))
    print(raw_bi_prob)

    raw_tri_prob = model.raw_trigram_probability(('START', 'START','the'))
    print(raw_tri_prob)

    smooth_tri_prob = model.smoothed_trigram_probability(('START', 'START','the'))
    print (smooth_tri_prob)

    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("Perplexity score for : " + sys.argv[2]+ " is = "+str(pp))



    # # Essay scoring experiment: 
    acc = essay_scoring_experiment("hw1_data\\ets_toefl_data\\train_high.txt", "hw1_data\\ets_toefl_data\\train_low.txt", "hw1_data\\ets_toefl_data\\test_high",\
                     "hw1_data\\ets_toefl_data\\test_low")
    print("Accuracy fraction for toefl data set is = " +str(acc))

