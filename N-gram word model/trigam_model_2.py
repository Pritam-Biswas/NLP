from collections import defaultdict
import math
import os,sys
import os.path
from collections import deque

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
    sequence = deque(sequence)
    if n>1:
        sequence.extendleft(['START']*(n-1))
    else:
        sequence.extendleft(['START'])
    sequence.extend(['STOP'])
    sequence = list(sequence)
    ngrams = []
    for seq in range(0,len(sequence)-(n-1)):
        ngrams.append(tuple(sequence[seq:seq+n]))
    return ngrams


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        self.total_words = 0

        for sentence in corpus:
            self.total_words+=len(sentence)
            grams = [get_ngrams(sentence, 1), get_ngrams(sentence, 2), get_ngrams(sentence, 3)]
            for ng, newList in enumerate(grams):
                for t in newList:
                    if ng is 0:
                        self.unigramcounts[t] += 1
                    elif ng is 1:
                        self.bigramcounts[t] += 1
                    else:
                        self.trigramcounts[t] += 1

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        try:
            return self.trigramcounts[trigram] / self.bigramcounts[trigram[0:2]]
        except Exception as e:
            return 0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        uni = (bigram[0],)
        try:
            return self.bigramcounts[bigram] / self.unigramcounts[uni]
        except Exception as e:
            return 0

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.
        try:
            return self.unigramcounts[unigram] / self.total_words
        except Exception as e:
            return 0

    def generate_sentence(self, t=20):
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
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0
        uni = (trigram[2],)
        return lambda1 * self.raw_trigram_probability(trigram)  + lambda2 * self.raw_bigram_probability(trigram[1:3]) + lambda3 * self.raw_unigram_probability(uni)

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigram = get_ngrams(sentence, 3)
        sent_prob = 0
        for tri in trigram:
            try:
                sent_prob += math.log2(self.smoothed_trigram_probability(tri))
            except ValueError:
                sent_prob += 0
        print ('Sentence probability:'+str(sent_prob))
        return sent_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """
        sent_logprob = 0
        wc =0
        count =0
        for ind, sentence in enumerate(corpus):
            count+=1
            print('Evaluating sentence :'+str(count))
            sent_logprob += self.sentence_logprob(sentence)
            wc += len(sentence)
        norm_sent_logprob = sent_logprob / wc
        return 2 ** (-norm_sent_logprob)

def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        pp_model1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        pp_model2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        if pp_model1 < pp_model2:
            correct+=1
        total+=1

    for f in os.listdir(testdir2):
        pp_model2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        pp_model1 = model1.perplexity(corpus_reader(os.path.join(testdir2,f),model1.lexicon))
        if pp_model1 > pp_model2:
            correct += 1
        total += 1

    return correct/total


if __name__ == "__main__":
    model = TrigramModel(sys.argv[1])
    print(model.total_words)

    # put test code here...
    # or run the script from the command line with
    # $ python -i prof_trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.

    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment("hw1_data\\ets_toefl_data\\train_high.txt", "hw1_data\\ets_toefl_data\\train_low.txt", "hw1_data\\ets_toefl_data\\test_high",\
    #                  "hw1_data\\ets_toefl_data\\test_low")
    # print(acc)