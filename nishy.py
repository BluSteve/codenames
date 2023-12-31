import itertools

import gensim
from nltk.corpus import wordnet as wn
from nltk.data import find
from nltk.metrics.distance import edit_distance

word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
SAMENESS_THRESHOLD = 0.5
GOOD_THRESHOLD = 0.1
EMPHASIS_FACTOR = 1
MAX_COMBI = 5


class NishyBot:
    def __init__(self, good, bad, okay, assassin):
        self.good = good
        self.bad = bad
        self.okay = okay
        self.assassin = assassin
        self.hints = None

    def filter_hints(self, hints):
        barray = [True] * len(hints)
        for word in self.good:
            leven = [edit_distance(x, word) for x in hints]

            for i in range(len(hints)):
                if leven[i] / len(word) < SAMENESS_THRESHOLD:
                    barray[i] = False

        final_hints = [x for i, x in enumerate(hints) if barray[i]]
        return final_hints

    def get_similar_hints(self):
        combined = set()

        for i in range(1, MAX_COMBI):
            for words in list(itertools.combinations(self.good, i)):
                try:
                    for hint in self.filter_hints([x[0] for x in model.most_similar(positive=list(words), topn=10)]):
                        combined.add(hint)
                except KeyError:
                    pass
        return list(combined)

    def get_google_words(self):
        with open('google-10000-english.txt', 'r') as f:
            google_words = f.readlines()
        google_words = [x.strip() for x in google_words]
        google_words = self.filter_hints(google_words)
        return google_words

    def to_synsets(words): # ALL synsets for a word. does not ensure 1 to 1 match
        r = []
        for x in words:
            synsets = wn.synsets(x)
            if len(synsets) > 0:
                r.append(synsets[0])

        return r

    def get_hints(self, n):
        good_synsets = NishyBot.to_synsets(self.good)
        bad_synsets = NishyBot.to_synsets(self.bad)
        okay_synsets = NishyBot.to_synsets(self.okay)
        assassin_synsets = NishyBot.to_synsets(self.assassin)

        hints_synsets = []
        similar_hints = []
        for word in self.get_similar_hints():
            synsets = wn.synsets(word)
            if len(synsets) > 0:
                hints_synsets.append(synsets[0])
                similar_hints.append(word)

        google_synsets = []
        final_google_words = []
        for word in self.get_google_words():
            synsets = wn.synsets(word)
            if len(synsets) > 0:
                google_synsets.append(synsets[0])
                final_google_words.append(word)

        print(len(hints_synsets), len(similar_hints), len(google_synsets), len(final_google_words))

        google_synsets = hints_synsets + google_synsets
        final_google_words = similar_hints + final_google_words

        to_sort = []

        for i, x in enumerate(google_synsets):
            good_sum = 0
            good_count = 0
            for y in good_synsets:
                ps = wn.path_similarity(x, y)
                good_sum += ps ** EMPHASIS_FACTOR
                if ps > GOOD_THRESHOLD:
                    # if (final_google_words[i] == 'instrumentation'):
                    #     print(y)
                    good_count += 1
            bad_sum = 0
            for y in bad_synsets:
                bad_sum += wn.path_similarity(x, y) ** EMPHASIS_FACTOR
            okay_sum = 0
            for y in okay_synsets:
                okay_sum += wn.path_similarity(x, y) ** EMPHASIS_FACTOR
            assassin_sum = 0
            for y in assassin_synsets:
                assassin_sum += wn.path_similarity(x, y) ** EMPHASIS_FACTOR

            index = good_sum * 2 - bad_sum * 2 - okay_sum - assassin_sum * 5

            to_sort.append((final_google_words[i], x, good_count, index))

            # print(x, index, good_sum, bad_sum, okay_sum, assassin_sum)

        to_sort.sort(key=lambda x: x[-1], reverse=True)

        dict = {}
        for x in to_sort[::-1]:
            dict[x[1]] = x

        vc = list(dict.values()).copy()
        vc.sort(key=lambda x: x[-1], reverse=True)

        to_give = [(x[0], x[2], x[3]) for x in vc]
        return to_give[:n]

    def get_shortlist(self, n=20):
        hints = self.get_hints(n)
        hints.sort(key=lambda x:x[1], reverse=True)
        return hints[:5]


if __name__ == '__main__':
    good = ['door','cheese','bank','sleep','anthem','forest','sink','jellyfish']
    bad = ['berry','paper','europe','star','bean','slipper','bomb','axe','blues']
    okay = ['book','china','brick','house','farm','curry','germany']
    assassin = ['medic']
    n = NishyBot(good, bad, okay, assassin)
    print(n.get_shortlist())
