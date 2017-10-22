from keras.preprocessing.text import Tokenizer
import numpy as np
import collections

class Review(object):
    def __init__(self, ProductId, UserId, ProfileName, Helpfulness, Score, Time, Summary, Text):
        self.ProductId = ProductId
        self.UserId = UserId
        self.ProfileName = ProfileName
        self.Helpfulness = Helpfulness
        self.Score = Score
        self.Time = None
        self.Summary = Summary
        self.Text = None
        self.Classification = None

    def __repr__(self):
        return """Review(ProductID = {0},
                  UserId = {1},
                  ProfileName = {2},
                  Helpfulness = {3},
                  Score = {4},
                  Time = None,
                  Summary = {5},
                  Text = None)""".format(self.ProductId, self.UserId, self.ProfileName, self.Helpfulness, self.Score, self.Summary)

class Parser(object):
    def __init__(self):
        self.reader = open("SentimentTrainingData.txt", "r")
        self.counter = 0
        self.reviews = []

    def ReadReview(self):
        self.counter += 1
        pid = self.reader.readline()[18:].strip()
        if(pid == ""):
            return False
        review = Review(pid, # ProductID
                        self.reader.readline()[14:].strip(), # UserID
                        self.reader.readline()[19:].strip(), # ProfileName
                        self.reader.readline()[19:].strip(), # Helpfulness
                        float(self.reader.readline()[13:].strip()), # Score
                        int(self.reader.readline()[12:].strip()), # Time
                        self.reader.readline()[15:].strip(), # Summary
                        self.reader.readline()[12:].strip()) # Text
        
        #print(review)
        if (self.counter % 10000 == 0):
            print(self.counter)
        review.Classification = 1 if review.Score >= 4 else 0
        self.reader.readline() # Advance
        if (review.Score == 3): # Ignore neutral
            pass
        else:
            self.reviews.append(review)
        return True
    
    def ParseReviews(self, num_words=None, skip_top=0, maxlen=None, seed=113, start_char=1, oov_char=2, index_from=3):
        tok = Tokenizer(num_words=None,
                        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                        lower=True,
                        split=" ",
                        char_level=False)

        np.random.seed(seed)
        np.random.shuffle(self.reviews)

        texts = [x.Summary for x in self.reviews]
        tok.fit_on_texts(texts)

        #print(tok.word_index) # Used this to determine how many unique words there were, as this is a listing of all the word indices.
        
        xs = tok.texts_to_sequences(texts)
        ys = np.array([x.Classification for x in self.reviews])
        
        if start_char is not None:
            xs = [[start_char] + [w + index_from for w in x] for x in xs]
        elif index_from:
            xs = [[w + index_from for w in x] for x in xs]

        if maxlen:
            xs, ys = _remove_long_seq(maxlen, xs, ys)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                             'Increase maxlen.')
        if not num_words:
            num_words = max([max(x) for x in xs])

        # by convention, use 2 as OOV word
        # reserve 'index_from' (=3 by default) characters:
        # 0 (padding), 1 (start), 2 (OOV)
        if oov_char is not None:
            xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
        else:
            xs = [[w for w in x if (skip_top <= w < num_words)] for x in xs]

        length = len(ys)
        eighty = int(length*0.8)
        x_train = xs[:eighty]
        x_test = xs[eighty:]
        y_train = ys[:eighty]
        y_test = ys[eighty:]
        return (x_train, y_train), (x_test, y_test)

def main():
    p = Parser()
    for i in range(100000):
        p.ReadReview()
    (x_train, y_train), (x_test, y_test) = p.ParseReviews()
    print(collections.Counter(y_train))
    print(collections.Counter(y_test))
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

if __name__ == '__main__':
    main()