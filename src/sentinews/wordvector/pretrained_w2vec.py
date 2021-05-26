from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec


# model = Word2Vec.load_word2vec_format("../../../sentinews/pre-trained/wikipedia-320.txt",
model = word2vec.KeyedVectors.load_word2vec_format("../../../pre-trained/wikipedia-160.txt", binary=False)
katvec = model['goed']
print(model.most_similar('goed'))
print(katvec)
