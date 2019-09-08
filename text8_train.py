from gensim.models import word2vec
import time

start_time = time.time()
sentence = word2vec.Text8Corpus('./text8')
model = word2vec.Word2Vec(sentence, size = 200)
print(time.time() - start_time)
model.save('./tmp/text8.model')