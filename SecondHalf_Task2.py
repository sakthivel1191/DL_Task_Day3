import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
print ("loading GloVe model...")
model=api.load("glove-wiki-gigaword-50")
print("Model loaded successfully")
print("\n5 Nearest words to king:")
for word,score in model.most_similar("king",topn=5):
    print(word,round(score,3))
print("\n5 Nearest words to computer:")
for word,score in model.most_similar("computer",topn=5):
    print(word,round(score,3))
print("\nAnalogy:king - man + woman")
result=model.most_similar(
    positive=["king","woman"],
    negative=["man"],
    topn=5
)

for word,score in result:
    print(word,round(score,3))
def sentence_vector(sentence):
    words=sentence.lower().split()
    vectors=[]
    for word in words:
        word=word.strip(".,!?")
        if word in model:
            vectors.append(model[word])
    if len(vectors)==0:
        return np.zeros(model.vector_size)
    return np.mean(vectors,axis=0)
sentence1="the cat sits on the mat"
sentence2="the dog sits on the floor"
vector1=sentence_vector(sentence1)
vector2=sentence_vector(sentence2)
similarity=cosine_similarity(
    [vector1],[vector2]
)[0][0]
print("\nSentence1:",sentence1)
print("Sentence2:",sentence2)
print("Sentence Similarity:",round(similarity,3))
words=[
    "king","queen","man","women","boy","girl",
    "father","mother","brother","sister",
    "computer","keyboard","screen","internet","software",
    "car","bus","train","road","travel",
    "apple","orange","banana","fruit","food",
    "dog","cat","bird","fish","animal"
]
word_vectors=[]
for word in words:
    word_vectors.append(model[word])
word_vectors=np.array(word_vectors)
pca=PCA(n_components=2)
vector_2d=pca.fit_transform(word_vectors)
plt.figure(figsize=(12,8))
plt.scatter(vector_2d[:,0],vector_2d[:,1])
for i in range(len(words)):
    plt.annotate(
        words[i],
        (vector_2d[i,0],vector_2d[i,1])
    )
plt.title("GloVe Word Embedding using PCA")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.grid(True)
plt.show()