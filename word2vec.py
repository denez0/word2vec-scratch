import matplotlib.pyplot as plt
import numpy as np
import re

text = '''Machine learning is the study of computer algorithms that \
    improve automatically through experience. It is seen as a \
    subset of artificial intelligence. Machine learning algorithms \
    build a mathematical model based on sample data, known as \
    training data, in order to make predictions or decisions without \
    being explicitly programmed to do so. Machine learning algorithms \
    are used in a wide variety of applications, such as email filtering \
    and computer vision, where it is difficult or infeasible to develop \
    conventional algorithms to perform the needed tasks.'''

def tokenize(text):
    pattern = re.compile(r"\b[\w']+")
    return pattern.findall(text.lower())
tokens = tokenize(text)
# print(tokens)

def mapping(tokens):
    word2id = {}
    id2word = {}
    for i, token in enumerate(set(tokens)):
        word2id[token] = i
        id2word[i] = token
    return word2id, id2word
word_to_id, id_to_word = mapping(tokens)
# print(word_to_id)
# print(id_to_word)

def concat(*iterables):
    for iterable in iterables:
        yield from iterable

def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res

def generate_training_data(tokens, word_to_id, window):
    """ returns two numpy arrays: X (center word one-hot vectors) and y (context word one-hot vectors)."""
    X = []
    y = []
    n_tokens = len(tokens)
    vocab_size = len(word_to_id)
    for i in range(n_tokens):
        idx = concat(
        range(max(0, i - window), i), 
        range(i+1, min(n_tokens, i + window + 1))
    )
        for j in idx:
            X.append(one_hot_encode(word_to_id[tokens[i]], vocab_size)) 
            y.append(one_hot_encode(word_to_id[tokens[j]], vocab_size))
    return np.asarray(X), np.asarray(y)

X, y = generate_training_data(tokens, word_to_id, 2)
# print(f"Generated training data: X shape = {X.shape}, y shape = {y.shape}")

np.random.seed(42)
def init_network(vocab_size, n_embedding):
    """ n_embedding is the dimension of the embedding space. 
    The output = xW1W2, where x is the one-hot vector of the input word, W1 is the embedding matrix, 
    and W2 is the output weight matrix. """
    # biases: Tomas Mikolov, stated that biases were excluded because they did not provide 
    # a significant improvement

    model = {
        "w1": np.random.randn(vocab_size, n_embedding),
        "w2": np.random.randn(n_embedding, vocab_size)
    }
    return model

# 100 or 300 to capture more semantic information, but for simplicity we use 10 here.
model = init_network(len(word_to_id), 10)

def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res

# retrieving intermediate values (activations) during the forward pass, 
# which are needed for backpropagation during training.
def forward(model, X, return_cache=True):
    cache = {}
    
    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])
    if not return_cache:
        return cache["z"]
    return cache

# print((X @ model["w1"] @ model["w2"]).shape)

# loss 
def cross_entropy(z, y):
    # add small value to avoid log(0)?
    return - np.sum(np.log(z) * y)


def backward(model, X, y, alpha):
    # Consider replacing @ .T with a different function for faster matrix multiplication.
    # Also, stochastic SGD could be used for larger datasets 
    cache = forward(model, X)
    da2 = cache["z"] - y 
    dw2 = cache["a1"].T @ da2 
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1
    assert(dw2.shape == model["w2"].shape)
    assert(dw1.shape == model["w1"].shape)
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], y)

# print(plt.style.available)
plt.style.use("seaborn-v0_8")
n_iter = 50
learning_rate = 0.01

history = [backward(model, X, y, learning_rate) for _ in range(n_iter)]
plt.plot(range(len(history)), history, color="skyblue")
plt.show()

learning = one_hot_encode(word_to_id["learning"], len(word_to_id))
result = forward(model, [learning], return_cache=False)[0] # [0] because forward returns a list of one element corresponding to [learning], we want to get that element
for word in (id_to_word[id] for id in np.argsort(result)[::-1]): 
    print(word)

def get_embedding(model, word):
    try:
        idx = word_to_id[word]
    except KeyError:
        print("`word` not in corpus")
    one_hot = one_hot_encode(idx, len(word_to_id))
    return forward(model, one_hot)["a1"]

print(get_embedding(model, "machine"))
