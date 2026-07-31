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

def mapping(tokens):
    word2id = {}
    id2word = {}
    # Use sorted(set()) to ensure consistency across separate executions
    for i, token in enumerate(sorted(set(tokens))):
        word2id[token] = i
        id2word[i] = token
    return word2id, id2word

word_to_id, id_to_word = mapping(tokens)

def concat(*iterables):
    for iterable in iterables:
        yield from iterable

def one_hot_encode(word_id, vocab_size):
    res = [0] * vocab_size
    res[word_id] = 1
    return res

def generate_training_data(tokens, word_to_id, window):
    """ Returns two numpy arrays: X (center word one-hot vectors) and y (context word one-hot vectors)."""
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
    and W2 is the output weight matrix. Biases: Tomas Mikolov stated that 
    biases can be excluded because they do not provide a significant improvement. """ 
    # Scale initial weights down to prevent exponential explosion in softmax
    model = {
        "w1": np.random.randn(vocab_size, n_embedding) * 0.01,
        "w2": np.random.randn(n_embedding, vocab_size) * 0.01
    }
    return model

# 100 or 300 to capture more semantic information, but for simplicity we use 10 here.
model = init_network(len(word_to_id), 10)

def softmax(X):
    # Vectorized, numerically stable calculation returning a NumPy matrix
    X_arr = np.asarray(X)
    exp_X = np.exp(X_arr - np.max(X_arr, axis=-1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

# Retrieving cached activations needed for backpropagation during training.
def forward(model, X, return_cache=True):
    cache = {}
    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])
    if not return_cache:
        return cache["z"]
    return cache

# Loss
def cross_entropy(z, y):
    # Clip z to prevent log(0) numerical underflow errors
    z = np.clip(z, 1e-15, 1.0 - 1e-15)
    return -np.sum(y * np.log(z))

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

# Train the network
# print(plt.style.available)
plt.style.use("seaborn-v0_8") 
n_iter = 50
learning_rate = 0.01

history = [backward(model, X, y, learning_rate) for _ in range(n_iter)]

# Plot verification
plt.plot(range(len(history)), history, color="skyblue")
plt.title("Word2Vec Training Loss Profile")
plt.xlabel("Iterations")
plt.ylabel("Cross-Entropy Loss")
plt.show()

# Inference evaluation
learning = one_hot_encode(word_to_id["learning"], len(word_to_id))
result = forward(model, [learning], return_cache=False)[0] 
print("--- Context prediction ranking for 'learning' ---")
for word in (id_to_word[idx] for idx in np.argsort(result)[::-1]): 
    print(word)

def get_embedding(model, word):
    try:
        idx = word_to_id[word]
    except KeyError:
        print(f"'{word}' not in corpus")
        return None
    one_hot = one_hot_encode(idx, len(word_to_id))
    # Enforce a 2D batch dimension array structure [one_hot] for forward pass matching
    return forward(model, [one_hot])["a1"][0]

print("\n--- Word Embedding Vector for 'machine' ---")
print(get_embedding(model, "machine"))
