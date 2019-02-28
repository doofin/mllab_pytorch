import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from underscore import _ as u

# https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens


def get_input_layer(word_idx, vocabulary_size):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x


def word2v():
    corpus = [
        'he is a king',
        'she is a queen',
        'he is a man',
        'she is a woman',
        'warsaw is poland capital',
        'berlin is germany capital',
        'paris is france capital',
    ]

    tokenized_corpus = tokenize_corpus(corpus)
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    vocabulary_size = len(vocabulary)

    window_size = 2
    idx_pairs1 = []
    # for each sentence
    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]
        # for each word, threated as center word
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs1.append((indices[center_word_pos], context_word_idx))

    idx_pairs = np.array(idx_pairs1) # word index lists ,It gives us list of pairs center word, context indices:
    print("idx_pairs1",idx_pairs)

    embedding_dims = 5
    W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
    W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
    num_epochs = 10100
    learning_rate = 0.001
    optimizer = torch.optim.Adam([W1, W2], lr=learning_rate)

    for epo in range(num_epochs):
        loss_val = 0
        for data, target in idx_pairs:
            x = Variable(get_input_layer(data, vocabulary_size)).float()
            y_true = Variable(torch.from_numpy(np.array([target])).long())

            y2 = torch.matmul(W2, torch.matmul(W1, x))

            loss = F.nll_loss(F.log_softmax(y2, dim=0).view(1, -1), y_true)
            loss_val += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epo % 10 == 0:
            print(f'Loss at epo {epo}: {loss_val / len(idx_pairs)}')


def main():
    word2v()


main()
