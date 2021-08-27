import numpy as np
import math
import gzip
from collections import defaultdict

tag_counts = defaultdict(int, {'--SIN_UNK--': 4274862, '--VERB--': 1135394, '--UNK_UNK--': 2818587, '--s--': 404515, '--SIN_THI--': 177387, '--PLU_UNK--': 1288778, '--PLU_THI--': 70723, '--UNK_THI--': 1954, '--SIN_FIR--': 37748, '--SIN_SEC--': 21574, '--PLU_FIR--': 18205, '--PLU_SEC--': 7550, '--UNK_FIR--': 70, '--UNK_SEC--': 234})
states = sorted(tag_counts.keys())
vocab = {}

A = np.loadtxt("/model/transition_matrix.txt.gz").reshape(14, 14)
B = np.loadtxt("/model/emission_matrix.txt.gz").reshape(14, 183991)

with gzip.open('/vocab/vocab.txt.gz','rt') as f:
    for line in f:
        word, number = line.split()
        vocab[word] = number

#insert where user enters sentence. ^ should be loaded at all times. ↓ should run when user enters in sentence.
sentence = ""

def initialize(states, tag_counts, A, B, sentence, vocab):
    num_tags = len(tag_counts)
    
    best_probs = np.zeros((num_tags, len(sentence)))
    
    best_paths = np.zeros((num_tags, len(sentence)), dtype=int)
    
    s_idx = states.index("--s--")
    
    for i in range(num_tags):
        
        if A[s_idx, i] == 0:
            
            best_probs[i,0] = float('-inf')
        
        else:
            
            best_probs[i,0] = math.log(A[s_idx, i]) + math.log(B[i, vocab[sentence[0]]])
                        
    return best_probs, best_paths

def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab):

    num_tags = best_probs.shape[0]
    
    for i in range(1, len(test_corpus)): 
            
        for j in range(num_tags):
            
            best_prob_i = float('-inf')
            
            best_path_i = None

            for k in range(num_tags): 
            
                prob = best_probs[k, i-1] + math.log(A[k, j]) + math.log(B[j, vocab[test_corpus[i]]])

                if prob > best_prob_i: 
                    
                    best_prob_i = prob

                    best_path_i = k

            best_probs[j,i] = best_prob_i
            
            best_paths[j,i] = best_path_i

    return best_probs, best_paths

def viterbi_backward(best_probs, best_paths, states):

    m = best_paths.shape[1] 
    
    z = [None] * m
    
    num_tags = best_probs.shape[0]
    
    best_prob_for_last_word = float('-inf')
    
    pred = [None] * m
    

    for k in range(num_tags):

        if best_probs[k, -1] > best_prob_for_last_word: 
            
            best_prob_for_last_word = best_probs[k, -1]
    
            z[m - 1] = k
            
    pred[m - 1] = states[k]
    
    for i in range(m-1, 0, -1): 
        
        pos_tag_for_word_i = best_paths[z[i], i]
        
        z[i - 1] = pos_tag_for_word_i
        
        pred[i - 1] = states[pos_tag_for_word_i] 
        
    return pred

best_probs, best_paths = initialize(states, tag_counts, A, B, sentence, vocab)
best_probs, best_paths = viterbi_forward(A, B, sentence, best_probs, best_paths, vocab)
prediction = viterbi_backward(best_probs, best_paths, states)
