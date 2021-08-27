import numpy as np
import math
import gzip
from collections import defaultdict

tag_counts = defaultdict(int, {'--SIN_UNK--': 4274862, '--VERB--': 1135394, '--UNK_UNK--': 2818587, '--s--': 404515, '--SIN_THI--': 177387, '--PLU_UNK--': 1288778, '--PLU_THI--': 70723, '--UNK_THI--': 1954, '--SIN_FIR--': 37748, '--SIN_SEC--': 21574, '--PLU_FIR--': 18205, '--PLU_SEC--': 7550, '--UNK_FIR--': 70, '--UNK_SEC--': 234})
states = sorted(tag_counts.keys())
vocab = {}

with gzip.open('/vocab/vocab.txt.gz','rt') as f:
    for line in f:
        word, number = line.split()
        vocab[word] = number

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

best_probs, best_paths = initialize(states, tag_counts, A, B, sentence, vocab)