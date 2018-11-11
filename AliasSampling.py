# This edited GitHubGist is forked from Jeremy Howard
# Original source: gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087

import numpy.random as npr, numpy as np
from numba import jit

@jit(nopython=True)
def sample(n, q, J, r1, r2):
    res = np.zeros(n, dtype=np.int32)
    lj = len(J)
    for i in range(n):
        kk = int(np.floor(r1[i]*lj))
        if r2[i] < q[kk]: res[i] = kk
        else: res[i] = J[kk]
    return res

class AliasSample():
    '''
    Randomly samples ~5x faster than np.random.choice() and np.searchsorted()

    ## Example of how to use AliasSample:
    # Define weights for sampling
    popSize = 30000
    sampSize = 5000
    prs = npr.random(popSize)
    prs /= prs.sum()

    # Draw a sample
    as = AliasSample(prs)  # returns an iterator
    sample = as.draw_n(sampSize)  # returns a sample
    '''
    def __init__(self, probs):
        self.K=K= len(probs)
        self.q=q= np.zeros(K)
        self.J=J= np.zeros(K, dtype=np.int)

        smaller,larger  = [],[]
        for kk, prob in enumerate(probs):
            q[kk] = K*prob
            if q[kk] < 1.0: smaller.append(kk)
            else: larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small,large = smaller.pop(),larger.pop()
            J[small] = large
            q[large] = q[large] - (1.0 - q[small])
            if q[large] < 1.0: smaller.append(large)
            else: larger.append(large)

    def draw_one(self):
        K,q,J = self.K,self.q,self.J
        kk = int(np.floor(npr.rand()*len(J)))
        if npr.rand() < q[kk]: return kk
        else: return J[kk]

    def draw_n(self, n):
        r1,r2 = npr.rand(n),npr.rand(n)
        return sample(n,self.q,self.J,r1,r2)
