import numpy as np
from ..heuristics import generate_random_sample
import pytest

def test_random_sample():
    indices = [0,3,4,6,10,35,20,2,1,11]
    n_samples = 3
    chosen, leftover = generate_random_sample(indices, n_samples)
    assert len(chosen) == n_samples
    assert len(leftover) == len(indices)-n_samples
    
def test_kmeans():
    rng = np.random.default_rng()
    indices = rng.choice(50000,30000, replace=False)
    n_samples = 1000
    model = None
    x = rng.random((50000,100,100,3))
    y = rng.integers(10, size=(50000,))
    data = zip(x,y)