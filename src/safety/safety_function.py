import numpy as np

" it calcualtes l(x) part of our eqn i.e immediate safety"

def ell_from_distance(min_dist, d_safe):
    ell = min_dist - d_safe
    # return np.clip(min_dist / d_safe , -1.0, 1.0) 
    return ell 