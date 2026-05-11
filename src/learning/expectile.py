import numpy as np
import torch

def expectile_loss(pred, target, tau= 0.99):
    diff = target - pred
    w = torch.where(diff>0 , tau, 1-tau)
    return (w*diff**2).mean() 



# def expectile_vectorized(values, tau=0.8, num_iters=50, lr=0.1):
#     values = np.array(values)
#     v = np.mean(values) # Initial guess
    
#     for _ in range(num_iters):
#         diff = values - v
#         # Vectorized weight calculation
#         weights = np.where(diff >= 0, tau, 1 - tau)
        
#         # Vectorized gradient calculation
#         grad = np.mean(-2 * weights * diff)
        
#         # Update
#         v -= lr * grad
        
#     return v



