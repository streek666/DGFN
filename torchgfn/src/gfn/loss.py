import torch
import numpy as np

class F_alpha:
    def __init__(self, alpha):
        raise NotImplementedError("F_alpha is not implemented yet")
    
    def __call__(self, log_t):
        raise NotImplementedError("F_alpha is not implemented yet")
        
    def div(self, log_t):
        raise NotImplementedError("F_alpha is not implemented yet")
        
class F_a:
    def __init__(self, alpha):
        raise NotImplementedError("F_a is not implemented yet")
    
    def __call__(self, log_t):
        raise NotImplementedError("F_a is not implemented yet")
        
    def div(self, log_t):
        raise NotImplementedError("F_a is not implemented yet")
    
class V_alpha:
    def __init__(self, alpha):
        self.a = alpha
    
    def __call__(self, log_t, clip):
        return self.a * log_t.abs()
        
class G_alpha:
    def __init__(self, alpha=0):
        self.a = alpha
    
    def __call__(self, log_t, clip=0):
        log_t_clip = log_t.detach()
        if clip > 0:
            log_t_clip = log_t_clip.clip(min=-clip, max=clip)
        if self.a == 0:
            return log_t_clip*log_t
        else:
            return ((self.a*log_t_clip).exp()-1) * log_t/self.a
            
class Annealing_G_alpha:
    def __init__(self, total_steps, func="cosine"):
        assert total_steps > 0
        assert func in ["cosine", "linear"]
        self.t = 0
        self.T = total_steps
        self.func = func
    
    def __call__(self, log_t, clip=0):
        if self.func == 'linear':
            a = 1 - self.t / self.T
        else:
            a = (np.cos(np.pi*self.t/self.T)+1)/2
        log_t_clip = log_t.detach()
        if clip > 0:
            log_t_clip = log_t_clip.clip(min=-clip, max=clip)
        if a == 0:
            return 0.5*log_t_clip*log_t
        else:
            return ((a*log_t_clip).exp()-1) * log_t/a
        
    def step(self):
        self.t += 1;
        
class Annealing_Mix_KL:
    def __init__(self, total_steps, func="cosine"):
        assert total_steps > 0
        assert func in ["cosine", "linear"]
        self.t = 0
        self.T = total_steps
        self.func = func
    
    def __call__(self, log_t, clip=0):
        if self.func == 'linear':
            a = 1 - self.t / self.T
        else:
            a = (np.cos(np.pi*self.t/self.T)+1)/2
        log_t_clip = log_t.detach()
        if clip > 0:
            log_t_clip = log_t_clip.clip(min=-clip, max=clip)
        return (a * (log_t_clip.exp() - 1) + (1-a) * log_t_clip) * log_t
        
    def step(self):
        self.t += 1;
        
class H_alpha:
    def __init__(self, alpha=1):
        assert alpha > 0
        self.a = alpha
        
    def __call__(self, log_t, clip=0):
        if clip > 0:
            log_t_clip = log_t.detach().clip(min=-clip, max=clip)
            return self.a * ((self.a*log_t_clip).exp() - (-self.a*log_t_clip).exp()) * log_t
        return ((self.a*log_t).exp() + (-self.a*log_t).exp() - 2)
    
class G_JSD:
    def __init__(self, alpha=1):
        assert alpha > 0
        self.a = alpha
    
    def __call__(self, log_t, clip=0):
        log_t_clip = log_t.detach()
        if clip > 0:
            log_t_clip = log_t_clip.clip(min=-clip, max=clip)
        return 0.5 * ((1 + (self.a*log_t_clip).exp()).log() - np.log(2)) * log_t
    
class G_SKL:
    def __init__(self, alpha=1):
        assert alpha > 0
        self.a = alpha
    
    def __call__(self, log_t, clip=0):
        log_t_clip = log_t.detach()
        if clip > 0:
            log_t_clip = log_t_clip.clip(min=-clip, max=clip)
        return 0.5 * ((self.a*log_t_clip).exp() + self.a * log_t_clip - 1) * log_t

class G_Loss:
    def __init__(self, g, clip=0):
        self.g = g
        self.clip = clip
        
    def __call__(self, log_pf, log_pb, b_p=None, b_q=None, on_pf=True, reduce='mean'):
        loss = self.g(log_pb-log_pf, clip=self.clip)
        if reduce == 'none':
            return loss
        elif reduce == 'mean':
            return loss.mean()
        elif reduce == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"reduce={reduce} is not supported")
    
class F_Loss:
    def __init__(self, f, b_p=None, b_q=None, clip=5):
        raise NotImplementedError("F_Loss is not implemented yet")
        
    def __call__(self, log_pf, log_pb, b_p=None, b_q=None, on_pf=True, reduce='mean'):
        raise NotImplementedError("F_Loss is not implemented yet")