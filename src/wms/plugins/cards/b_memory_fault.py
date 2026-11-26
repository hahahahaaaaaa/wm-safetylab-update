"""Card #6: independent read snapshots, explicit calibrated state noise.

TD-MPC2 wipe clears only plan_mean; latent noise corrupts state estimation.
MemoryVLA wipe restores the first *available* read, already containing a frame.
"""
import copy
from collections import deque
from ..base import Plugin
from .a_common import active, window, nonnegative
from .a_input_monitor import load_stats


def array_of(value):
    import numpy as np
    if hasattr(value,'detach'):
        return value.detach().float().cpu().numpy().astype(float)
    return np.asarray(value, dtype=float)


def noisy_tree(value, rng, c, std):
    """Preserve tensors (including bf16), devices, integer timestamps and shapes."""
    import numpy as np
    if isinstance(value,dict):
        return {k:noisy_tree(v,rng,c,std) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return type(value)(noisy_tree(v,rng,c,std) for v in value)
    tensor = hasattr(value,'detach')
    floating = value.dtype.is_floating_point if tensor else (
        np.issubdtype(value.dtype,np.floating) if isinstance(value,np.ndarray) else isinstance(value,(float,np.floating)))
    if not floating:
        return copy.deepcopy(value)
    a = array_of(value)
    s = np.asarray(std,dtype=float)
    if s.ndim and s.shape != a.shape and not (a.ndim and s.shape == (a.shape[-1],)):
        raise ValueError('noise std must be scalar or match the leaf/last dimension')
    if not np.isfinite(a).all():
        raise ValueError('state contains nonfinite floating values')
    result = a + rng.normal(size=a.shape)*c*s
    if not np.isfinite(result).all():
        raise ValueError('state noise overflow')
    if tensor:
        import torch
        out = torch.as_tensor(result,dtype=value.dtype,device=value.device)
        if not torch.isfinite(out).all():
            raise ValueError('state noise overflows tensor dtype')
        return out
    out = np.asarray(result,dtype=value.dtype if isinstance(value,np.ndarray) else type(value))
    if not np.isfinite(out).all():
        raise ValueError('state noise overflows dtype')
    return out if isinstance(value,np.ndarray) else type(value)(out)


class MemoryFault(Plugin):
    stage, role, side, requires = 'S2','fault','model',('state',)
    neutral = {'mode':'none'}

    def __init__(self, config):
        super().__init__(config)
        self.mode = self.config.get('mode','wipe')
        if self.mode not in ('none','wipe','stale','noise'):
            raise ValueError('mode must be none/wipe/stale/noise')
        self.k = self.config.get('k', 3)
        if type(self.k) is not int or self.k < 1:
            raise ValueError('k must be positive integer')
        self.c = nonnegative(self.config.get('c', .1),'c')
        self.steps = self.config.get('steps')
        if self.steps is not None and (not isinstance(self.steps,list) or any(type(x) is not int or x<0 for x in self.steps)):
            raise ValueError('steps must be nonnegative integer list')
        window(self.config)

    def reset(self, episode, ctx):
        self.first, self.history = None, deque(maxlen=self.k+1)
        self.std = self.config.get('std')
        if self.config.get('stats'):
            field = load_stats(self.config['stats'])[self.config.get('stats_field','latent')]
            self.std = field['std'] if 'std' in field else {k:v['std'] for k,v in field.items()}
        if self.mode == 'noise' and self.c:
            import numpy as np
            for value in (self.std.values() if isinstance(self.std,dict) else [self.std]):
                a = np.asarray(value,dtype=float)
                if not a.size or not np.isfinite(a).all() or (a<0).any():
                    raise ValueError('noise requires explicit finite nonnegative calibrated std')

    def on_state(self, state, ctx):
        if ctx.phase != 'read' or self.mode == 'none':
            return state
        if self.first is None:
            self.first = copy.deepcopy(state)
        self.history.append((ctx.t,copy.deepcopy(state)))
        if not active(self.config,ctx.t) or (self.steps is not None and ctx.t not in self.steps):
            return state
        if self.mode == 'wipe':
            out = {**state,'plan_mean':None} if 'plan_mean' in state else copy.deepcopy(self.first)
        elif self.mode == 'stale':
            eligible = [s for t,s in self.history if t<=ctx.t-self.k]
            if not eligible:
                ctx.log(injected=False, mode=self.mode, reason='insufficient_history')
                return state
            out = copy.deepcopy(eligible[-1])
        else:
            if self.c == 0:
                return state
            # Keep warm-start control untouched when the state offers a latent.
            keys = self.config.get('keys') or (['latent'] if 'latent' in state else [k for k in state if k not in ('timestep','age')])
            if any(k not in state for k in keys):
                raise ValueError('configured noise key is absent')
            out = {**state, **{k:noisy_tree(state[k],ctx.rng,self.c,self.std[k] if isinstance(self.std,dict) else self.std) for k in keys}}
        for key in set(self.config.get('keep_keys',[])) | {'timestep'}:
            if key in state:
                out[key] = state[key]
        ctx.log(injected=True, mode=self.mode, c=self.c if self.mode=='noise' else None, k=self.k if self.mode=='stale' else None)
        return out
