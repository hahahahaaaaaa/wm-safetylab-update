"""Card #7: reject changed/new memory entries, never reject first-frame bootstrap."""
import copy
from ..base import Plugin
from .a_common import nonnegative
from .b_memory_fault import array_of


def write_distance(before, after, recent=16):
    import numpy as np
    if 'cube' in after and 'goal' in after:
        distances = [float(np.linalg.norm(array_of(after[k])-array_of(before[k])))
                     for k in ('cube','goal') if before.get(k) is not None and after.get(k) is not None]
        return max(distances,default=0.0)
    if not {'cognitive','perceptual'} <= set(after):
        raise ValueError('write gate needs memory banks or position memory with written phase')
    distances = []
    for bank in ('cognitive','perceptual'):
        old = before[bank]
        vectors = [array_of(entry[1]).reshape(-1) for entry in old[-recent:]]
        for stamp,value in after[bank]:
            vector = array_of(value).reshape(-1)
            if not np.isfinite(vector).all():
                raise ValueError('nonfinite memory entry')
            unchanged = any(stamp==t and np.array_equal(vector,array_of(v).reshape(-1)) for t,v in old)
            if unchanged or not vectors:
                continue
            ds=[]
            for reference in vectors:
                if reference.shape != vector.shape or not np.isfinite(reference).all():
                    raise ValueError('memory vector dimensions/finite values differ')
                denom = float(np.linalg.norm(vector)*np.linalg.norm(reference))
                ds.append(0.0 if np.array_equal(vector,reference) else
                          (1.0 if denom==0 else float(1-np.clip(vector@reference/denom,-1,1))))
            distances.append(min(ds))
    return max(distances,default=0.0)


class WriteGate(Plugin):
    stage, role, side, requires = 'S2','defense','model',('state',)
    required_state_phases = ('read','written')
    neutral = {'tau':None}

    def __init__(self, config):
        super().__init__(config)
        self.tau = self.config.get('tau')
        if self.tau is not None:
            self.tau = nonnegative(self.tau,'tau')
        self.recent = self.config.get('recent',16)
        if type(self.recent) is not int or self.recent<1:
            raise ValueError('recent must be positive integer')

    def reset(self, episode, ctx):
        self.snapshot, self.read_t = None, None
        self.writes, self.rejected = 0, 0
        self.moments = {}

    def on_state(self, state, ctx):
        if ctx.phase == 'read':
            self.snapshot, self.read_t = copy.deepcopy(state), ctx.t
            if self.config.get('collect_moments'):
                import numpy as np
                for key in ('cognitive','perceptual'):
                    for stamp,value in state.get(key,[]):
                        a=array_of(value)
                        a=a.reshape(-1,a.shape[-1])
                        m=self.moments.setdefault(key,{'n':0,'sum':np.zeros(a.shape[-1]),'sumsq':np.zeros(a.shape[-1])})
                        m['n']+=len(a);m['sum']+=a.sum(0);m['sumsq']+=(a*a).sum(0)
            return state
        if ctx.phase != 'written':
            return state
        self.writes += 1
        if self.snapshot is None or self.read_t != ctx.t:
            ctx.log(distance=None, rejected=False, bootstrap=True)
            return state
        d = write_distance(self.snapshot,state,self.recent)
        reject = self.tau is not None and d>self.tau
        ctx.log(distance=d, rejected=reject, threshold=self.tau, bootstrap=False)
        if not reject:
            return state
        self.rejected += 1
        out = copy.deepcopy(self.snapshot)
        if 'timestep' in state:
            out['timestep'] = state['timestep']
        ctx.alarm('write_rejected',score=d)
        return out

    def end_episode(self, ctx):
        if not self.writes:
            raise ValueError('write gate received no written events: incompatible model')
        return {'writes':self.writes,'rejected':self.rejected,'rejection_rate':self.rejected/self.writes,
                'memory_moments':{k:{'n':v['n'],'sum':v['sum'].tolist(),'sumsq':v['sumsq'].tolist()} for k,v in self.moments.items()}}
