import copy
from types import SimpleNamespace
import numpy as np
import pytest
from wms.perturb import brightness, contrast, color_bias, affine
from wms.plugins.cards.a_obs_corruption import ObsCorruption, VectorObsNoise
from wms.plugins.cards.a_camera_jitter import CameraJitter
from wms.plugins.cards.a_patch_attack import FixedPatch
from wms.plugins.cards.a_frame_guard import FrameGuard
from wms.plugins.cards.a_input_monitor import InputMahalanobis, VectorInputMahalanobis
from wms.plugins.cards.b_memory_fault import MemoryFault, noisy_tree
from wms.plugins.cards.b_write_gate import WriteGate, write_distance

def ctx(t=0,phase='read'):
    events,alarms=[],[]
    return SimpleNamespace(t=t,phase=phase,rng=np.random.default_rng(42),
       log=lambda **kw:events.append(kw),alarm=lambda reason,**kw:alarms.append((reason,kw)),events=events,alarms=alarms)

def obs(value=50):
    return {'rgb':{'cam':np.full((32,32,3),value,np.uint8)},'proprio':[0.,0.]}

@pytest.mark.parametrize('fn',[brightness,contrast,color_bias])
def test_image_transforms(fn):
    image=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
    before=image.copy()
    assert fn(image,0) is image
    diffs=[]
    for severity in range(1,6):
        out=fn(image,severity)
        assert out.dtype==np.uint8 and out.shape==image.shape
        diffs.append(np.abs(out.astype(float)-image).mean())
    assert all(b>=a for a,b in zip(diffs,diffs[1:]))
    np.testing.assert_array_equal(image,before)

def test_affine_inverse_integer_center():
    image=np.arange(20*20*3,dtype=np.uint8).reshape(20,20,3)
    assert affine(image) is image
    out=affine(affine(image,dx=3),dx=-3)
    np.testing.assert_array_equal(out[:,3:-3],image[:,3:-3])

@pytest.mark.parametrize('kind',['noise','brightness','contrast','color_bias','blur'])
def test_corruptions_windows_rng_and_neutral(kind):
    x=obs(); p=ObsCorruption({'kind':kind,'severity':3,'start':2,'end':3})
    assert p.on_observation(x,ctx(1)) is x
    assert p.on_observation(x,ctx(3)) is x
    a=p.on_observation(x,ctx(2)); b=p.on_observation(x,ctx(2))
    np.testing.assert_array_equal(a['rgb']['cam'],b['rgb']['cam'])
    assert x['rgb']['cam'].min()==50
    assert ObsCorruption({'severity':0}).on_observation(x,ctx()) is x

def test_jitter_persistent_and_reset():
    p=CameraJitter({}); c=ctx(); p.reset(0,c)
    p.on_observation(obs(),c); first=dict(p.transforms)
    c.t=1; p.on_observation(obs(),c); assert p.transforms==first
    p.reset(1,c); assert p.transforms=={}
    p=CameraJitter({'shift':0,'rotate':0,'scale':0}); x=obs()
    assert p.on_observation(x,c) is x

def test_patch_area_and_no_mutation():
    x=obs(); c=ctx(); p=FixedPatch({'area':.1})
    out=p.on_observation(x,c)
    assert (out['rgb']['cam'][...,0]==0).sum()/1024 <= .1
    assert x['rgb']['cam'].min()==50 and c.events[0]['actual_area']<=.1
    with pytest.raises(ValueError): FixedPatch({'area':.01,'box':[0,0,1,1]})

def test_patch_pixel_budget_equal_across_locations_and_threat():
    from wms.plugins.base import threat_card
    side=.1**.5
    counts=[]
    for x in (.0,.04,.13,.57):
        out=FixedPatch({'area':.1,'box':[x,.1,x+side,.1+side]}).on_observation(obs(),ctx())
        counts.append(int((out['rgb']['cam'][...,0]==0).sum()))
    assert len(set(counts))==1
    assert threat_card(FixedPatch,{'area':.2})['budget']['area']==.2

def test_frame_guard_spike_and_forced_release():
    p=FrameGuard({'tau':10}); c=ctx(); p.reset(0,c)
    x=obs(); assert p.on_observation(x,c) is x
    c.t=1; spike=obs(200); assert p.on_observation(spike,c)['rgb']['cam'].max()==50
    c.t=2; assert p.on_observation(x,c) is x
    assert len(c.alarms)==1
    for i in range(5): assert p.on_observation(spike,c)['rgb']['cam'].max()==50
    assert p.on_observation(spike,c) is spike
    p.reset(1,c); assert p.on_observation(spike,c) is spike

def test_monitor_readonly_and_mahalanobis():
    p=VectorInputMahalanobis({'mean':[0,0],'std':[2,1],'threshold':2})
    c=ctx(); p.reset(0,c); x=[4.,0.]
    assert p.on_observation(x,c) is x and c.events[-1]['score']==2
    assert not c.alarms

def test_image_monitor_features_reset():
    p=InputMahalanobis({'mean':[0]*7,'std':[1]*7}); c=ctx(); p.reset(0,c)
    x=obs(); assert p.on_observation(x,c) is x
    assert c.events[-1]['features'][-1]==0
    p.on_observation(obs(60),c); assert c.events[-1]['features'][-1]==10
    p.reset(1,c); p.on_observation(obs(60),c); assert c.events[-1]['features'][-1]==0

def test_memory_stale_snapshot_timestamps():
    p=MemoryFault({'mode':'stale','k':2}); c=ctx(); p.reset(0,c)
    states=[]
    for t in range(4):
        c.t=t; s={'timestep':t,'cognitive':[(t,np.array([float(t)]))],'perceptual':[]}; states.append(s)
        out=p.on_state(s,c)
        if t>=2: assert out['cognitive'][0][1][0]==t-2 and out['timestep']==t
    states[0]['cognitive'][0][1][0]=900
    assert p.first['cognitive'][0][1][0]==0
    p.reset(1,c); assert p.first is None

def test_wipe_td_preserves_current_latent():
    p=MemoryFault({'mode':'wipe','steps':[10]}); c=ctx(); p.reset(0,c)
    s={'latent':np.ones(3,np.float32),'plan_mean':np.ones((2,3))}
    assert p.on_state(s,c) is s
    c.t=10; out=p.on_state(s,c)
    assert out['plan_mean'] is None and out['latent'] is s['latent']

def test_memory_noise_keys_dtype_and_neutral():
    p=MemoryFault({'mode':'noise','std':[1,2,3],'c':1}); c=ctx(); p.reset(0,c)
    s={'latent':np.zeros(3,np.float32),'plan_mean':np.ones((2,3))}
    out=p.on_state(s,c)
    assert out['latent'].dtype==np.float32 and out['plan_mean'] is s['plan_mean']
    assert np.all(s['latent']==0) and not np.all(out['latent']==0)
    c.phase='written'; assert p.on_state(s,c) is s

def test_memory_bfloat16_and_no_alias():
    torch=pytest.importorskip('torch')
    x=torch.ones((1,4),dtype=torch.bfloat16)
    out=noisy_tree([(2,x)],np.random.default_rng(1),.3,.5)
    assert out[0][0]==2 and out[0][1].dtype==x.dtype and out[0][1].device==x.device
    assert torch.all(x==1)

def test_write_gate_toy_jump_and_bootstrap():
    p=WriteGate({'tau':.1}); c=ctx(); p.reset(0,c)
    old={'cube':[.1,.2],'goal':[.8,.8],'age':{'cube':0,'goal':0}}
    p.on_state(old,c)
    c.phase='written'; new={**old,'cube':[.4,.2]}
    out=p.on_state(new,c)
    assert out==old and len(c.alarms)==1 and new['cube']==[.4,.2]
    c.t=1; assert p.on_state(new,c) is new # missing read must not reuse prior step
    assert p.end_episode(c)['writes']==2

def test_write_gate_rewritten_timestamp_and_timestep():
    a={'timestep':3,'cognitive':[(1,np.array([1.,0.]))],'perceptual':[]}
    b={'timestep':4,'cognitive':[(1,np.array([0.,1.]))],'perceptual':[]}
    assert write_distance(a,b)==1
    p=WriteGate({'tau':.5}); c=ctx(); p.reset(0,c); p.on_state(a,c)
    c.phase='written'; out=p.on_state(b,c)
    assert out['timestep']==4; np.testing.assert_array_equal(out['cognitive'][0][1],[1,0])
    p=WriteGate({'tau':None}); c=ctx(); p.reset(0,c); p.on_state(a,c)
    c.phase='written'; assert p.on_state(b,c) is b and c.events[-1]['distance']==1

@pytest.mark.parametrize('cls,cfg',[(ObsCorruption,{'severity':6}),(CameraJitter,{'scale':1}),
    (FixedPatch,{'area':float('nan')}),(FrameGuard,{'tau':-1}),(MemoryFault,{'k':0}),(WriteGate,{'recent':0})])
def test_bad_configs(cls,cfg):
    with pytest.raises(ValueError): cls(cfg)

def test_write_gate_rejects_readonly_model_before_run():
    from wms.plugins.resolve import check
    spec={'id':'gate','entrypoint':'wms.plugins.cards.b_write_gate:WriteGate','config':{}}
    config={'params':{'plugins':[spec]},'model':{'probes':['state']},'environment':{'capabilities':[]}}
    assert any('state phases' in e for e in check(config))
    config['model']['state_phases']=['read','written']; assert check(config)==[]
