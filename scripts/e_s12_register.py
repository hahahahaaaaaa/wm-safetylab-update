"""One-time source registration for S1/S2; preserves existing S3/S4 entries."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def update(path, group, entries):
    path=ROOT/path
    data=json.loads(path.read_text(encoding='utf-8'))
    data[group].update(entries)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def entry(module, cls, stage, role, requires, config, description):
    return dict(entrypoint=f'wms.plugins.cards.{module}:{cls}',stage=stage,role=role,requires=requires,
                config=config,description=description,**({'side':'model'} if stage=='S2' else {}))
a={
 'a-obs-corruption':entry('a_obs_corruption','ObsCorruption','S1','fault',['rgb'],{'kind':'noise','severity':3},'#1 图像观测损坏，五级剂量，0 为中性'),
 'a-camera-jitter':entry('a_camera_jitter','CameraJitter','S1','fault',['rgb'],{'shift':.05,'rotate':5,'scale':.05,'mode':'persistent'},'#2 图像平面仿射抖动：持续或每步'),
 'a-patch-attack':entry('a_patch_attack','FixedPatch','S1','attack',['rgb'],{'area':.1},'#3 固定遮挡块；外部黑盒搜索，训练/留出分离'),
 'a-frame-guard':entry('a_frame_guard','FrameGuard','S1','defense',['rgb'],{'tau':None,'max_replacements':5},'#4 异常帧替换；真实实验须提供干净标定阈值'),
 'a-input-monitor':entry('a_input_monitor','InputMahalanobis','S1','monitor',['rgb'],{'mean':[0]*7,'std':[1]*7,'threshold':None},'#5 图像输入异常监控；默认仅 Toy 接口检查，真实实验必须拟合 stats'),
 'a-input-monitor-vector':entry('a_input_monitor','VectorInputMahalanobis','S1','monitor',['vector'],{'mean':[0]*7,'std':[1]*7,'threshold':None},'#5 向量输入异常监控；默认 Toy 7 维，真实实验必须拟合 stats')}
b={
 'b-memory-fault':entry('b_memory_fault','MemoryFault','S2','fault',['state'],{'mode':'wipe','start':10},'#6 记忆清空/过时/标定噪声；TD-MPC2 latent 噪声为状态估计损坏'),
 'b-write-gate':entry('b_write_gate','WriteGate','S2','defense',['state'],{'tau':None,'recent':16},'#7 记忆写入门控：仅支持 read + written 模型，需干净标定阈值')}
update('native/plugins_a/recipes.json','plugins',a)
update('native/plugins_b/recipes.json','plugins',b)
for path, models in [('native/toy/recipes.json',['toy-centroid','toy-centroid-memory']),('native/memoryvla/recipes.json',['memoryvla-bridge','memoryvla-bridge-nomem'])]:
    file=ROOT/path
    data=json.loads(file.read_text())
    for model in models:
        data['models'][model]['state_phases']=['read','written']
    file.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
