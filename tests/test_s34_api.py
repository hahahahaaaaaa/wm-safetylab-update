from wms.core.registry import Registry
from wms.redteam.catalog import load_catalog
from wms.redteam.schema import ExperimentConfig
from wms.redteam.validation import validate_config


def test_recipe_inherited_seeds_are_overridden():
    registry = Registry()
    registry.items['recipes']['toy-episode-occlusion']['params']['seeds'] = [999]
    config = ExperimentConfig(name='seed regression',model='toy-planner',environments=['toy-pick-vector'],
                              seeds={'start':2000,'episodes':2})
    result = validate_config(config,load_catalog(registry),registry)
    assert result.valid
    resolved = registry.resolve(result.recipes[0])
    assert resolved['params']['seeds'] == [2000,2001]


def test_effective_thresholds_are_per_environment():
    registry = Registry()
    catalog = load_catalog(registry)
    envs = {e.id:e for e in catalog.environments}
    assert envs['maniskill-pushcube-state-independent'].hazard_thresholds['fast_speed'] == 1.02
    assert envs['simpler-bridge-carrot'].hazard_thresholds['fast_speed'] == .3
    assert all(p in {p.id for p in catalog.plugins} for p in ('b-score-misspec','b-constrained-rerank','b-prediction-noise','b-residual-monitor'))
