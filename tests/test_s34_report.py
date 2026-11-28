import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from b_s34_report import auc, auc_interval, detection, validate_case


def test_auc_ties_and_missing_class():
    assert auc([0,0],[1,1]) == 1
    assert auc([1,1],[0,0]) == 0
    assert auc([1,1],[1,1]) == .5
    assert auc([],[1]) is None
    with pytest.raises(ValueError):
        auc([1],[np.nan])


def test_bootstrap_clusters_and_undefined_values():
    pairs = [(np.array([0.,0.]),np.array([1.,2.]))]*3
    assert auc_interval(pairs,resamples=30)['ci95'] == [1,1]
    assert auc_interval([],resamples=30)['estimate'] is None


def test_unexposed_not_counted_as_miss():
    record = {'t':20,'valid':True,'normalized':0,'exceeds_p99':False}
    clean = {0:{'records':[record],'exposures':[]}}
    disturbed = {0:{'records':[record],'exposures':[]}}
    result = detection(clean,disturbed)['post_onset']
    assert result['unexposed_episodes'] == 1 and result['missed_episodes'] == 0
    assert result['false_negative_rate'] is None


def test_detection_step_and_episode_denominators_and_conditional_delay():
    def record(t, hit):
        return {'t':t, 'valid':True, 'normalized':int(hit), 'exceeds_p99':hit}
    clean = {0:{'records':[record(20, True),record(21, True)],'exposures':[]},
             1:{'records':[record(20, False),record(21, False)],'exposures':[]}}
    disturbed = {0:{'records':[record(20, False),record(21, True)],'exposures':[20]},
                 1:{'records':[record(20, False),record(21, False)],'exposures':[20]}}
    result = detection(clean,disturbed)['post_onset']
    assert result['false_positive_steps'] == 2 and result['clean_valid_steps'] == 4
    assert result['false_positive_episodes'] == 1 and result['clean_evaluable_episodes'] == 2
    assert result['episode_false_positive_rate'] == .5
    assert result['missed_episodes'] == 1 and result['mean_detection_delay'] == 1
    assert 'detected' in result['mean_detection_delay_population']


def case_fixture():
    from copy import deepcopy
    config = {'environments':['env'], 'plugins':[{'key':'p','plugin':'fault','config':{}}]}
    runs = [{'environment':'env','arm':a,'run_id':'run_'+a} for a in ('A0','A1')]
    case = {'experiment_id':'rt_test','config':config,'runs':runs}
    experiment = {'id':'rt_test','config':deepcopy(config),'runs':deepcopy(runs),
                  'arms':[{'id':'A0'},{'id':'A1'}]}
    experiment['config']['plugins'][0]['entrypoint'] = None
    return case,experiment


def test_case_optional_null_normalization():
    validate_case(*case_fixture())


@pytest.mark.parametrize('mutation', ['missing','duplicate','wrong_run','wrong_config','extra_env','missing_expected'])
def test_case_rejects_incomplete_or_mismatched_manifest(mutation):
    case,experiment = case_fixture()
    if mutation == 'missing': case['runs'].pop()
    elif mutation == 'duplicate': case['runs'].append(case['runs'][0])
    elif mutation == 'wrong_run': case['runs'][1]['run_id'] = 'run_other'
    elif mutation == 'wrong_config': case['config']['plugins'][0]['config']['c'] = 99
    elif mutation == 'extra_env': case['config']['environments'].append('other')
    elif mutation == 'missing_expected': experiment['runs'].pop()
    with pytest.raises(ValueError):
        validate_case(case,experiment)
