import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from b_s34_report import auc, auc_interval, detection


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
