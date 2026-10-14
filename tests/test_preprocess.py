import numpy as np

from aegis.data.preprocess import TokenCache, build_inputs, normalize

CODE = "int f(int a) {\n  // TODO: check bounds\n  return a +   1; /* fine */\n}\n"


def test_normalize_modes():
    assert normalize(CODE, "raw") == CODE
    assert "\n  " not in normalize(CODE, "collapse")
    assert normalize(CODE, "single_line") == "int f(int a) { // TODO: check bounds return a + 1; /* fine */ }"
    cf = normalize(CODE, "comments_first")
    assert cf.startswith("// TODO: check bounds /* fine */")
    assert "return a + 1;" in cf


def test_build_inputs_truncation():
    ids = np.arange(10, 30, dtype=np.int32)
    cache = TokenCache(ids, np.array([0, 20]))
    head = build_inputs(cache, cls_id=0, sep_id=2, max_length=8)[0]
    assert head.tolist() == [0, 10, 11, 12, 13, 14, 15, 2]
    ht = build_inputs(cache, cls_id=0, sep_id=2, max_length=8, truncation="head_tail")[0]
    assert ht[0] == 0 and ht[-1] == 2 and len(ht) == 8 and ht[-2] == 29


def test_build_inputs_no_truncation_when_short():
    cache = TokenCache(np.array([5, 6], dtype=np.int32), np.array([0, 2]))
    assert build_inputs(cache, 0, 2, 16)[0].tolist() == [0, 5, 6, 2]
