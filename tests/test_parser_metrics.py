import numpy as np
import pytest
from lvlm_count.counting import (parse_count, ParseError, count_with_retry, LVLM)
from lvlm_count import metrics as M


@pytest.mark.parametrize("t,v", [("[[twelve]]", 12), ("[[ 12 ]]", 12), ("about [[12]] objects", 12),
                                 ("[[twenty-one]]", 21), ("[[1,200]]", 1200), ("[[0]]", 0), ("x [[3]] y [[4]]", 4)])
def test_parse_ok(t, v):
    assert parse_count(t) == v


@pytest.mark.parametrize("t", ["12", "[[many]]", "[[]]", "", "[[ ]]"])
def test_parse_bad(t):
    with pytest.raises(ParseError):
        parse_count(t)


class Flaky(LVLM):
    def __init__(self, outs): self.outs, self.i = outs, 0
    def ask(self, image, prompt):
        o = self.outs[min(self.i, len(self.outs) - 1)]; self.i += 1; return o


def test_retry_then_success():
    r = count_with_retry(Flaky(["no idea", "bad", "[[7]]"]), None, "q")
    assert r.value == 7 and r.failures == 2 and r.attempts == 3


def test_retry_cap():
    r = count_with_retry(Flaky(["nope"]), None, "q", max_retries=10)
    assert r.value is None and r.attempts == 10 and r.failures == 10


def test_metrics_hand_computed():
    gt, p = [10, 20, 30], [12, 20, 25]
    assert M.mae(gt, p) == pytest.approx(7 / 3)
    assert M.rmse(gt, p) == pytest.approx(np.sqrt((4 + 0 + 25) / 3))
    assert M.acc_within(gt, p, 0) == pytest.approx(100 / 3)
    assert M.acc_within(gt, p, 5) == pytest.approx(100.0)
    assert M.bias(gt, p) == pytest.approx(-1.0)


def test_t_ci_and_bootstrap():
    m, h = M.t_ci([1, 2, 3, 4, 5, 6])
    assert m == 3.5 and h == pytest.approx(2.5706 * 0.76376, abs=1e-3)  # t(5,.975)*sem
    gt = np.arange(1, 101); a = gt + 10; b = gt + 1
    r = M.paired_bootstrap(gt, a, b, n_boot=500)
    assert r["diff"] == pytest.approx(9) and r["frac_b_wins"] == 1.0 and r["ci"][0] > 0
    r2 = M.paired_bootstrap(gt, a, a, n_boot=500)
    assert r2["diff"] == 0 and r2["ci"] == (0.0, 0.0)


@pytest.mark.parametrize("t,v", [("[5]", 5), ("5", 5), ("five.", 5), (" 12 ", 12), ("[ 7 ]", 7)])
def test_lenient_ok(t, v):
    assert parse_count(t, lenient=True) == v
    with pytest.raises(ParseError):
        parse_count(t)                      # strict (spec) regex still rejects these


@pytest.mark.parametrize("t", ["I see 5 balloons", "There are about [5] of them, maybe more", "", "many"])
def test_lenient_still_rejects_prose(t):
    with pytest.raises(ParseError):
        parse_count(t, lenient=True)
