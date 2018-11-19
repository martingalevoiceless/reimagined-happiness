from hypothesis import given
import numpy
import msgpack
import hashlib
from hypothesis.strategies import floats, lists, builds, integers, tuples, sampled_from, composite, shared
from web import state
import functools
finite_floats = functools.partial(floats, allow_nan=False, allow_infinity=False)
kind_floats = lambda *a, **kw: finite_floats(*a, **kw).map(lambda x: round(x, 4))

class FakeFiles:
    def __init__(self, draw, items):
        self.draw = draw
        self.af = [{"hash": hashlib.sha256(str(i).encode("utf-8")).hexdigest()[:10], "value": j[0], "type": j[1]} for i,j in enumerate(items)]
        self.bh = {x["hash"]: x for x in self.af}
        self.comparisons = {}
    def get_all_images(self):
        return self.af, self.bh
    def compare(self, h1, h2):
        if (h1,h2) not in self.comparisons:
            self.comparisons[(h1, h2)] = self.draw(kind_floats(min_value=0, max_value=1), label="saved comparison")
        h1v = self.bh[h1]
        h2v = self.bh[h2]
        if h1v["type"] is not None and h2v["type"] is not None and h1v["type"] != h2v["type"]:
            return None, "incomparable"
        r = (self.draw(kind_floats(min_value=0, max_value=0.5), label="comparison") + h1v["value"]) - (self.comparisons[(h1,h2)] + h2v["value"])
        if abs(r) < 0.1:
            return None, "too_close"
        return r > 0, "good"
types = [None, "a", "b", "c"]
@composite
def fakefiles_(draw, *, files=None, minfiles=0, maxfiles=40):
    if files is None:
        files = lists(tuples(kind_floats(min_value=-5, max_value=5), sampled_from(types)), max_size=maxfiles, min_size=minfiles)
    info = draw(files, label="file values and types")
    res = FakeFiles(draw, info)
    return res
fakefiles = lambda *a, **kw: shared(fakefiles_(*a, **kw), key="files")

@composite
def make_stats_(draw, *, minholdout=0, **kwargs):
    # TODO: reuse ratio? more realistic distribution
    stats = state.Stats()
    kwargs["minfiles"] = kwargs.get("minfiles", 0) + minholdout
    kwargs["maxfiles"] = kwargs.get("minfiles", 40) + minholdout
    files = draw(fakefiles(**kwargs))
    comparisons_per_file = draw(integers(min_value=0, max_value=7), label="comparisons per file")
    assert len(files.af) - minholdout >= 0
    compared_files = draw(integers(min_value=0, max_value=len(files.af)-minholdout), label="files to use")
    stats._test_compared_pool = files.af[:compared_files]
    stats._test_holdout_pool = files.af[compared_files:]
    stats._test_files = files

    if compared_files < 2:
        return stats
    for x in range(comparisons_per_file * compared_files):
        h1i = draw(integers(min_value=0, max_value=compared_files-1), label="first item")
        h2i = draw(integers(min_value=0, max_value=compared_files-2), label="second item")
        h1 = stats._test_compared_pool[h1i]
        h2 = (stats._test_compared_pool[:h1i] + stats._test_compared_pool[h1i+1:])[h2i]
        first_won, kind = files.compare(h1["hash"], h2["hash"])
        pair = state.as_pair(h1, h2)
        if kind == "too_close":
            stats.too_close[pair] = True
        elif kind == "incomparable":
            stats.incomparable_pairs[pair] = stats.incomparable_pairs.get(pair, 0) + 1
        else:
            winning, losing = (h1, h2) if first_won else (h2, h1)
            stats.record_win(winning, losing)
            if losing["value"] < 0.1:
                stats.dislike[losing["hash"]] = True
    return stats
make_stats = lambda *a, **kw: shared(make_stats_(*a, **kw), key="stats")

@composite
def precise_model(draw, **kwargs):
    stats = draw(make_stats(**kwargs))
    model = state.Model(
        all_items=[x["hash"] for x in stats._test_compared_pool],
        model=[x["value"] for x in stats._test_compared_pool]
    )
    model._test_stats = stats
    model._test_files = stats._test_files
    return model
    

@given(floats(), finite_floats(), finite_floats())
def test_clamp(x, y, z):
    y,z = sorted((y,z))
    assert y <= state.clamp(x, y, z) <= z

@given(make_stats())
def test_serialize_stats(stats):
    a = msgpack.packb(stats.to_msgpack(), use_bin_type=True)
    b = msgpack.unpackb(a, raw=False, use_list=False)
    deserialized = state.Stats(*b)
    assert deserialized == stats

@given(precise_model(minholdout=1))
def test_next_index(model):
    f = model._test_stats._test_holdout_pool[0]
    assert model.calc_next_index(f["hash"], ([50], [50])) == len(model.model)//2

@given(precise_model(maxfiles=4))
def test_next_index_median(model, dists):
    f = model._test_files.af[0]
    m = sorted(model.model)

    after = 0
    if len(m):
        dists = (-state.clamp(-dists[0], -max(m), max(m)), state.clamp(dists[1], -max(m), max(m)))
    halfway = (dists[0]+dists[1]) / 2
    dists = ([dists[0]], [dists[1]])
    #halfway = (m[0] + m[-1]) / 2.0
    before = len(m)
    for i, x in enumerate(m):
        if halfway > x:
            after = i
            before = i+1

    val = model.calc_next_index(f["hash"], dists)
    assert val == before


