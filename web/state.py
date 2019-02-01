import choix
import subprocess
import time
import sys
import numpy
import tempfile
import msgpack
import json
import os
from web import choix_custom
from .util import timing, nanguard
from web import util
from .files import duration, extract_time, ue
from web import opts
from importlib import reload
from web import state_
from web import model_
from web import stats_

# long video
    #TODO: detect cuts https://pyscenedetect.readthedocs.io/en/latest/
# text
    #TODO: clip extraction from text
    #TODO: download text datasets
# data collection
    #TODO: allow painting a winning region
    #TODO: "which is more similar" queries
        # can do full on embedding style where it's unrelated things every time
        # can also try finding exemplars and compare to them
    #TODO: "which is a better exemplar" queries
    #TODO: comparison decay
# vision
    # TODO: look into pruning based transfer learning




def reload_all():
    global opts
    global state_
    global model_
    global stats_
    global util
    global choix_custom
    #global seen_
    try:
        opts = reload(opts)
        state_ = reload(state_)
        model_ = reload(model_)
        stats_ = reload(stats_)
        util = reload(util)
        choix_custom = reload(choix_custom)
    except (SyntaxError, IndentationError):
        import traceback; traceback.print_exc()


class SeenPool:
    def __init__(self):
        self.last_seen = {}
        self.seen_suppression = {}

    def bulk_check_seen(self, mult=1):
        with timing("bulk_check_seen"):
            # hooray! as of 3.7, dict preserves insertion order <3
            now = time.time()
            if not self.seen_suppression:
                return set()
            k = numpy.array(list(self.seen_suppression.keys()))
            ss = numpy.array(list(self.seen_suppression.values()))
            ls = numpy.array(list(self.last_seen.values()))
            #assert dict(zip(list(k), list(ss))) == self.seen_suppression
            #assert dict(zip(list(k), list(ls))) == self.last_seen
            r = numpy.random.lognormal(opts.seen_noise_lmean, opts.seen_noise_lstd, size=len(ss))
            ns = ls + ss * r * mult
            s = now < ns
            return set(k[s])

    def mark_seen(self, h, at=None):
        now = time.time()
        if at is None:
            at = now
        if at < nanguard(now / 10):
            print("invalid at", at)
            return
        if at > nanguard(now*10):
            at = nanguard(at / 1000)

        ls = self.last_seen.get(h, at)
        if ls <= nanguard(now / 10):
            raise Exception()
        self.last_seen[h] = int(at)
        delta = nanguard(at - ls)
        if delta < 0: print(f"warning: delta not greater than zero: {delta} - {h}")
        delta = max(delta, 0)
        self.seen_suppression[h] = max(int(nanguard(util.squash(self.seen_suppression.get(h, opts.seen_suppression_min) * opts.seen_suppression_rate - delta, opts.seen_suppression_max))), opts.seen_suppression_min)

class Stats:
    def __init__(self, pair_wins=None, dislike=None, too_close=None, incomparable_pairs=None, triplet_diffs=None):
        self.pair_wins = pair_wins or {}
        self.triplet_diffs = triplet_diffs or {}
        self.dislike = dislike or {}
        self.too_close = too_close or {}
        self.incomparable_pairs = incomparable_pairs or {}

    @property
    def pair_wins(self):
        return self._pair_wins
    @pair_wins.setter
    def pair_wins(self, val):
        self._pair_wins = val
        self.loss_counts = {}
        self.win_counts = {}
        self.comparisons = {}
        for pair, values in val.items():
            if values[1]:
                self.loss_counts[pair[0]] = self.loss_counts.get(pair[0], 0) + values[1]
                self.win_counts[pair[1]] = self.win_counts.get(pair[1], 0) + values[1]
            if values[0]:
                self.loss_counts[pair[1]] = self.loss_counts.get(pair[1], 0) + values[0]
                self.win_counts[pair[0]] = self.win_counts.get(pair[0], 0) + values[0]
            self.comparisons.setdefault(pair[0], {})[pair[1]] = values
            self.comparisons.setdefault(pair[1], {})[pair[0]] = values[::-1]

    @property
    def triplet_diffs(self):
        return self._triplet_diffs

    @triplet_diffs.setter
    def triplet_diffs(self, val):
        self._triplet_diffs = val
        self.similarity = {}
        for pair, values in val.items():
            for ia,ib,ic in itertools.permutations(list(range(3)), 3):
                self.similarity.setdefault(pair[ia], {}).setdefault(pair[ib], {})[pair[ic]] = (values[ia], values[ib], values[ic])

    def to_msgpack(self):
        return [
            self.pair_wins,
            self.dislike,
            self.too_close,
            self.incomparable_pairs
        ]

    #def update_ratios(self, h):
    #    wins = self.win_counts.get(h, 0)
    #    losses = self.loss_counts.get(h, 0)
    #    ratio = (wins+1) / (losses+1)
    #    included = not (ratio > opts.winning_ratio or ratio < opts.losing_ratio or (wins + losses) > opts.max_shows)
    #    return ratio, included

    def __eq__(self, other):
        return (self.pair_wins == other.pair_wins and
                self.dislike == other.dislike and
                self.too_close == other.too_close and
                self.incomparable_pairs == other.incomparable_pairs and
                self.comparisons == other.comparisons and
                self.win_counts == other.win_counts and
                self.loss_counts == other.loss_counts)

    def __repr__(self):
        return (
            f"Stats(pair_wins={self.pair_wins}, dislike={self.dislike}, too_close={self.too_close}, incomparable_pairs={self.incomparable_pairs}, comparisons={self.comparisons}, win_counts={self.win_counts}, loss_counts={self.loss_counts})"
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            res = getattr(stats_, name)
            if hasattr(res, "__get__"):
                res = res.__get__(self, type(self))
            return res

class Model:
    def __init__(self, all_items=None, model=None,
            searching_pool=None, inversions=None, new=None,
            newh=None, newp=None, distances=None,
            all_sim_items=None, searching_pool_sim=None,
            vec_file=None, af2=None):
        self.all_items = all_items or []
        self.model = nanguard(model or [])

        self.all_sim_items = all_sim_items or []
        vec_stds = None
        vec_means = None
        next_directions = None
        if vec_file is not None and os.path.exists(vec_file):
            with numpy.load(vec_file) as reader:
                vec_means = reader["vec_means"]
                vec_stds = reader["vec_stds"]
                next_directions = reader["next_directions"]
        self.vec_means = nanguard(vec_means if vec_means is not None else [])
        self.vec_stds = nanguard(vec_stds if vec_stds is not None else [])
        self.next_directions = nanguard(next_directions if next_directions is not None else numpy.asarray([]))

        self.searching_pool = searching_pool or {}
        print("setting searching_pool_sim:", len(searching_pool_sim) if searching_pool_sim is not None else None)
        self.searching_pool_sim = searching_pool_sim or {}
        self.inversions = inversions or {}

        self.af2 = af2 or []
        self.bh2 = {x[0]: x for x in self.af2}
        self.new = numpy.asarray(new if new is not None else [])
        self.newh = newh if newh is not None else []
        self.newp = newp if newp is not None else {}
        self.distances = distances or {}
        self.ranked_before = False

    def fixh(self, fh):
        self.newh = [fh(h) for h in self.newh]
        self.all_items = [fh(h) for h in self.all_items]
        self.searching_pool = {fh(h): y for h, y in self.searching_pool.items()}
        self.all_sim_items = [fh(h) for h in self.all_sim_items]
        self.searching_pool_sim = {fh(h): y for h, y in self.searching_pool_sim.items()}
        self.af2 = [(fh(h), y) for (h, y) in self.af2]
        self.bh2 = {fh(h): y for h, y in self.bh2.items()}
        self.newp = {fh(h): y for h, y in self.newp.items()}

    def to_msgpack(self, direction, include_derived=True):
        if os.path.exists("/run/shm"):
            td = "/run/shm/app/"
        else:
            td = os.path.join(tempfile.gettempdir(), "app")
        if not os.path.exists(td):
            os.makedirs(td)
        filename = os.path.join(td, direction+"_vecs.npz")
        numpy.savez(filename, vec_means=self.vec_means,vec_stds=self.vec_stds, next_directions=self.next_directions)
        return [
            self.all_items,
            self.model.tolist(),

            self.searching_pool,
            self.inversions,
            self.new.tolist() if include_derived else [],
            self.newh if include_derived else [],
            self.newp if include_derived else {},
            self.distances,
            self.all_sim_items,
            self.searching_pool_sim,
            filename
        ]

    @property
    def all_items(self):
        return self._all_items
    @all_items.setter
    def all_items(self, val):
        if type(val) != list:
            val = list(val)
        self._all_items = val
        self.ids = {value: index for index, value in enumerate(self._all_items)}
    @property
    def all_sim_items(self):
        return self._all_sim_items
    @all_sim_items.setter
    def all_sim_items(self, val):
        if type(val) != list:
            val = list(val)
        self._all_sim_items = val
        self.ids_sim = {value: index for index, value in enumerate(self._all_sim_items)}

    @property
    def vec_means(self):
        return self._vec_means
    @vec_means.setter
    def vec_means(self, val):
        self._vec_means = numpy.asarray(val)
    @property
    def vec_stds(self):
        return self._vec_stds
    @vec_stds.setter
    def vec_stds(self, val):
        self._vec_stds = numpy.asarray(val)

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, val):
        self._model = numpy.asarray(val)
        self.sequence = numpy.argsort(val)
        self.sorted_model = self._model[self.sequence]
        self.sorted_hashes = numpy.asarray(self.all_items)[self.sequence]
        self.sorted_ids = {value: index for index, value in enumerate(self.sorted_hashes)}

    def min(self):
        return self.min_()
    def max(self):
        return self.max_()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            res = getattr(model_, name)
            if hasattr(res, "__get__"):
                res = res.__get__(self, type(self))
            return res

class State:
    def __init__(self, preffile, files, tempdir, update=True, do_reap=True):
        self.preffile = os.path.abspath(preffile)
        self.do_reap = do_reap
        self.do_update = update
        self.files = files
        self.tempdir = tempdir
        self.subproc = None
        self.history = []
        self.current = None
        self.read_needed = True
        self.force_current = False
        self.lock = None

        self.seen_allowed = set()
        self.removed_pool = set()
        self.extra_pool = set()
        self.inversion_fixes = {}
        self.fixed_inversions = set()
        self.dirty = set()

        self.stats = Stats()
        af, _ = files.get_all_images()
        self.af = []
        for x in af:
            self.af.extend(state_.rand_video_fragments(x))

        self.bh = {x["hash"]: x for x in self.af + af}
        for f in list(self.bh.values()):
            _, colon, postfix = f["hash"].partition(":")
            for h in f.get("other_hashes", []):# in f:
                self.bh[h+colon+postfix] = f
        new = util.softmax([1] * len(self.af)) if len(af) else []
        newh = [x["hash"] for x in self.af]
        if tempdir is not None:
            self.infofile = os.path.join(self.tempdir, "info")
            self.inputfile = os.path.join(self.tempdir, "input")
            self.outputfile = os.path.join(self.tempdir, "output")
            self.completionfile = os.path.join(self.tempdir, "completion")
            self.readyfile = os.path.join(self.tempdir, "ready")
            self.affile = os.path.join(self.tempdir, "af2")
        self.af2 = [(x["hash"], x["parent"]) for x in self.af]
        self.model_ = Model(af2=self.af2, new=new, newh=newh)
        self.seen = SeenPool()
        self.last_slow = 0

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            res = getattr(state_, name)
            if hasattr(res, "__get__"):
                res = res.__get__(self, type(self))
            return res
