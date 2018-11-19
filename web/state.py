import choix
import subprocess
import time
import sys
import numpy
import msgpack
import json
import os
from .choix_custom import *
from .util import timing
from .files import duration, extract_time, ue

# long video
    #TODO: image or clip extraction from videos
        #TODO: detect cuts https://pyscenedetect.readthedocs.io/en/latest/
# text
    #TODO: clip extraction from text
    #TODO: download text datasets
# mobile
    #TODO: fix actual-mobile ui
    #TODO: mobile videos
# scanning
    #TODO: better exclusion
# data collection
    #TODO: allow painting a winning region
    #TODO: "which is more similar" queries
        # can do full on embedding style where it's unrelated things every time
        # can also try finding exemplars and compare to them
    #TODO: "which is a better exemplar" queries
    #TODO: comparison decay
# vision
    # TODO: look into pruning based transfer learning

class opts:
    minview = 900
    loss_threshold = 0
    explore_target = 10
    max_shows = 5
    neighborhood = 1
    neighborhood_ratio = 0.7
    high_quality_ratio = 0.0

    @classmethod
    def precision_func(cls, x):
        #(opts.min_target_precision + (pos ** opts.target_precision_curve) * opts.target_precision_top)
        #min_target_precision = 4
        J = 6.7
        z = 1800
        j = 1+(J/z)
        m = 2
        return m+j**(z**x)-j
        #target_precision_curve = 40
        #target_precision_top = 20
    drop_ratio = 20
    drop_min = 4
    model_drop_min = 4
    max_delay = 10
    inversion_threshold = 0.2222
    softmin_falloff_per_unit = 10.0
    inversion_neighborhood = 0.7
    inversion_max = 20
    seen_suppression_max = 24*2*60*60
    seen_suppression_min = 30
    seen_suppression_rate = 2
    fix_inversions = False

def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=0)

def squash(x, amount):
    amount /= 2
    return amount * (4/(1+numpy.exp(-min(max(x/amount,-300), 300))) - 2)

def rand_video_fragments(f, num_samples=None):
    dur = None
    if "video" in f and f["video"] and "min_time" not in f:
        dur = duration(f)
    if dur is None or dur < 10:
        return [f]
    results = []
    dc = int(min(numpy.sqrt(dur),dur/5))
    if dc == 0:
        return [f]
    dr = dur/dc
    num_samples = num_samples or int(dc)
    positions = random.sample(list(range(dc)), num_samples)
    for r in positions:
        start = dr * r
        end   = dr * (r+1)
        results.append(extract_time(f, start, end))
    return results


class SeenPool:
    def __init__(self):
        self.last_seen = {}
        self.seen_suppression = {}

    def bulk_check_seen(self):
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
            r = numpy.random.lognormal(0, 1, size=len(ss))
            ns = ls + ss * r
            s = now < ns
            return set(k[s])

    def mark_seen(self, h, at=None):
        now = time.time()
        if at is None:
            at = now
        if at < now / 10:
            print("invalid at", at)
            return
        if at > now*10:
            at = at / 1000

        ls = self.last_seen.get(h, at)
        assert ls > now / 10
        self.last_seen[h] = int(at)
        delta = at - ls
        assert delta < 60*60*24*30
        if delta < 0: print(f"warning: delta not greater than zero: {delta} - {h}")
        delta = max(delta, 0)
        self.seen_suppression[h] = max(int(squash(self.seen_suppression.get(h, opts.seen_suppression_min) * opts.seen_suppression_rate - delta, opts.seen_suppression_max)), opts.seen_suppression_min)

def clamp(x, m, M):
    return min(M, max(m, x))

def as_pair(v1, v2, extra_1=None, extra_2=None):
    if type(v1) == dict: v1 = v1.get("hash")
    if type(v2) == dict: v2 = v2.get("hash")
    pw = sorted([
        (v1, extra_1),
        (v2, extra_2)
    ])
    pair, values = zip(*pw)
    if extra_1 is not None or extra_2 is not None:
        return pair, values
    else:
        return pair

class Stats:
    def __init__(self, pair_wins=None, dislike=None, too_close=None, incomparable_pairs=None):
        self.pair_wins = pair_wins or {}
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

    def record_win(self, winning, losing):
        self.win_counts[winning["hash"]] = self.win_counts.get(winning["hash"], 0) + 1
        self.loss_counts[losing["hash"]] = self.loss_counts.get(losing["hash"], 0) + 1
        #w_ratio, w_incl = self.update_ratios(winning["hash"])
        #l_ratio, l_incl = self.update_ratios(losing["hash"])
        pair, values = as_pair(winning, losing, 1, 0)

        if pair in self.pair_wins:
            w1, w2 = self.pair_wins[pair]
            n1, n2 = values
            values = w1+n1, w2+n2
        self.pair_wins[pair] = values

        self.comparisons.setdefault(pair[0], {})[pair[1]] = values
        self.comparisons.setdefault(pair[1], {})[pair[0]] = values[::-1]
        #print(f'incremented winner win count to {self.win_counts[winning["hash"]]}, loser lose count to {self.loss_counts[losing["hash"]]} (flc: {self.filtered_loss_counts.get(losing["hash"], None)}; w: {w_ratio:0.2f}, {w_incl}; l: {l_ratio:0.2f}, {l_incl})')

    def update(self, item):
        item["dur"] = item.get("viewend", 0)-item.get("viewstart", 0)
        if item.get("dur",0)<opts.minview:
            print("skipped due to too-low view duration", item)
            return
        pair = as_pair(*item["items"])
        if type(item.get("preference", None)) != dict:
            winner = item.get("preference", 1) - 1
            too_close = False
            incomparable = False
            dislike = None
        else:
            info = item.get("preference", {})
            winner = info.get("prefer", 1) - 1
            too_close = info.get("too_close", False)
            incomparable = info.get("incomparable", False)
            dislike = info.get("dislike", None)
        if not dislike:
            dislike = [0,0]
        for f, dis in zip(item["items"], dislike):
            if dis:
                self.dislike[f["hash"]] = True
                #print("dislike",f)
            elif f["hash"] in self.dislike:
                del self.dislike[f["hash"]]
                #print("undislike",f)
        if any(dislike): return

        if too_close:
            self.too_close[pair] = self.too_close.get(pair, 0) + 1
            #self.record_win(*item["items"])
            #self.record_win(*item["items"][::-1])
        elif incomparable:
            self.incomparable_pairs[pair] = self.incomparable_pairs.get(pair, 0) + 1
        else:
            winning = item["items"][winner]
            losing = item["items"][1-winner]
            self.record_win(winning, losing)

    def from_history(self, history):
        keep = []
        wts = False
        for item in history:
            if not "items" in item:
                continue
            if "parent" in item:
                path = "/" + item["parent"] + "/" + item["name"]
                assert item["hash"] == hashlib.sha256(path.encode("utf-8", ue)).hexdigest()
            item["dur"] = item.get("viewend", 0)-item.get("viewstart", 0)
            if item.get("dur",0)<opts.minview:
                if not wts:
                    keep.pop()
                wts=True
                continue
            wts = False
            keep.append(item)
        for item in keep:
            self.update(item)
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

class Model:
    def __init__(self, all_items=None, model=None, searching_pool=None, inversions=None, new=None, newh=None, newp=None, distances=None, af2=None):
        self.all_items = all_items or []
        self.model = model or []

        self.searching_pool = searching_pool or {}
        self.inversions = inversions or {}

        self.af2 = af2 or []
        self.bh2 = {x[0]: x for x in self.af2}
        self.new = numpy.asarray(new if new is not None else [])
        self.newh = newh if newh is not None else []
        self.newp = newp if newp is not None else {}
        self.distances = distances or {}

    def fixh(self, fh):
        self.newh = [fh(h) for h in self.newh]
        self.all_items = [fh(h) for h in self.all_items]
        self.searching_pool = {fh(h): y for h, y in self.searching_pool.items()}
        self.af2 = [(fh(h), y) for (h, y) in self.af2]
        self.bh2 = {fh(h): y for h, y in self.bh2.items()}
        self.newp = {fh(h): y for h, y in self.newp.items()}

    def to_msgpack(self, include_derived=True):
        return [
            self.all_items,
            self.model.tolist(),

            self.searching_pool,
            self.inversions,
            self.new.tolist() if include_derived else [],
            self.newh if include_derived else [],
            self.newp if include_derived else {},
            self.distances,
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
    def model(self):
        return self._model
    @model.setter
    def model(self, val):
        self._model = numpy.asarray(val)
        self.sequence = numpy.argsort(val)
        self.sorted_model = self._model[self.sequence]
        self.sorted_hashes = numpy.asarray(self.all_items)[self.sequence]
        self.sorted_ids = {value: index for index, value in enumerate(self.sorted_hashes)}

    def add(self, x):
        assert x not in self.ids
        self.ids[x] = len(self.all_items)
        self.all_items.append(x)

    def getid(self, x):
        #assert not self.is_dropped(x)

        if x not in self.ids:
            self.add(x)
        return self.ids[x]

    def is_dropped(self, stats, h):
        id = self.ids.get(h)
        wc = stats.win_counts.get(h, 0)
        lc = stats.loss_counts.get(h, 0)
        res = 0
        
        if id is not None and id<len(self.sorted_model) and id<len(self.model):
            v = self.model[id]
            if v < -self.sorted_model[-1] and lc+wc >= opts.model_drop_min:
                res = 1

        if lc > max(wc*opts.drop_ratio, opts.drop_min) or h in stats.dislike:
            res = 2
        #id = self.getid(h)
        #if id < len(self.model):
        #    v = self.model[id]

        #if res: print(f"dropped: lc={self.loss_counts.get(h)}, wc={self.win_counts.get(h)}, v={v:.4f}")
        return res

    def min(self):
        return self.sorted_model[0] if len(self.sorted_model)>1 else -10
    def max(self):
        return self.sorted_model[-1] if len(self.sorted_model)>1 else 10

    def getprob_(self, item1, item2):
        with timing("getprob_"):
            a = self.getid(item1["hash"])
            b = self.getid(item2["hash"])
            a_new = False
            b_new = False
            if a >= len(self.model):
                a_new = True
            if b >= len(self.model):
                b_new = True
            #self.calculate_ranking()
            if not len(self.model) or a_new or b_new:
                return f"no model yet. new: {a_new}", f"no model yet. new: {b_new}"
            ra, rb = choix.probabilities([a, b], self.model)
            return f"prob: {ra:.2f}, val: {self.model[a]:.4f}, new: {a_new}",f"prob: {rb:.2f}, val: {self.model[b]:.4f}, new: {b_new}"


    def calculate_ranking(self, stats, extra=False):
        with timing("calculate_ranking"):
            pairs = []
            if len(stats.pair_wins) < 3:
                return
            if any((self.is_dropped(stats, x)>1) for x in self.all_items):
                self.all_items = []
                self.ids = {}
                self.model = []

            for pair, rel_wins in stats.pair_wins.items():
                if pair in stats.incomparable_pairs: continue
                ratio = (rel_wins[0] + 1) / (rel_wins[0] + rel_wins[1] + 2)
                if (self.is_dropped(stats, pair[0])>1) or (self.is_dropped(stats, pair[1])>1):
                    continue
                if pair in stats.too_close:
                    rel_wins = tuple([x + sum(rel_wins) + 3 * stats.too_close[pair] for x in rel_wins])
                pairs.extend([(self.getid(pair[0]), self.getid(pair[1]))] * rel_wins[0])
                pairs.extend([(self.getid(pair[1]), self.getid(pair[0]))] * rel_wins[1])
            if len(self.model) and len(self.all_items) > len(self.model):
                newlen = len(self.all_items) - len(self.model)
                gp = choix.generate_params(newlen, 0.1)
                #print(gp.shape)
                self.model = numpy.concatenate((self.model, gp))
            prev = None
            if len(self.model):
                prev = numpy.array(self.model)
            start = time.time()
            print(f"ranking {len(self.all_items)} items...")
            max_iter = 40
            if extra:
                max_iter = 10000
            self.model, iters, e, t = ilsr_pairwise(len(self.all_items), pairs, alpha=0.0001, initial_params=self.model if len(self.model) else None, max_iter=max_iter)
            end = time.time()
            print(f"done ranking, took {end-start:0.3f}, {iters} iters, last update norm: {e} (/param = {t})")
            #print(self.clamp(self.model - prev))

    def getidx(self, val):
        return numpy.searchsorted(self.sorted_model, val)
    def getval(self, h):
        id = self.getid(h)
        if id >= len(self.model):
            return None
        return self.model[id]

    def calc_next_index(self, h, vals, dists_out, debug=False, force=False):
        modelmin = self.min()
        modelmax = self.max()
        low_, high_ = self.softmin(vals[0], inv=True), self.softmin(vals[1])
        low = max(low_, -modelmax)
        high = min(high_, modelmax)
        dists_out[h] = high - low

        widx, lidx = self.getidx(low), self.getidx(high)
        mid = (low + high)/2
        midx = self.getidx(mid)
        pos = (mid-modelmin)/(modelmax-modelmin)
        prec = opts.precision_func(pos)
        delta = len(self.model)/prec
        #if low < modelmin: widx -= max((delta - 2), 1)
        #if high > modelmax: lidx += max((delta - 2), 1)
        #if debug: print(pos, delta, low, high, widx, lidx)
        if len(vals[0]) + len(vals[1]) > 5 and abs(lidx - widx) < delta * 2 and not force:
            return None, dict(locals())
        return midx, dict(locals())

    def softmin(self, directional_distances, inv=False):
        if not len(directional_distances):
            return 100
        vals = numpy.array(directional_distances) #numpy.log2(numpy.maximum(directional_distances*10, 0.0001))
        if inv: vals=-vals
        weight = numpy.minimum(opts.softmin_falloff_per_unit ** -vals, 2000000)
        sum = numpy.sum(weight * vals)
        total_weight = numpy.sum(weight)
        res = sum / total_weight
        if inv: res = -res
        return res
        #return (2 ** (sum / total_weight))/10
    # ^ use that to calculate the neighborhood an image needs on each side
    # then keep an index of what images need what neighborhoods
    # update images whose neighborhoods are too large - greater than 100 images, maybe? 30?
    # though might want to do this on the score, rather than the index, so if images get densely packed in score
    # it'll still stop asking about them
    # then this can replace comparison counts and ratio counts
    # get directional_distances from comparisons
    # possibly also treat strongly inverted comparisons as something to redisplay,
    # see if you want to change your mind
    # subject to seen-suppression of pool, of course

    def check_inversion(self, stats, pair):
        rel_wins = stats.pair_wins.get(pair, [0, 0])
        win_prob = choix.probabilities([self.getid(pair[0]), self.getid(pair[1])], self.model)[0]
        win_ratio = (rel_wins[0]) / (rel_wins[0] + rel_wins[1])
        if win_ratio < 0.5:
            win_prob = 1-win_prob
        inverted = win_prob < opts.inversion_threshold
        if sum(rel_wins) >= opts.inversion_max:
            inverted = False # not the problem at this point
        return win_prob, inverted

    def calculate_nearest_neighborhood(self, stats, hashes_to_debug, extra=False, save=True):
        with timing("calculate_nearest_neighborhood"):
            distances = {h: ([-50],[50]) for h in self.all_items}
            sp = {}
            if save: self.searching_pool=sp

            iv = {}
            if save: self.inversions = iv
            inversions = {}
            inversion_pool = set()
            dists = {}
            if save:
                self.distances = dists
            for pair, rel_wins in stats.pair_wins.items():
                if self.is_dropped(stats, pair[0]) or self.is_dropped(stats, pair[1]):
                    continue
                if pair in stats.incomparable_pairs: continue
                if pair in stats.too_close: continue
                win_ratio = (rel_wins[0]) / (rel_wins[0] + rel_wins[1])
                win, loss = pair[0], pair[1]
                count = rel_wins[0] + rel_wins[1]
                if win_ratio < 0.5:
                    rel_wins = rel_wins[::-1]
                    win, loss = pair[1], pair[0]
                    win_ratio = 1-win_ratio
                win_prob, inverted = self.check_inversion(stats, pair)
                if win_prob < 0.5:
                    win_inversions, _ = inversions.setdefault(win, ([],[]))
                    _, loss_inversions = inversions.setdefault(loss, ([],[]))
                    win_inversions.append((pair, win_ratio, win_prob, count))
                    loss_inversions.append((pair, win_ratio, win_prob, count))
                if inverted:
                    inversion_pool.add(pair[0])
                    inversion_pool.add(pair[1])
                    iv[pair] = ((win_ratio - win_prob) * 2)

                if rel_wins[1] != 0:
                    # don't include ambiguous comparisons when tallying distances
                    # should reduce risk of getting in tangles
                    continue
                wval = self.getval(win)
                lval = self.getval(loss)
                if wval is None or lval is None: continue
                dist = wval - lval
                distances[win][0].append(lval)
                distances[loss][1].append(wval)
            modelmin = self.min()
            modelmax = self.max()
            for h, vals in distances.items():
                if self.is_dropped(stats, h): continue
                nextidx,_ = self.calc_next_index(h, vals, dists)
                if nextidx is not None and h in self.bh2:
                    sp[h] = True

            #q = []
            if True:
                for h in hashes_to_debug:
                    hidx = self.sorted_ids.get(h, -1)
                    in_pool = h in sp
                    in_inv_pool = h in inversion_pool
                    #if not in_pool or in_inv_pool: continue
                    dists = distances[h]
                    win_inversions, loss_inversions = inversions.get(h, ([], []))
                    wdist, ldist = self.softmin(dists[0]), self.softmin(dists[1])
                    val = self.getval(h)
                    pos = (val-modelmin)/(modelmax-modelmin)
                    delta = int(len(self.model)/(opts.min_target_precision + (pos ** opts.target_precision_curve) * opts.target_precision_top))
                    lc = stats.loss_counts.get(h, 0)
                    wc = stats.win_counts.get(h, 0)
                    ##ld = "done" if ld is None else f"{ld:4d}"
                    ##wd = "done" if wd is None else f"{wd:4d}"
                    #if not wc or not lc or ld or wd: continue
                    ##print(f"{wd} <= {wc:4d}   {lc:4d} => {ld}")
                    rows = []

                    low = val - wdist
                    high = val + ldist
                    widx2, lidx2 = self.getidx(low), self.getidx(high)
                    #if low < modelmin: widx2 -= max((delta - 2), 1)
                    #if high > modelmax: lidx2 += max((delta - 2), 1)

                    widx, lidx = max(0, self.getidx(val - wdist)), min(len(self.model)-1, self.getidx(val + ldist))
                    midx = (widx + lidx)//2
                    mval = self.sorted_model[midx]
                    mdist = mval - val

                    low = max(low, -modelmax)
                    high = min(high, modelmax)
                    vval = (low + high)/2
                    vdist = vval-val
                    vidx  = self.getidx(vval)

                    waidx = max(0,hidx - 1)
                    laidx = min(len(self.model)-1, hidx + 1)
                    la_h = self.sorted_hashes[laidx]
                    wa_h = self.sorted_hashes[waidx]
                    la_val = self.getval(la_h)
                    wa_val = self.getval(wa_h)

                    wtidx = hidx-delta
                    ltidx = hidx+delta
                    wthresh_h = self.sorted_hashes[max(0,wtidx)]
                    lthresh_h = self.sorted_hashes[min(ltidx, len(self.sorted_hashes)-1)]
                    wthresh_val = self.getval(wthresh_h)
                    lthresh_val = self.getval(lthresh_h)
                    print()
                    print(f"wc={wc:2d}")
                    print(f"lc={lc:2d}")
                    print(f"lc+wc={lc+wc:2d}")
                    print(f"in_pool={in_pool}")
                    print(f"in_inv_pool={in_inv_pool}")
                    for pair, win_ratio, win_prob, count in win_inversions:
                        other = [x for x in pair if x != h][0]
                        other_idx = self.sorted_ids[other]
                        other_val = self.getval(other)
                        rows.append((other_val-val, 9, "iw", other_val, other_idx, other_idx-hidx, f"unexpected win; win ratio: {win_ratio} ({count} samples), expected win prob: {win_prob}"))
                    for pair, loss_ratio, loss_prob, count in loss_inversions:
                        other = [x for x in pair if x != h][0]
                        other_idx = self.sorted_ids[other]
                        other_val = self.getval(other)
                        rows.append((other_val-val, 3, "il", other_val, other_idx, other_idx-hidx, f"unexpected loss; loss ratio: {loss_ratio} ({count} samples), expected loss prob: {loss_prob}"))
                    for windist in dists[0]:
                        if windist == 9 or windist < 0: continue
                        rows.append((-windist, 0, "win", val-windist, self.getidx(val-windist), self.getidx(val-windist)-hidx, f"expected win"))
                    for lossdist in dists[1]:
                        if lossdist == 9 or lossdist < 0: continue
                        rows.append((lossdist, 12, "los", val+lossdist, self.getidx(val+lossdist), self.getidx(val+lossdist)-hidx, f"expected loss"))
                    rows.append((modelmin-val, 1, "W", modelmin, 0, 0-hidx, "model boundary low"))
                    rows.append((-wdist, 1, "w", val-wdist, widx2, widx2-hidx, "win boundary"))
                    rows.append((wthresh_val-val, 2, "wt", wthresh_val, wtidx, wtidx-hidx, "search precision threshold, win side"))
                    rows.append((vdist, 4, "v", vval, vidx, vidx-hidx, "midpoint in value space"))
                    rows.append((mdist, 5, "m", mval, midx, midx-hidx, "midpoint in index space"))
                    rows.append((wa_val-val, 6, "wa", wa_val, waidx, waidx-hidx, "prev neighbor"))
                    rows.append((0, 7, "", val, hidx, 0, "item"))
                    rows.append((la_val-val, 8, "la", la_val, laidx, laidx-hidx, "next neighbor"))
                    rows.append((lthresh_val-val, 10, "lt", lthresh_val, ltidx, ltidx-hidx, "search precision threshold, loss side"))
                    rows.append((ldist, 11, "l", val+ldist, lidx2, lidx2-hidx, "loss boundary"))
                    rows.append((modelmax-val, 11, "L", modelmax, len(self.model)-1, (len(self.model)-1)-hidx, "model boundary high"))
                    maxstep = 12
                    for dist, step, label, val, idx, idxdist, desc in sorted(rows, key=lambda x: (x[4], x[0])):
                        prefix = label.rjust(step*2).ljust(24)
                        label = label.rjust(3)
                        print(f" {prefix} | {label}dist={dist:7.4f} {label}val={val:7.4f} {label}idx={idx:5d} {label}idxdist={idxdist:5d} {desc}")
                    #print(sorted(dists[0]))
                    #print(f"  wdist={-wdist:7.4f}   wval={val-wdist:7.4f}  widx={widx:5d}  widxdist={self.getidx(val-wdist)-hidx:5d}")
                    #print(f"wthresh={wthresh_val-val:7.4f}  wtval={wthresh_val:7.4f} wtidx={wtidx:5d} wtidxdist={wtidx-hidx:5d}")
                    #print(f"  vdist={vdist:7.4f}   vval={vval:7.4f}  vidx={vidx:5d}  vidxdist={vidx-hidx:5d}")
                    #print(f"  mdist={mdist:7.4f}   mval={mval:7.4f}  midx={midx:5d}  midxdist={midx-hidx:5d}")
                    #print(f" wadist={wa_val-val:7.4f}  waval={wa_val:7.4f} waidx={waidx:5d} waidxdist={waidx-hidx:5d}")
                    #print(f"                   val={val:7.4f}  hidx={hidx:5d}")
                    #print(f" ladist={la_val-val:7.4f}  laval={la_val:7.4f} laidx={laidx:5d} laidxdist={laidx-hidx:5d}")
                    #print(f"lthresh={lthresh_val-val:7.4f}  ltval={lthresh_val:7.4f} ltidx={ltidx:5d} ltidxdist={ltidx-hidx:5d}")
                    #print(f"  ldist={ldist:7.4f}   lval={val+ldist:7.4f}  lidx={lidx:5d}  lidxdist={self.getidx(val+ldist)-hidx:5d}")
                    #print(sorted(dists[1]))
            #print(f"min(model): {min(self.model)}, max(model): {max(self.model)}, mean(model): {numpy.mean(self.model)}")
            print(f"searching_pool: {len(sp)}, inversions: {len(iv)}")
    def update_new_pool(self):
        with timing("update_new_pool"):
            af = self.af2
            bh = self.bh2
            ds={}
            for h, v in zip(self.all_items, self.model):
                i = bh.get(h)
                if i is None:continue
                d=i[1]
                dp = d.split("/")
                if ":" in h:
                    dp.append(h)
                u = self.distances.get(h, 0.5)
                for slice in range(1, max(2,len(dp))):
                    ds.setdefault("/".join(dp[:slice+1]), []).append((v, max(0.1, u)))
            new = []
            newh = []
            newp = {}
            cc = {}
            for i in af:
                h=i[0]
                if h in self.ids: continue
                d=i[1]
                dp = d.split("/")
                if ":" in h:
                    dp.append(h)
                mean = 0
                stddev = 4
                var = stddev ** 2
                for slice in range(1, max(2,len(dp))):
                    dn="/".join(dp[:slice+1])
                    if dn not in ds: break
                    if dn not in cc:
                        vs = numpy.array(ds[dn])
                        smean, std = numpy.mean(vs, axis=0)
                        #_, var_ = self.update_bayes(0, 1, len(vs), smean, std ** 2)
                        mean, var = self.update_bayes(mean, var, len(vs), smean,max(std**2, numpy.var(vs[:,0]))) #var_ * len(vs))
                        cc[dn] = mean, var * len(vs)
                        #print(dn, "smean,std:", smean, std, "mean:", mean, "std:", numpy.sqrt(var), "p_std:", numpy.sqrt(var*len(vs)), "n:", len(vs))
                    mean, var = cc[dn]
                stddev = numpy.sqrt(var)
                new.append(mean + 1 * stddev)
                newh.append(h)
                newp[h] = mean
            self.new = softmax(new)
            self.newh = newh
            self.newp = newp
    def update_bayes(self, mean, var, n, sample_mean, sample_var):
        mean = ( sample_var * mean +n * var * sample_mean ) * 1/(n * var+ sample_var)
        var = ( sample_var * var)/(n * var+ sample_var)
        return mean, var

    def update_bayes_multi(self, mean, stddev, sample_mean, sample_stddev):
        sample_var = sample_stddev ** 2
        var = stddev ** 2
        upper2 = sample_var * sample_mean

        upper1 = sample_var[0] * mean 
        denom = 1/(var + sample_var[0])
        mean = upper1 * denom + upper2[0] * denom
        var = var * sample_var[0] * denom
        
        upper1 = sample_var[1] * upper1 * denom + sample_var[1] * upper2[0] * denom
        denom = 1/(var + sample_var[1])
        mean = upper1 * denom + upper2 * denom
        var = var * sample_var[1] * denom

    def slow_calculations(self, stats, hashes_to_debug, extra=False):
        self.calculate_ranking(stats, extra)
        self.calculate_nearest_neighborhood(stats, hashes_to_debug, extra)
        self.update_new_pool()
    def __repr__(self):
        return f"Model(all_items={self.all_items}, model={self.model.tolist()}, searching_pool={self.searching_pool}, inversions={self.inversions})"

class State:
    def __init__(self, files, tempdir):
        self.files = files
        self.tempdir = tempdir
        self.subproc = None
        self.history = []

        self.removed_pool = set()
        self.extra_pool = set()
        self.inversion_fixes = {}
        self.fixed_inversions = set()

        self.stats = Stats()
        af, _ = files.get_all_images()
        self.af = []
        for x in af:
            self.af.extend(rand_video_fragments(x))

        self.bh = {x["hash"]: x for x in self.af + af}
        for f in list(self.bh.values()):
            _, colon, postfix = f["hash"].partition(":")
            for h in f.get("other_hashes", []):# in f:
                self.bh[h+colon+postfix] = f
        new = softmax([1] * len(self.af)) if len(af) else []
        newh = [x["hash"] for x in self.af]
        self.inputfile = os.path.join(self.tempdir, "input")
        self.outputfile = os.path.join(self.tempdir, "output")
        self.completionfile = os.path.join(self.tempdir, "completion")
        self.affile = os.path.join(self.tempdir, "af2")
        self.af2 = [(x["hash"], x["parent"]) for x in self.af]
        self.model_ = Model(af2=self.af2, new=new, newh=newh)
        self.seen = SeenPool()
        self.last_slow = 0

    def fh(self, h):
        return self.bh.get(h, {}).get("hash", h)

    def read(self):
        try:
            os.makedirs(self.tempdir)
        except FileExistsError:
            pass
        with open(self.affile, "wb") as writer:
            msgpack.pack(self.af2, writer, use_bin_type=True)

        if os.path.exists("preferences.json"):
            with open("preferences.json", "r") as reader:
                for line in reader:
                    if not line: continue
                    self.history.append(json.loads(line))
                    if "items" in self.history[-1]:
                        self.history[-1]["items"] = [self.bh.get(item["hash"], item) for item in self.history[-1]["items"]]

        self.stats.from_history(self.history)
        for x in self.history:
            if "items" not in x: continue
            i = x["items"]
            self.seen.mark_seen(i[0]["hash"], x["viewstart"])
            self.seen.mark_seen(i[1]["hash"], x["viewstart"])
        z = numpy.array(list(self.seen.seen_suppression.values()))
        ss = self.seen.bulk_check_seen()
        print(f"seen_suppression: mean={numpy.mean(z)}, max={numpy.max(z)}, min={numpy.min(z)}, median={numpy.median(z)}, seen={len(ss)}")
        if not self.reap_slow():
            self.launch_slow(wait=True)

    def select_next(self):
        self.reap_slow()
        #return (
        #    bh[self.model_.sorted_hashes[max(self.getidx(-max(self.model)), 0)]], 
        #    bh[self.model_.sorted_hashes[min(self.getidx(-min(self.model)), len(self.model)-1)]],
        #    "x", "x")
        print()
        print(f"begin select_next.")
        bulk_seen = self.seen.bulk_check_seen()
        print("bulk_seen:", len(bulk_seen))
        last_winner = None
        indices = {}
        if self.history:
            last = self.history[-1]
            last_options = last["items"]
            indices[last_options[0]["hash"]] = -1
            indices[last_options[1]["hash"]] = 1
            pref = last.get("preference", None)
            if type(pref) == dict:
                pref = pref.get("prefer", None)
            if pref is not None:
                last_winner = last_options[pref - 1]

        first = second = None

        for x in range(40):
            first = firstlabel = None
            second = secondlabel = None

            base_pool = ((self.model_.searching_pool.keys() | self.extra_pool) - self.removed_pool)
            randpool_set = base_pool #- bulk_seen
            randpool =  list(randpool_set)
            print(f"randpool size: {len(randpool)}")
            if not len(randpool):
                self.reap_slow(True)
                self.launch_slow()
            prior = None

            force_neighborhood = False
            if opts.fix_inversions:
                inversions = self.model_.inversions
            else:
                inversions = {}

            if last_winner and last_winner["hash"] in base_pool:
                first = last_winner
                firstlabel = "last winner"
            elif random.random() < opts.high_quality_ratio and len(self.model_.sorted_hashes):
                hidx = max(min(len(self.model_.sorted_hashes)-1, int((random.random() ** (1/6)) * len(self.model_.sorted_hashes))), 0)
                h = self.model_.sorted_hashes[hidx]
                if h not in self.bh:
                    print("fail on missing high quality item")
                    continue
                if h in bulk_seen:
                    print("fail on recently seen high quality item")
                    continue
                first = self.bh[h]
                firstlabel = f"high-quality ({hidx}/{len(self.model_.sorted_hashes)}):"
            elif (random.random() > (len(randpool) + len(inversions)) / opts.explore_target and len(self.model_.newh)):
                firsth = numpy.random.choice(self.model_.newh, p=self.model_.new)
                prior = self.model_.newp[firsth]
                first = self.bh[firsth]
                firstlabel = "explore"
            elif random.randrange(0, len(randpool) + len(inversions)) < len(randpool):
                h = random.choice(randpool)
                if h not in self.bh:
                    print("fail on missing randpool item")
                    continue
                first = self.bh[h]
                firstlabel = f"randpool (lc: {self.stats.loss_counts.get(h)})"
            else:
                for x in range(10):
                    pair, prob = random.choice(list(inversions.items()))
                    if random.random() < prob:
                        break
                else:
                    print("pulled 10 items from inversions pool, all too close to bother with")
                    continue
                if random.random() < opts.inversion_neighborhood:
                    first = self.bh[random.sample(pair,2)[0]]
                    force_neighborhood = True
                else:
                    first, second = self.bh[pair[0]], self.bh[pair[1]]
                    
                firstlabel = secondlabel = "inversions"
            first, = rand_video_fragments(first, 1)

            if second is None:
                firsthash = first["hash"]
                if prior is not None:
                    idx = self.model_.getidx(prior)
                else:
                    comparisons = self.stats.comparisons.get(firsthash, {})
                    dists = ([-50], [50])
                    # TODO: use directory prior for initial comparison?
                    for other_hash, wins in comparisons.items():
                        other_val = self.model_.getval(other_hash)
                        if other_val is None: continue
                        if wins[0] > wins[1] and wins[1] == 0:
                            dists[0].append(other_val)
                        elif wins[0] < wins[1] and wins[0] == 0:
                            dists[1].append(other_val)
                        else:
                            continue
                    idx,c = self.model_.calc_next_index(firsthash, dists, {}, debug=True, force=force_neighborhood)
                    del c["self"]
                    if idx is None:
                        print(f"attempted neighborhood on {firsthash}, but next index None, marking done")
                        self.removed_pool.add(firsthash)
                        continue
                        #idx = self.model_.sorted_ids[firsthash]
                nbs = max(3,int(numpy.random.lognormal(0, 1)*opts.neighborhood))
                print("nbs", nbs)

                zone=min(len(self.model_.sorted_hashes)-(1+nbs), max(nbs, idx))
                sss = (list(self.model_.sorted_hashes[zone-nbs:zone+nbs]))
                if not len(sss):
                    print("fail on empty slice of sorted hashes")
                    continue
                for x in range(10):
                    h = random.choice(sss)
                    if h == firsthash:
                        continue
                    if h not in bulk_seen:
                        break
                else:
                    print("fail on seen all of neighborhood too recently")
                    continue

                if h not in self.bh:
                    print("fail on missing neighborhood")
                    continue
                second = self.bh[h]
                secondlabel = f"neighborhood ({idx}->{self.model_.sorted_ids[h]})"
            second, = rand_video_fragments(second, 1)
            print("result:")
            print(firstlabel)
            print(secondlabel)

            if self.model_.is_dropped(self.stats, first["hash"]) or self.model_.is_dropped(self.stats, second["hash"]):
                print("one of the options was dropped, resampling")
                continue
            if as_pair(first, second) in self.stats.incomparable_pairs:
                print("got an incomparable pair, resampling")
                continue
            if as_pair(first,second) in self.stats.too_close:
                print("pair marked too close, resampling")
                continue
            if first is None or second is None:
                print("first or second is None", first, second)
                continue
            if first.get("hash") == second.get("hash"):
                print("same val for both")
                continue
            break
        else:
            assert False, f"Failed to find a pair. {len(self.history)}, {len(self.model_.model)}, {len(randpool)}, {len(self.model_.searching_pool)}, {last_winner}, {last_winner['hash'] in base_pool}"

        with timing("last bit"):
            proba, probb = self.model_.getprob_(first, second)
            firstlabel += ", " + proba
            secondlabel += ", " + probb
            paired_up = list(zip([first, second], [firstlabel, secondlabel], [random.random()-0.5, 0]))
            paired_up = sorted(paired_up, key=lambda x: indices.get(x[0]["hash"], x[2]))
            pa, pb = paired_up
            a, proba, _ = pa
            b, probb, _ = pb
            print("1:", proba)
            print("2:", probb)
            self.seen.mark_seen(a["hash"])
            self.seen.mark_seen(b["hash"])
            if a["hash"] not in self.model_.sorted_ids:
                self.extra_pool.add(a["hash"])
            if b["hash"] not in self.model_.sorted_ids:
                self.extra_pool.add(b["hash"])
        sys.stdout.flush()
        return a, b, proba, probb


        #for d, vs in sorted(list(ds.items())):
        #    print(f"{d}: {len(vs):6d} samples, {numpy.std(vs):6.3f} std, {numpy.mean(vs):6.3f} mean, {min(vs):6.3f} min, {max(vs):6.3f} max, {numpy.median(vs):6.3f} median")

    def reap_slow(self, wait=False):
        if self.subproc is not None and self.subproc.poll() is None and not os.path.exists(self.completionfile):
            if wait:
                while not os.path.exists(self.completionfile):
                    # TODO: less dumb thing than this
                    time.sleep(0.05)
            else:
                return False
        if self.subproc and self.subproc.poll() is not None:
            self.subproc = None
            
        if os.path.exists(self.completionfile):
            # have one waiting to read
            with timing("reap_slow::readback"):
                with open(self.outputfile, "rb") as reader:
                    packed_model = msgpack.unpack(reader, use_list=False, raw=False)
                self.model_ = Model(*packed_model)
                self.model_.fixh(self.fh)
                self.removed_pool = set()
                self.extra_pool = set()
                self.inversion_fixes = {}
                self.fixed_inversions = set()
                return True
        return False

    def launch_slow(self, wait=False):
        with timing("launch_slow"):
            packed_stats = self.stats.to_msgpack()
            packed_model = self.model_.to_msgpack(include_derived=False)
            with open(self.inputfile, "wb") as writer:
                msgpack.pack([packed_stats, packed_model], writer, use_bin_type=True)
            if os.path.exists(self.completionfile):
                os.unlink(self.completionfile)

        if self.subproc is None:
            self.subproc = subprocess.Popen([sys.executable, "-m", "web", self.inputfile, self.affile, self.outputfile, self.completionfile], stdin=open("/dev/null", "rb"))
        if wait:
            self.reap_slow(True)

    def update(self, info, file1, file2):
        with timing("update::save"):
            with open("preferences.json", "a") as appender:
                a = dict(info)
                a["items"] = [self.bh.get(file1, {"hash": file1}), self.bh.get(file2, {"hash": file2})]
                appender.write(json.dumps(a) + "\n")
            pair = as_pair(file1, file2)

        with timing("update::update"):
            self.history.append(a)
            self.stats.update(a)
            if pair in self.model_.inversions:
                self.inversion_fixes[pair] = self.inversion_fixes.get(pair, 0) + 1
                if self.inversion_fixes[pair] > 2 or not self.model_.check_inversion(self.stats, pair)[1]:
                    self.fixed_inversions.add(pair)
            print("updated from", self.model_.sorted_ids.get(file1, None), self.model_.sorted_ids.get(file2, None))
        self.launch_slow()
