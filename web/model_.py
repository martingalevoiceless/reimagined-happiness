import time
import numpy

def add(self, x):
    if x in self.ids:
        raise Exception()
    self.ids[x] = len(self.all_items)
    self.all_items.append(x)

def getid(self, x):
    if x not in self.ids:
        self.add(x)
    return self.ids[x]

def is_dropped(self, stats, h):
    from web import opts
    id = self.ids.get(h)
    wc = stats.win_counts.get(h, 0)
    lc = stats.loss_counts.get(h, 0)
    res = 0
    
    #if id is not None and id<len(self.sorted_model) and id<len(self.model):
    #    v = self.model[id]
    #    if v < -self.sorted_model[-1] and lc+wc >= opts.model_drop_min:
    #        res = 1

    if lc > max(wc*opts.drop_ratio, opts.drop_min):
        res = 2
    elif h in stats.dislike:
        res = 3
    #id = self.getid(h)
    #if id < len(self.model):
    #    v = self.model[id]

    #if res: print(f"dropped: lc={self.loss_counts.get(h)}, wc={self.win_counts.get(h)}, v={v:.4f}")
    return res

def min_(self):
    return self.sorted_model[0] if len(self.sorted_model)>1 else -10
def max_(self):
    return self.sorted_model[-1] if len(self.sorted_model)>1 else 10

def getprob(self, item1, item2):
    import choix
    from web.util import nanguard
    a = self.getid(item1["hash"] if type(item1) == dict else item1)
    b = self.getid(item2["hash"] if type(item2) == dict else item2)
    a_new = False
    b_new = False
    if a >= len(self.model):
        a_new = True
    if b >= len(self.model):
        b_new = True
    #self.calculate_ranking()
    if not len(self.model) or a_new or b_new:
        return 0.5, 0.5
        #return f"no model yet. new: {a_new}", f"no model yet. new: {b_new}"
    ra, rb = choix.probabilities([a, b], self.model)
    return nanguard(ra, "ra"), nanguard(rb, "rb")
    #return f"prob: {ra:.2f}, val: {self.model[a]:.4f}, new: {a_new}",f"prob: {rb:.2f}, val: {self.model[b]:.4f}, new: {b_new}"

def prepare_pairs(self, stats):
    from web.util import nanguard
    from web import opts
    pairs = []
    #removeme = []
    #for x in self.all_items:
    #    if self.is_dropped(stats, x)>1 and x in self.ids:
    #        removeme.append(self.ids[x])
    #if removeme:
    #    model = list(self.model)
    #    all_items = list(self.all_items)
    #    assert type(self.all_items) == list
    #    for idx in sorted(removeme)[::-1]:
    #        del all_items[idx]
    #        del model[idx]
    #    self.model=numpy.array(model)
    #    self.all_items =numpy.array(all_items)
    #self.ids = {x: idx for idx, x in enumerate(self.all_items)}

    for pair, rel_wins in stats.pair_wins.items():
        if pair in stats.incomparable_pairs: continue
        if (self.is_dropped(stats, pair[0])>1) or (self.is_dropped(stats, pair[1])>1):
            continue
        if pair[0] not in self.ids or pair[1] not in self.ids:
            continue
        if pair in stats.too_close:
            rel_wins = tuple([nanguard(x + sum(rel_wins) + opts.too_close_boost * stats.too_close[pair]) for x in rel_wins])
        if not sum(rel_wins):
            continue
        ratio = nanguard((rel_wins[0]) / (rel_wins[0] + rel_wins[1] ))
        scale = 1#sigmoid(3*(sum(rel_wins)-1))
        rel_wins = [scale*ratio, scale*(1-ratio)]
        id0, id1 = self.ids[pair[0]], self.ids[pair[1]]
        
        if rel_wins[0]:
            pairs.append((id0, id1, nanguard(rel_wins[0])))
        if rel_wins[1]:
            pairs.append((id1, id0, nanguard(rel_wins[1])))
    return pairs

def extend_model(self, stats):
    from web.util import nanguard
    for pair, rel_wins in stats.pair_wins.items():
        self.getid(pair[0])
        self.getid(pair[1])
    if len(self.model) and len(self.all_items) > len(self.model):
        newlen = len(self.all_items) - len(self.model)
        #gp = choix.generate_params(newlen, 0.1)
        #print(gp.shape)
        newvals = []
        for h in self.all_items[len(self.model):]:
            dists, weights = self.calculate_dists((stats, stats.comparisons.get(h, {})))
            info = self.calc_next_index(h, dists, weights, {})[1]
            newvals.append(nanguard(info["mid"]))
        if len(newvals) != newlen:
            raise Exception()
        self.model = numpy.concatenate((self.model, newvals))
    #if any(self.is_dropped(stats, x) >= 2 for x in self.all_items):
    #    self.model, self.all_items = zip(*(
    #        (v, h)
    #        for v, h
    #        in zip(self.model, self.all_items)
    #        if not self.is_dropped(stats, h) >= 2
    #    ))
        #for idx, x in list(enumerate(self.all_items))[::-1]:
        #    if self.is_dropped(stats, x) >= 3:
        #        model.pop(idx)
        #        self.all_items.pop(idx)
        #self.model = numpy.array(model)

def calculate_ranking(self, stats, extra=False):
    "calculate ranking. returns whether more is needed"
    from web import opts
    from web.choix_custom import ilsr_pairwise
    from web.util import timing
    if len(stats.pair_wins) < opts.min_to_rank:
        return False
    with timing("calculate_ranking", 0.1):
        self.extend_model(stats)
        pairs = self.prepare_pairs(stats)
        prev = None
        if len(self.model):
            prev = numpy.array(self.model)
        start = time.time()
        print(f"ranking {len(self.all_items)} items...")
        max_iter = 100
        tol = opts.initial_tolerance
        if self.ranked_before:
            max_iter = 10
            tol = opts.ongoing_tolerance
        self.ranked_before = True
        self.model, iters, e, t = ilsr_pairwise(len(self.all_items), pairs, alpha=opts.alpha(len(self.all_items)), params=self.model if len(self.model) else None, max_iter=max_iter, tol=tol)
        end = time.time()
        print()
        print(f"done ranking, took {end-start:0.3f}, {iters} iters, last update norm: {e} (/param = {t})")
        #if iters == max_iter-1:
        #    return True
        return False
        #print(self.clamp(self.model - prev))

def getidx(self, val):
    from web.util import nanguard
    return nanguard(numpy.searchsorted(self.sorted_model, val))
def getval(self, h):
    from web.util import nanguard
    id = self.getid(h)
    if id >= len(self.model):
        return None
    return nanguard(self.model[id])

def calc_next_index(self, h, vals, weights, dists_out, debug=False, force=False, existing_val=None):
    from web.util import nanguard
    from web import opts
    modelmin = self.min()
    modelmax = self.max()
    low_, high_ = self.softmin(vals[0], inv=True), self.softmin(vals[1])
    low = max(low_, -modelmax)
    high = min(high_, modelmax)
    dists_out[h] = high - low

    widx, lidx = self.getidx(low), self.getidx(high)
    mid = existing_val
    if mid is None:
        mid = (low + high)/2
    midx = self.getidx(mid)
    pos = nanguard((high-modelmin)/(modelmax-modelmin))
    prec = opts.precision_func(pos)
    delta = nanguard(len(self.model)/prec) * 2
    #if low < modelmin: widx -= max((delta - 2), 1)
    #if high > modelmax: lidx += max((delta - 2), 1)
    #if debug: print(pos, delta, low, high, widx, lidx)
    seen_enough = len(vals[0]) > opts.min_clean_wins or len(vals[0]) + len(vals[1]) > opts.min_clean_compares
    finished_enough = max(0, lidx - widx) < delta
    is_goat = widx >= nanguard(len(self.model) - opts.goat_window)
    info = {
        "finished_enough": finished_enough,
        "seen_enough": seen_enough,
        "is_goat": is_goat,
        "wlen": len(vals[0]),
        "llen": len(vals[1]),
        "prec": prec,
        "midx": midx,
        "lidx": lidx,
        "widx": widx,
        "mid": mid,
        "pos": pos,
        "delta": delta,
        "adelta": max(0, lidx-widx),
        "high": high,
        "low": low,
    }
    if len(vals[0]) > 0 and (is_goat or len(vals[1]) > 1) and seen_enough and finished_enough and not force:
        return None,info
    return midx,info

def weighted_softmin(self, a, weights, inv=False):
    from web.util import softmax
    from web import opts
    a = numpy.array(a)
    if inv: a=-a
    sm = softmax(-numpy.array(a) * opts.weighted_softmin_sharpness)
    sm2 = sm * weights #numpy.exp(-b/1000)
    av = numpy.average(a, weights=sm2)
    return av

def softmin(self, directional_distances, inv=False):
    from web.util import nanguard
    from web import opts
    if not len(directional_distances):
        return 100
    vals = nanguard(numpy.array(directional_distances)) #numpy.log2(numpy.maximum(directional_distances*10, 0.0001))
    if inv: vals=-vals
    weight = nanguard(numpy.minimum(opts.softmin_falloff_per_unit ** -vals, 2000000))
    sum = numpy.sum(weight * vals)
    total_weight = numpy.sum(weight)
    res = nanguard(sum / total_weight)
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
    from web.util import nanguard
    from web import opts
    if pair[0] not in self.sorted_ids or pair[1] not in self.sorted_ids:
        print("unknown", pair)
        return 0.5, False, {}
    valwin = self.getval(pair[0])
    vallose = self.getval(pair[1])
    rel_wins = stats.pair_wins.get(pair, [0, 0])
    o = 0.1
    win_ratio = (rel_wins[0]+o) / (rel_wins[0] + rel_wins[1] + o*2)
    if rel_wins[0] < rel_wins[1]:
        valwin, vallose = vallose, valwin
        win_ratio = 1-win_ratio
    if valwin > vallose:
        return 101.0, False, {}
    if sum(rel_wins) >= opts.inversion_max:
        #print("inverted", sum(rel_wins))
        return 505, False, {}
    center = (valwin+vallose)/2
    logit = abs(numpy.log(win_ratio/(1-win_ratio)))
    distance = (valwin - center) * logit

    idxlose, idxwin = self.getidx(center-distance), self.getidx(center+distance)

    modelmin = self.min()
    modelmax = self.max()
    midx = self.getidx(center)
    pos = (center-modelmin)/(modelmax-modelmin)
    prec = opts.inversion_precision_func(pos)
    delta = len(self.model)/prec
    actual_delta = max(0, idxlose - idxwin)
    ratio = actual_delta / (delta*2)
    badly_inverted = 0 if ratio < 1 else ratio
    #print(f"idxlose: {idxlose}, idxwin: {idxwin}, prec: {prec}, delta*2: {delta*2}, vallose: {vallose}, valwin: {valwin}, center: {center}, logit: {logit}, badly_inverted: {badly_inverted}")
    return 123, badly_inverted, {"midx": midx, "pos": pos, "prec": prec, "delta": delta, "actual_delta": actual_delta, "ratio": ratio, "badly_inverted": badly_inverted}

def calculate_nearest_neighborhood(self, stats, hashes_to_debug, extra=False, save=True):
    from web import opts
    from web.util import timing
    with timing("calculate_nearest_neighborhood", 0.1):
        distances = {h: ([-50],[50]) for h in self.all_items}
        weights   = {h: ([1],[1]) for h in self.all_items}
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
            win_prob, inverted, details = self.check_inversion(stats, pair)
            if win_prob < 0.5:
                win_inversions, _ = inversions.setdefault(win, ([],[]))
                _, loss_inversions = inversions.setdefault(loss, ([],[]))
                win_inversions.append((pair, win_ratio, win_prob, count))
                loss_inversions.append((pair, win_ratio, win_prob, count))
            if inverted:
                inversion_pool.add(pair[0])
                inversion_pool.add(pair[1])
                iv[pair] = inverted
            decayed_ratio = (((rel_wins[0] + 1) / (rel_wins[0] + rel_wins[1] + 2)) - 0.5) * 2

            if win_ratio < opts.ambiguity_threshold:
                # don't include ambiguous comparisons when tallying distances
                # should reduce risk of getting in tangles
                continue
            wval = self.getval(win)
            lval = self.getval(loss)
            if wval is None or lval is None: continue
            dist = wval - lval
            distances[win][0].append(lval)
            distances[loss][1].append(wval)
            weights[win][0].append(decayed_ratio)
            weights[loss][1].append(decayed_ratio)
        modelmin = self.min()
        modelmax = self.max()
        for h, vals in distances.items():
            if self.is_dropped(stats, h): continue
            nextidx, _ = self.calc_next_index(h, vals, weights.get(h), dists, existing_val=self.getval(h))
            if (nextidx is not None or not stats.win_counts.get(h)) and h in self.bh2:
                sp[h] = True
        print("len(sp)", len(sp), "len(distances)", len(distances), len(self.bh2), len([x for x in self.all_items if not stats.win_counts.get(x)]), len([x for x in self.all_items if not stats.win_counts.get(x) and not self.is_dropped(stats, x)]),)

        #q = []
        if True:
            for h in hashes_to_debug:
                hidx = self.sorted_ids.get(h, -1)
                in_pool = h in sp
                in_inv_pool = h in inversion_pool
                #if not in_pool or in_inv_pool: continue
                dists = distances[h]
                win_inversions, loss_inversions = inversions.get(h, ([], []))
                wdist, ldist = self.weighted_softmin(*dists[0]), self.weighted_softmin(*dists[1])
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
def update_new_pool(self, stats):
    from web.util import timing, softmax
    with timing("update_new_pool", 0.1):
        af = self.af2
        bh = self.bh2
        ds={}
        for h, v in zip(self.all_items, self.model):
            i = bh.get(h)
            if i is None: continue
            d=i[1]
            dp = [d]
            if ":" in h:
                dp.append(h)
            #u = self.distances.get(h, 0.5)
            for slice in range(0, max(1, len(dp))):
                ds.setdefault("/".join(dp[:slice+1]), []).append(v)
        for h in stats.dislike.keys():
            i = bh.get(h)
            if i is None: continue
            d=i[1]
            dp = [d]
            if ":" in h:
                dp.append(h)
            for slice in range(0, max(1,len(dp))):
                ds.setdefault("/".join(dp[:slice+1]), []).append(-3)
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
            est = 0
            #stddev = 4
            #var = stddev ** 2
            for slice in range(1, max(2,len(dp))):
                dn="/".join(dp[:slice+1])
                if dn not in ds: break
                if not ds[dn]:
                    continue
                if dn not in cc:
                    #vs = numpy.array(ds[dn])
                    #smean, std = numpy.mean(vs, axis=0)
                    ##_, var_ = self.update_bayes(0, 1, len(vs), smean, std ** 2)
                    #mean, var = self.update_bayes(mean, var, len(vs), smean,max(std**2, numpy.var(vs[:,0]))) #var_ * len(vs))
                    cc[dn] = sum(ds[dn]) / (len(ds[dn]) ** 0.5), sum(ds[dn])/len(ds[dn]) #mean, var * len(vs)
                    #print(dn, "smean,std:", smean, std, "mean:", mean, "std:", numpy.sqrt(var), "p_std:", numpy.sqrt(var*len(vs)), "n:", len(vs))
                est, mean = cc[dn]
            #stddev = numpy.sqrt(var)
            new.append(est/100.0)
            newh.append(h)
            newp[h] = mean
        new = numpy.array(new)
        sorted_ = numpy.argsort(new)[-1000:]
        self.new = softmax(new[sorted_])
        self.newh = numpy.array(newh)[sorted_].tolist()
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

def calculate_dists(self, comparisons):
    stats, comparisons = comparisons
    dists = ([-50], [50])
    weights = ([1], [1])
    for other_hash, wins in comparisons.items():
        other_val = self.getval(other_hash)
        if other_val is None:
            continue
        if self.is_dropped(stats, other_hash):
            continue
        if wins[0] > wins[1]:
            decayed_ratio = ((wins[0]/ max(1e-10, wins[0] + wins[1])) - 0.5) * 2
            dists[0].append(other_val)
            weights[0].append(decayed_ratio)
        elif wins[0] < wins[1]:
            decayed_ratio = (((wins[1]) / max(1e-10, wins[0] + wins[1])) - 0.5) * 2
            dists[1].append(other_val)
            weights[1].append(decayed_ratio)
        else:
            continue
    return dists, weights

def slow_calculations(self, stats, hashes_to_debug, extra=False):
    more = self.calculate_ranking(stats, extra)
    self.calculate_nearest_neighborhood(stats, hashes_to_debug, extra)
    self.update_new_pool(stats)
    return more

def __repr__(self):
    return f"Model(all_items={self.all_items}, model={self.model.tolist()}, searching_pool={self.searching_pool}, inversions={self.inversions})"
