import time
import hashlib
import itertools

def record_win(self, winning, losing, decay, separation):
    from web.util import sigmoid, nanguard, as_pair
    #w_ratio, w_incl = self.update_ratios(winning["hash"])
    #l_ratio, l_incl = self.update_ratios(losing["hash"])
    ratio = sigmoid(separation)
    decay2 = separation ** 0.25
    winning_val = decay*ratio*decay2
    losing_val = decay*(1-ratio)*decay2
    self.win_counts[winning["hash"]] = self.win_counts.get(winning["hash"], 0) + winning_val
    self.loss_counts[losing["hash"]] = self.loss_counts.get(losing["hash"], 0) + winning_val
    self.win_counts[losing["hash"]] = self.win_counts.get(losing["hash"], 0) + losing_val
    self.loss_counts[winning["hash"]] = self.loss_counts.get(winning["hash"], 0) + losing_val
    pair, values = as_pair(winning, losing, extras=(winning_val, losing_val))

    if pair in self.pair_wins:
        w1, w2 = self.pair_wins[pair]
        n1, n2 = values
        values = w1+n1, w2+n2
    self.pair_wins[pair] = values

    self.comparisons.setdefault(pair[0], {})[pair[1]] = values
    self.comparisons.setdefault(pair[1], {})[pair[0]] = values[::-1]
    #print(f'incremented winner win count to {self.win_counts[winning["hash"]]}, loser lose count to {self.loss_counts[losing["hash"]]} (flc: {self.filtered_loss_counts.get(losing["hash"], None)}; w: {w_ratio:0.2f}, {w_incl}; l: {l_ratio:0.2f}, {l_incl})')

def record_similar(self, sim1, sim2, notsim, decay, separation, invert):
    from web.util import sigmoid, nanguard, as_pair
    
    #w_ratio, w_incl = self.update_ratios(winning["hash"])
    #l_ratio, l_incl = self.update_ratios(losing["hash"])
    ratio = sigmoid(separation)
    decay2 = separation ** 0.25
    diff_val = decay*ratio*decay2
    sim_val = (decay*(1-ratio)*decay2) / 2
    #self.sim_counts[sim1["hash"]] = self.win_counts.get(sim1["hash"], 0) + winning_val
    #self.asim_counts[notsim["hash"]] = self.loss_counts.get(losing["hash"], 0) + winning_val

    #self.sim_counts[asim["hash"]] = self.sim_counts.get(asim["hash"], 0) + asim_val
    #self.asim_counts[simning["hash"]] = self.asim_counts.get(simning["hash"], 0) + asim_val
    # TODO: could be better. should really be working with edges
    pair, values = as_pair(sim1, sim2, notsim, extras=((sim_val, sim_val, diff_val) if not invert else (diff_val, diff_val, sim_val)))

    if pair in self.triplet_diffs:
        w1, w2, w3 = self.triplet_diffs[pair]
        n1, n2, n3 = values
        values = w1+n1, w2+n2, w3+n3
    self.triplet_diffs[pair] = values

    for ia,ib,ic in itertools.permutations(list(range(3)), 3):
        self.similarity.setdefault(pair[ia], {}).setdefault(pair[ib], {})[pair[ic]] = (values[ia], values[ib], values[ic])
    #self.comparisons.setdefault(pair[1], {})[pair[0]] = values[::-1]
    #print(f'incremented winner win count to {self.win_counts[winning["hash"]]}, loser lose count to {self.loss_counts[losing["hash"]]} (flc: {self.filtered_loss_counts.get(losing["hash"], None)}; w: {w_ratio:0.2f}, {w_incl}; l: {l_ratio:0.2f}, {l_incl})')

def update(self, item, initial=False):
    from web.util import nanguard, as_pair, sigmoid
    from web import opts
    age = time.time() - (nanguard(item.get("viewend", 0), "update.viewend") / 1000)
    decay = nanguard(opts.comparison_decay_func(age))
    item["dur"] = nanguard(item.get("viewend", 0)-item.get("viewstart", 0), "update.dur")
    mag_decay = nanguard(max(min(opts.initial_mag, opts.initial_mag/ max(1, (item["dur"] / 1000))), opts.min_mag))
    if nanguard(item.get("dur",0))<opts.minview and not item.get("fast"):
        print("\033[31mskipped due to too-low view duration", item, "\033[m")
        return
    if not initial:
        print(f"\033[38mvd: {item.get('dur',0)}, mag_decay: {mag_decay}, age: {age}, time_decay: {decay}\033[m")
    pair = as_pair(*item["items"])
    if "similarity" in item:
        info = item["similarity"]
        winner = None
        least_similar = info.get("least_similar", None)
        most_similar = info.get("most_similar", None)
        if type(most_similar) == list:
            least_similar = list(set(range(3)) - set(most_similar))[0]
            most_similar = None
    elif type(item.get("preference", None)) != dict:
        info = {}
        winner = nanguard(item.get("preference", 1) - 1)
        least_similar = None
    else:
        info = item.get("preference", {})
        winner = nanguard(info.get("prefer", 1) - 1)
        least_similar = None

    too_close = info.get("too_close", False)
    incomparable = info.get("incomparable", False)
    dislike = info.get("dislike", None)
    strong = info.get("strong", None)

    if type(item.get("info")) == dict and item["info"].get("t") == ["inversions", "inversions"]:
        mag_decay = mag_decay * opts.inversion_compare_boost + sum(self.pair_wins.get(pair, [0, 0])) * opts.inversion_compare_relboost * mag_decay
    if not dislike:
        dislike = [0] * len(item["items"])
    for f, dis in zip(item["items"], dislike):
        if dis:
            self.dislike[f["hash"]] = True
            #print("dislike",f)
        elif f["hash"] in self.dislike:
            del self.dislike[f["hash"]]
            #print("undislike",f)
    if any(dislike): return

    if too_close:
        self.too_close[pair] = self.too_close.get(pair, 0) + nanguard(2 * decay * (1-sigmoid(mag_decay)))
        #self.record_win(*item["items"])
        #self.record_win(*item["items"][::-1])
    elif incomparable:
        pair = as_pair(*item["items"], strip=True)
        self.incomparable_pairs[pair] = self.incomparable_pairs.get(pair, 0) + nanguard(decay/mag_decay)
    elif winner is not None:
        winning = item["items"][winner]
        losing = item["items"][1-winner]
        self.record_win(winning, losing, nanguard(decay), nanguard(mag_decay))
    elif least_similar is not None or most_similar is not None:
        sim = least_similar if least_similar is not None else most_similar
        assert 0 <= sim <= 2
        s1, s2 = [x for i, x in enumerate(item["items"]) if i != sim]
        s3 = item["items"][sim]
        self.record_similar(s1, s2, s3, nanguard(decay), nanguard(mag_decay), invert=(most_similar is not None))

def from_history(self, history):
    from web.util import nanguard
    from web import opts
    keep = []
    wts = False
    for idx, item in enumerate(history):
        if not "items" in item:
            continue
        if "parent" in item:
            path = "/" + item["parent"] + "/" + item["name"]
            if not (item["hash"] == hashlib.sha256(path.encode("utf-8", ue)).hexdigest()):
                raise Exception()
        item["dur"] = nanguard(item.get("viewend", 0), f"from_history#{idx}.viewend")-nanguard(item.get("viewstart", 0), f"from_history#{idx}.viewstart")
        if nanguard(item.get("dur",0))<opts.minview:
            if not wts and keep:
                keep.pop()
            wts=True
            continue
        wts = False
        keep.append(item)
    for item in keep:
        self.update(item, initial=True)
