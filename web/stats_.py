import time
import hashlib

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
    if type(item.get("preference", None)) != dict:
        winner = nanguard(item.get("preference", 1) - 1)
        too_close = False
        incomparable = False
        dislike = None
        strong = False
    else:
        info = item.get("preference", {})
        winner = nanguard(info.get("prefer", 1) - 1)
        too_close = info.get("too_close", False)
        incomparable = info.get("incomparable", False)
        dislike = info.get("dislike", None)
        strong = info.get("strong", None)
    if type(item.get("info")) == dict and item["info"].get("t") == ["inversions", "inversions"]:
        mag_decay = mag_decay * opts.inversion_compare_boost + sum(self.pair_wins.get(pair, [0, 0])) * opts.inversion_compare_relboost * mag_decay
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
        self.too_close[pair] = self.too_close.get(pair, 0) + nanguard(2 * decay * (1-sigmoid(mag_decay)))
        #self.record_win(*item["items"])
        #self.record_win(*item["items"][::-1])
    elif incomparable:
        pair = as_pair(*item["items"], strip=True)
        self.incomparable_pairs[pair] = self.incomparable_pairs.get(pair, 0) + nanguard(decay/mag_decay)
    else:
        winning = item["items"][winner]
        losing = item["items"][1-winner]
        self.record_win(winning, losing, nanguard(decay), nanguard(mag_decay))

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
