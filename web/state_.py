import numpy
import random
import json
import sys
import os
import subprocess
import msgpack
import time
def rand_video_fragments(f, existing_hash=None, num_samples=None):
    from web import opts
    from web.util import nanguard, timing
    from web.files import duration, extract_time
    with timing("rand_video_fragments", 0.1):
        dur = None
        if "video" in f and f["video"] and "min_time" not in f:
            dur = duration(f)
        if dur is None:
            return [f]
        results = []
        if existing_hash is not None:
            h, s, e = existing_hash.split(":")
            s = float(s)
            e = float(e)
            results.append(extract_time(f, s, e))
            print(results)
            return results
        if dur < opts.min_frag_length:
            return [f]
        dc = int(min(nanguard(numpy.sqrt(dur)),nanguard(dur/5)))
        if dc == 0:
            return [f]
        dr = nanguard(dur/dc)
        num_samples = num_samples or int(dc)
        positions = random.sample(list(range(dc)), num_samples)
        for r in positions:
            start = dr * r
            end   = dr * (r+1)
            results.append(extract_time(f, start, end))
        return results

def fh(self, h):
    return self.bh.get(h, {}).get("hash", h)

def read_from_file(self, reader):
    from web.util import timing
    with timing("read_from_file"):
        new_history = []
        for line in reader:
            if not line: continue
            l = json.loads(line)
            if "items" in l:
                for x in l.get("items", []):
                    if type(x["hash"]) == dict:
                        # fix oopsie
                        x["hash"] = x["hash"]["hash"]
                l["items"] = [self.bh.get(item["hash"], item) for item in l["items"]]
            if "current" in l:
                self.current = [x["hash"] for x in l["items"]]
                for x in self.history[-1]["items"]:
                    if x["hash"] in self.seen_allowed:
                        self.seen_allowed.remove(x["hash"])
                continue
            self.history.append(l)
            new_history.append(l)
            if type(self.history[-1].get("preference")) == dict and self.history[-1]["preference"].get("undo"):
                self.history.pop()
                if new_history:
                    new_history.pop()
                continue
            elif type(self.history[-1].get("preference")) == dict and self.history[-1]["preference"].get("not_sure"):
                for x in self.history[-1]["items"]:
                    self.seen_allowed.add(x["hash"])
            if "items" in self.history[-1]:
                try:
                    self.history[-1]["items"] = [self.bh.get(item["hash"], item) for item in self.history[-1]["items"]]
                except:
                    print(self.history[-1])
                    raise
    with timing("from_history"):
        self.stats.from_history(new_history)

def read(self):
    from web.util import timing
    if self.do_update:
        with timing("write affile"):
            try:
                os.makedirs(self.tempdir)
            except FileExistsError:
                pass
            with open(self.affile, "wb") as writer:
                msgpack.pack(self.af2, writer, use_bin_type=True)

    if os.path.exists(self.preffile):
        with open(self.preffile, "r") as reader:
            self.read_from_file(reader)

    with timing("mark history seen"):
        for x in self.history:
            if "items" not in x: continue
            i = x["items"]
            self.seen.mark_seen(i[0]["hash"], x["viewstart"])
            self.seen.mark_seen(i[1]["hash"], x["viewstart"])
    z = numpy.array(list(self.seen.seen_suppression.values()) or [0])
    ss = self.seen.bulk_check_seen()
    print(f"seen_suppression: mean={numpy.mean(z)}, max={numpy.max(z)}, min={numpy.min(z)}, median={numpy.median(z)}, seen={len(ss)}")
    print(f"history: {len(self.history)}")
    if self.do_reap:
        wait = not self.reap_slow(eager=True)
        if self.do_update:
            self.launch_slow(wait=wait)

def geth(self, x):
    if x not in self.bh and x.partition(":")[0] in self.bh:
        print("geth not known", x)
        self.bh[x] = rand_video_fragments(self.bh[x.partition(":")[0]], x)[-1]
    if x not in self.bh:
        return None
    return self.bh[x]

def select_next(self, path):
    from web import opts
    from web.util import timing, as_pair
    with timing("select_next"):
        self.reap_slow()
        #return (
        #    bh[self.model_.sorted_hashes[max(self.getidx(-max(self.model)), 0)]], 
        #    bh[self.model_.sorted_hashes[min(self.getidx(-min(self.model)), len(self.model)-1)]],
        #    "x", "x")
        #print()
        #print(f"begin select_next.")
            
        m = 1
        #capped = self.stats.loss_counts.keys() | set([x for x, y in self.stats.win_counts.items() if y > 4])
        bulk_seen = self.seen.bulk_check_seen(m) & self.seen_allowed
        bulk_seen_l = self.seen.bulk_check_seen(m*0.3)  & self.seen_allowed
        af, _ = self.files.get_all_images()
        print("s:", len(bulk_seen), "m:", len(self.model_.model), "np:", len(self.model_.newh),"ai:", len(self.model_.all_items), "af:", len(self.af2), "ic:", len(af), f"h: {len(self.history)}")
        last_winner = None
        indices = {}
        with timing("get_last"):
            if self.history:
                iters = 0
                skip_winners = (random.random() ** opts.recent_wins_curve) * opts.recent_wins
                skipped = 0
                skipped_hashes = set()
                while last_winner is None:
                    indices = {}
                    iters += 1
                    if iters > opts.recent_wins * 2:
                        break
                    last = self.history[-iters]
                    last_info = last.get("info", None)
                    last_options = last["items"]
                    indices[last_options[0]["hash"]] = -1
                    indices[last_options[1]["hash"]] = 1
                    pref = last.get("preference", None)
                    if type(pref) == dict:
                        pref = pref.get("prefer", None)
                    if pref is not None:
                        skip = opts.pref_skip
                        res = last_options[pref - 1]
                    else:
                        skip = opts.nopref_skip
                        res = random.choice(last_options)
                    if res["hash"] in skipped_hashes:
                        continue
                    skipped += skip
                    skipped_hashes.add(res["hash"])
                    if skipped >= skip_winners:
                        last_winner = res
                        last_winner_type = last_info.get("t", ['unk','unk']) if type(last_info) == dict else ['unk', 'unk']

        first = second = None
        no_vfrag = False

        with timing("loop"):
            for x in range(800):
                with timing("loop head"):
                    first = firstlabel = None
                    second = secondlabel = None

                    base_pool = ((self.model_.searching_pool.keys() | self.extra_pool) - self.removed_pool)
                    randpool_set = base_pool - bulk_seen
                    randpool =  list(randpool_set)
                    print(f"randpool size: {len(randpool)}, inversions: {len(self.model_.inversions)}")
                    prior = None

                    force = x > 790
                    force_neighborhood = x > 400
                    if x % 50 == 49:
                        m = m * 0.5
                        bulk_seen = self.seen.bulk_check_seen(m) & self.seen_allowed
                        bulk_seen_l = self.seen.bulk_check_seen(m*0.3) & self.seen_allowed
                        print(f"decay bulk seen by {m}, new size {len(bulk_seen)}")
                    if opts.fix_inversions or x > 700:
                        inversions = {x: y for x,y in self.model_.inversions.items() if x not in self.fixed_inversions}
                    else:
                        inversions = {}
                with timing("select first"):
                    if len(path) == 1 and self.geth(path[0]):
                        with timing("manual"):
                            first = self.geth(path[0])
                            firstlabel = "manual"
                            force_neighborhood = True
                    elif self.lock is not None:
                        first = self.geth(self.lock)
                        firstlabel = "lock"
                        force_neighborhood = True
                    elif self.current:
                        with timing("current"):
                            firsth, secondh = self.current
                            first = self.geth(firsth)
                            second = self.geth(secondh)
                            if not first or not second:
                                self.current = None
                                continue
                            firstlabel = secondlabel = "current"
                            force = True
                            if self.force_current:
                                self.force_current = False
                                firstlabel = secondlabel = "undo"
                                break
                    elif last_winner and random.random() < (opts.last_winner_last_winner_prob if 'last winner' in last_winner_type else opts.last_winner_prob):
                        with timing("last_winner"):
                            first = last_winner
                            firstlabel = "last winner"
                    elif random.random() < opts.high_quality_ratio and len(self.model_.sorted_hashes):
                        with timing("high_quality"):
                            hidx = max(min(len(self.model_.sorted_hashes)-1, int((random.random() ** (1/10)) * len(self.model_.sorted_hashes))), 0)
                            h = self.model_.sorted_hashes[hidx]
                            if h in bulk_seen:
                                print("WARNING: fail on recently seen high quality item")
                                continue
                            first = self.geth(h)
                            if first is None:
                                print("WARNING: fail on missing high quality item")
                                continue
                            firstlabel = f"high-quality ({hidx}/{len(self.model_.sorted_hashes)}):"
                    elif (random.random() > len(randpool) / opts.explore_target and len(self.model_.newh)):
                        with timing("newh"):
                            firsth = numpy.random.choice(self.model_.newh, p=self.model_.new)
                            prior = self.model_.newp[firsth]
                            first = self.geth(firsth)
                            assert first
                            firstlabel = "explore"
                    elif random.randrange(0, int(len(randpool) + min(opts.inversion_max_count, len(inversions))/opts.inversion_ratio)) < len(randpool):
                        with timing("randpool"):
                            h = random.choice(randpool)
                            first = self.geth(h)
                            if not first:
                                print("WARNING: fail on missing randpool item")
                                continue
                            firstlabel = f"randpool (lc: {self.stats.loss_counts.get(h)})"
                    else:
                        with timing("inversions"):
                            invs = sorted(list(inversions.items()), key=lambda x: -x[1])[:opts.inversion_max_count]
                            for x in range(10):
                                idx = max(0, min(len(invs)-1, int((random.random() ** 4) * len(invs)) ))
                                pair=invs[idx][0]
                                if pair[0] in bulk_seen or pair[1] in bulk_seen:
                                    self.fixed_inversions.add(pair)
                                    continue
                                if not self.geth(pair[0]) or not self.geth(pair[1]):
                                    continue
                                if not self.model_.check_inversion(self.stats, pair)[1]:
                                    continue
                                break
                            else:
                                print("WARNING: pulled 10 items from inversions pool, all too close to bother with")
                                continue
                            if random.random() < opts.inversion_neighborhood:
                                first = self.geth(random.sample(pair,2)[0])

                                #force_neighborhood = True
                            else:
                                first, second = self.geth(pair[0]), self.geth(pair[1])
                                
                            firstlabel = secondlabel = "inversions"
                            no_vfrag = True
                if not no_vfrag:
                    first, = rand_video_fragments(first, num_samples=1)

                with timing("select second"):
                    if second is None:
                        firsthash = first["hash"]
                        if prior is not None:
                            idx = self.model_.getidx(prior)
                        else:
                            comparisons = self.stats.comparisons.get(firsthash, {})
                            dists, weights = self.model_.calculate_dists((self.stats, comparisons))
                            idx, info = self.model_.calc_next_index(firsthash, dists, weights, {}, debug=True, force=force_neighborhood, existing_val=self.model_.getval(firsthash))
                            #print("calc_next_index info:", firsthash, dists, info, idx, force_neighborhood)
                            if idx is None:
                                #print("done")
                                #print(f"attempted neighborhood on {firsthash}, but next index None, marking done")
                                self.removed_pool.add(firsthash)
                                continue
                                #idx = self.model_.sorted_ids[firsthash]
                        nbs = max(opts.max_neighborhood,int(numpy.random.lognormal(0, 1)*len(self.model_.model)/opts.neighborhood_func(idx / len(self.model_.model))))
                        #print("nbs", nbs)

                        zone=min(len(self.model_.sorted_hashes)-(1+nbs), max(nbs, idx))
                        sss = (list(self.model_.sorted_hashes[max(0, zone-nbs):zone+nbs]))
                        if not len(sss):
                            print("WARNING: fail on empty slice of sorted hashes")
                            continue
                        for x in range(10):
                            try:
                                h = random.choice(sss)
                            except IndexError:
                                print(x, sss, firsthash, nbs, zone, len(self.model_.sorted_hashes), prior, self.model_.sorted_hashes[zone-nbs:zone+nbs], self.model_.sorted_hashes[:nbs])
                                raise
                            if h == firsthash:
                                continue
                            if h not in bulk_seen_l:
                                break
                            nbs += 10*(x+1)
                            sss = (list(self.model_.sorted_hashes[max(0, zone-nbs):zone+nbs]))
                        else:
                            print("WARNING: fail on seen all of neighborhood too recently")
                            continue

                        second = self.geth(h)
                        if not second:
                            print("WARNING: fail on missing neighborhood", h)
                            continue
                        secondlabel = f"neighborhood ({idx}->{self.model_.sorted_ids[h]})"
                        if force_neighborhood:
                            secondlabel += " force_neighborhood"
                if not no_vfrag:
                    second, = rand_video_fragments(second, num_samples=1)
                #print("result:")
                #print(firstlabel)
                #print(secondlabel)

                with timing("checks"):
                    if not force:
                        if self.model_.is_dropped(self.stats, first["hash"]) or self.model_.is_dropped(self.stats, second["hash"]):
                            print("WARNING: one of the options was dropped, resampling")
                            continue
                    if as_pair(first, second, strip=True) in self.stats.incomparable_pairs:
                        print("WARNING: got an incomparable pair, resampling")
                        continue
                    if as_pair(first,second) in self.stats.too_close:
                        print("WARNING: pair marked too close, resampling")
                        continue
                    if first is None or second is None:
                        print("WARNING: first or second is None", first, second)
                        continue
                    if first.get("hash") == second.get("hash"):
                        print("WARNING: same val for both")
                        continue
                break
            else:
                raise Exception(f"Failed to find a pair. {len(self.history)}, {len(self.model_.model)}, {len(randpool)}, {len(self.model_.searching_pool)}, {last_winner}, {last_winner['hash'] in base_pool}")

        with timing("after loop"):
            proba, probb = self.getinfo(first, second, bulk_seen=bulk_seen)
            proba.append(firstlabel)
            probb.append(secondlabel)
            paired_up = list(zip([first, second], [proba, probb], [random.random()-0.5, 0]))
            paired_up = sorted(paired_up, key=lambda x: indices.get(x[0]["hash"], x[2]))
            pa, pb = paired_up
            a, proba, _ = pa
            b, probb, _ = pb
            #print("1:", proba[-1])
            #print("2:", probb[-1])
            self.seen.mark_seen(a["hash"])
            self.seen.mark_seen(b["hash"])
            if a["hash"] in self.seen_allowed:
                self.seen_allowed.remove(a["hash"])
            if b["hash"] in self.seen_allowed:
                self.seen_allowed.remove(b["hash"])
            if a["hash"] not in self.model_.sorted_ids:
                self.extra_pool.add(a["hash"])
            if b["hash"] not in self.model_.sorted_ids:
                self.extra_pool.add(b["hash"])
            sys.stdout.flush()
            self.current = a["hash"], b["hash"]
        with timing("save view"):
            if self.do_update:
                with open(self.preffile, "a") as appender:
                    appender.write(json.dumps({
                        "current": time.time(),
                        "items": [{"hash": a["hash"]}, {"hash": b["hash"]}],
                    }) + "\n")
        info = {"t": [firstlabel, secondlabel], "i": [proba, probb]}
        return a, b, proba, probb, info

def select_next_similarity(self, path):
    pass

    #for d, vs in sorted(list(ds.items())):
    #    print(f"{d}: {len(vs):6d} samples, {numpy.std(vs):6.3f} std, {numpy.mean(vs):6.3f} mean, {min(vs):6.3f} min, {max(vs):6.3f} max, {numpy.median(vs):6.3f} median")
def getinfo(self, *hs, bulk_seen=None, do_seen=True, pools=True, details=True, debugpools=True):
    from web.util import as_pair, nanguard
    from web import opts
    hs = [x.get("hash") if type(x) == dict else x for x in hs]
    pools_=pools
    result = []
    if bulk_seen is None:
        bulk_seen = set()
        if pools_ and do_seen:
            bulk_seen = self.seen.bulk_check_seen()
    for x in hs:
        ires = []
        result.append(ires)
        pools = []
        if debugpools and x in self.removed_pool: pools.append("state.removed")
        if debugpools and x in self.extra_pool: pools.append("state.extra")
        if details and x in self.model_.searching_pool: pools.append("searching")
        if debugpools and x in self.model_.ids: pools.append("model.all_items")
        if debugpools and x in self.stats.comparisons: pools.append("stats")
        if details and x in bulk_seen: pools.append("bulk_seen")
        if x in self.stats.dislike: pools.append("BLACKLISTED")
        elif self.model_.is_dropped(self.stats, x): pools.append("DROPPED")
        msg = ""
        if x in self.model_.newp:
            if debugpools: pools.append("model.newp")
            msg += f"prior: {self.model_.newp[x]:0.2f}"
        if x in self.model_.sorted_ids:
            id = self.model_.sorted_ids[x]
            if debugpools: pools.append("model.sorted")
            msg += f"#{id}, val: {self.model_.sorted_model[id]:0.2f}"
        if x in self.model_.distances:
            msg += f", dists: {self.model_.distances[x]:0.2f}"
        if msg:
            ires.append(msg)
        if details:
            ires.append(f"item: +{self.stats.win_counts.get(x, 0):0.2f}, -{self.stats.loss_counts.get(x, 0):0.2f}")
        if do_seen:
            comparisons = self.stats.comparisons.get(x, {})
            dists, weights = self.model_.calculate_dists((self.stats, comparisons))
            idx, info = self.model_.calc_next_index(x, dists, weights, {}, debug=True, existing_val=self.model_.getval(x))
            ires.append(f"nextidx: {idx}, {int(info['finished_enough'])},{int(info['seen_enough'])},{int(info['is_goat'])},{info['prec']:0.1f},{info['delta']:0.0f},{info['adelta']:0.0f}")
            #pools.append("model.distances")
        if pools_ and pools:
            ires.append(f"{', '.join(pools)}")

    if len(hs) == 2:
        a, b = hs
        pair = as_pair(a, b)
        pair_s = as_pair(a, b, strip=True)
        ires = []
        pools = []
        if pools_:
            if debugpools and pair in self.fixed_inversions: pools.append("state")
            if debugpools and pair in self.inversion_fixes: pools.append("state.inversion_fixes")
            if pair in self.model_.inversions: pools.append("INVERSION")
            if pair in self.stats.too_close:
                #pools.append(f"stats.too_close")
                ires.append(f"TOO CLOSE x{self.stats.too_close[pair]:0.2f}")
            if pair_s in self.stats.incomparable_pairs: pools.append("NON-COMPARABLE")
        if pools and pools_:
            ires.append(f"{', '.join(pools)}")
        rel_wins = self.stats.comparisons.get(a, {}).get(b, (0, 0))
        if pair in self.stats.too_close:
            rel_wins = tuple([nanguard(x + sum(rel_wins) + opts.too_close_boost * self.stats.too_close[pair]) for x in rel_wins])
        wins_a, wins_b = rel_wins
        proba, probb = self.model_.getprob(a, b)
        if ires:
            for x in result:
                x.append("---")
                x.extend(ires)
        result[0].append(f"{100*(wins_a/(max(wins_b+wins_a, 0.0001))):0.1f}% ratio ({wins_a:0.2f}/{wins_b:0.2f}), model={100*proba:.1f}%, ")
        result[1].append(f"{100*(wins_b/(max(wins_a+wins_b, 0.0001))):0.1f}% ratio ({wins_b:0.2f}/{wins_a:0.2f}), model={100*probb:.1f}%, ")

    return result

def reap_slow(self, wait=False, eager=False):
    from web.util import timing
    from web.state import Model
    with timing("reap_slow"):
        checkfile = self.outputfile if eager else self.readyfile
        read_needed = self.read_needed
        if self.subproc is not None and self.subproc.poll() is None and not os.path.exists(checkfile):
            with timing("wait"):
                read_needed = True
                if wait:
                    while not os.path.exists(checkfile):
                        # TODO: less dumb thing than this
                        time.sleep(0.1)
                else:
                    return False
        if self.subproc and self.subproc.poll() is not None:
            self.subproc = None
            
        with timing("check_and_load"):
            if os.path.exists(checkfile):
                print("\033[31mLOADING\033[m")
                with timing("load"):
                    if os.path.exists(self.readyfile) and self.do_update:
                        os.unlink(self.readyfile)
                    # have one waiting to read
                    with timing("read"):
                        with open(self.outputfile, "rb") as reader:
                            packed_model = msgpack.unpack(reader, use_list=False, raw=False)
                    with timing("model()"):
                        old_m = self.model_.model
                        self.model_ = Model(*packed_model)
                        new_m = self.model_.model
                        min_l = min(len(old_m), len(new_m))
                        delta = numpy.sum(numpy.abs(old_m[:min_l] - new_m[:min_l]))
                        print("reap model calculation, delta:", delta)
                    with timing("fixh"):
                        self.model_.fixh(self.fh)
                    self.removed_pool = set()
                    self.extra_pool = set()
                    self.inversion_fixes = {}
                    self.fixed_inversions = set()
                    self.dirty = set()
                    self.read_needed = False
                    return True
        return False

def launch_slow(self, wait=False):
    from web.util import timing
    with timing("launch_slow"):
        if not self.do_update:
            raise Exception()
        with timing("to_msgpack (dicts)"):
            packed_stats = self.stats.to_msgpack()
            packed_model = self.model_.to_msgpack(include_derived=False)
        with timing("write"):
            self.read_needed = True
            with open(self.inputfile, "wb") as writer:
                msgpack.pack([packed_stats, packed_model], writer, use_bin_type=True)
        if os.path.exists(self.completionfile):
            os.unlink(self.completionfile)

        with timing("launch"):
            if self.subproc is None:
                args = [sys.executable, "-m", "web", self.files.base, self.preffile, self.outputfile, self.completionfile, self.readyfile]
                print(" ".join(args))
                self.subproc = subprocess.Popen(args, stdin=open(os.devnull, "rb"))
        if wait:
            with timing("wait"):
                self.reap_slow(True)

def update(self, info, file1, file2, file3=None):
    from web.util import timing, as_pair
    from web.state import Stats
    with timing("update"):
        if not self.do_update:
            raise Exception
        with timing("save"):
            with open(self.preffile, "a") as appender:
                a = dict(info)
                a["items"] = [self.geth(file1) or {"hash": file1}, self.geth(file2) or {"hash": file2}] + ([self.geth(file3) or {"hash": file3}] if file3 else [])
                appender.write(json.dumps(a) + "\n")
        pair = as_pair(file1, file2, file3)
        if "similarity" in a:
            self.history.append(a)
            self.stats.update(a)
            return
        elif "lock" in a.get("preference",{}):
            self.lock = a.get("preference",{}).get("lock")
        elif a.get("preference",{}).get("undo"):
            with timing("undo"):
                prev = self.history.pop()["items"]
                self.current = [x["hash"] for x in prev]
                self.force_current = True
                self.stats = Stats()
                self.stats.from_history(self.history)
            self.launch_slow()
        elif a.get("preference",{}).get("not_sure"):
            self.seen_allowed.add(file1)
            self.seen_allowed.add(file2)
            self.current = None
        else:
            with timing("add to history"):
                self.dirty.add(pair[0])
                self.dirty.add(pair[1])
                self.history.append(a)
                self.stats.update(a)
                self.current = None
                if pair in self.model_.inversions:
                    self.inversion_fixes[pair] = self.inversion_fixes.get(pair, 0) + 1
                    if self.inversion_fixes[pair] > 2 or not self.model_.check_inversion(self.stats, pair)[1]:
                        self.fixed_inversions.add(pair)
            self.launch_slow()
