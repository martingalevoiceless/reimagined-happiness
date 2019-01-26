import hashlib
import csv
import random
import sys
import subprocess
import json
import collections
import stat
import time
from .util import timing
import json
try:
    from winmagic import magic
except ImportError:
    import magic
import msgpack
import os
ue = "surrogatepass"

class files_opts:
    # fixme: "video/x-flv", also wmv, etc, mpg, blah blah
    image_magics = ["image/jpeg", "image/gif", "image/png", "video/mp4", "image/webp", "video/webm", "video/x-m4v", "video/quicktime"]
    excluded_names = ["var", "usr", "etc", "lib", "opt", "dev", ".thumbnails", "app", "web", "js", ".git", "Facebook", "Android"]


def duration(_json):
    if "video" in _json:
        if _json["video"] is None:
            return None
        _json=_json["video"]

    if 'format' in _json:
        if 'duration' in _json['format']:
            return float(_json['format']['duration'])

    if 'streams' in _json:
        # commonly stream 0 is the video
        for s in _json['streams']:
            if 'duration' in s:
                return float(s['duration'])
    return None

def extract_time(f, start, stop):
    f = dict(f)
    f["min_time"] = start
    f["max_time"] = stop
    f["hash"] = hashlib.sha256(f["path"].encode("utf-8", ue)).hexdigest() + f":{start:.2f}:{stop:.2f}"
    return f

class FilesCache:
    def __init__(self, base, cache=True, save=True, cache2=False):
        self.save = save
        self.base = base
        self.results = {}
        self.magics = {}
        self.videos = {}
        self.vcode="5"
        self.results_from_cache = False
        loaded_results = None
        self.all_images = []
        self.by_hash = {}

        if cache2:
            with timing("read images"):
                try:
                    with open("all_images.msgpack", "rb") as r:
                        self.all_images = msgpack.unpack(r, raw=False, unicode_errors="surrogatepass")
                    for x in self.all_images:
                        self.by_hash[x["hash"]] = x
                        for y in x.get("other_hashes", ()):
                            self.by_hash[y] = x
                except FileNotFoundError:
                    pass
        with timing("read allfiles"):
            try:
                with open('allfiles', 'r') as reader:
                    loaded_results = json.loads(reader.read())
            except FileNotFoundError:
                pass
        with timing("transform allfiles"):
            if loaded_results is not None:
                if cache:
                    self.results = loaded_results
                    self.results_from_cache = True
                for x in loaded_results.values():
                    if "hash" in x and "magic" in x:
                        self.magics[x["hash"]] = x["magic"]
                    if "video" in x:
                        vinfo = {"video": x["video"]}
                        if "video_err" in x:
                            vinfo["video_err"] = x["video_err"]
                        self.videos[x["hash"]] = vinfo
        self.entries = []
        self.entries_from_cache = False
        if cache:
            with timing("allfile_e_c read"):
                try:
                    with open('allfiles_e_c'+self.vcode, 'r') as reader:
                        self.entries = json.loads(reader.read())
                        self.entries_from_cache = True
                except FileNotFoundError:
                    pass

        self.allfiles = []
        self.get_all_images()

    def check(self, key):
        if any((x in files_opts.excluded_names) for x in key.split("/")):
            return False
        return True

    def findall(self):
        if self.entries:
            return self.entries
        print("loading scan of directory...")
        with timing("os.walk"):
            self.entries = []
            for root, dirs, files in os.walk(self.base):
                dirs[:] = [x for x in dirs[:] if self.check(os.path.join(root, x))]
                print(root)
                for name in dirs + files:
                    self.entries.append((root, name))
        if self.save:
            with timing("save allfiles_e_c"):
                try:
                    os.rename("allfiles_e_c"+self.vcode, "allfiles_e_c" + self.vcode + "." + str(int(time.time())))
                except FileNotFoundError:
                    pass
                with open("allfiles_e_c"+self.vcode, "w")  as writer:
                    writer.write(json.dumps(self.entries))
        print("scan done")
        return self.entries

    def get_allfiles(self):
        self.entries = self.findall()

        if len(self.allfiles) == len(self.entries):
            return self.allfiles
        print("loading file and magic data (this will take quite a while, but will be cached)")
        with timing("get_allfiles"):
            out = self.allfiles
            self.results = {key: value for key, value in self.results.items() if self.check(key)}
            K = 1000
            KK = 10000
            lastq = len(self.results) // K
            lastqq = len(self.results) // KK
            print()
            start = time.time()
            for root, name in self.entries:
                relpath_root = os.path.relpath(root, self.base)
                key = os.path.join(relpath_root, name)
                #print(key)
                if key in self.results:
                    out.append(self.results[key])
                    x = self.results[key]
                else:
                    res = self.get_stat(root, name, children=False)
                    res["parent"] = relpath_root
                    self.results[key] = res
                    out.append(res)
                #res["path"] = os.path.join(relpath_root, name)
                if len(self.results) // K > lastq:
                    lastq = len(self.results) // K
                    now = time.time()
                    if len(self.results) // KK == lastqq: #and now - start <= 30:
                        print("progress:", len(self.results), len(self.entries), 100 * len(self.results) / len(self.entries))
                        #pass
                        #sys.stdout.write("o")
                        #sys.stdout.flush()
                    else:
                        lastqq = len(self.results) // KK
            sys.stdout.flush()
        if self.save and not self.results_from_cache:
            with timing("saving"):
                print("saving, progress:", len(self.results), len(self.entries), 100 * len(self.results) / len(self.entries))
                try:
                    os.rename("allfiles", "allfiles." + str(int(time.time())))
                except FileNotFoundError:
                    pass
                with open("allfiles", "w")  as writer:
                    writer.write(json.dumps(self.results))
                print("done")
                sys.stdout.flush()
        return out

    def get_all_images(self):
        with timing("get_all_images"):
            if not self.all_images:
                print("filtering images...")
                with timing("filtering"):
                    self.all_images = [x for x in self.get_allfiles() if x.get("magic", "error") in files_opts.image_magics]
                print("saving hashes...")
                with timing("by_hash"):
                    self.by_hash = {hashlib.sha256(x["path"].encode("utf-8", ue)).hexdigest(): x for x in self.all_images}
                with timing("check errors"):
                    for i in self.all_images:
                        if "error" not in i:
                            print(i)
                            break
                with timing("make sure hashes are up to date"):
                    for h, v in self.by_hash.items():
                        assert v["hash"] == h
                        v["hash"] = h
                with timing("merge dupes"):
                    try:
                        a = list(csv.reader(open("duplicates.csv", "r")))[1:]
                    except FileNotFoundError:
                        a = []
                    groups = {}
                    for group, filename, folder, size, match in a:
                        g = groups.setdefault(group, [])
                        g.append(os.path.join(folder, filename))
                    for g in groups.values():
                        if any("/fun/app/" in q for q in g): continue
                        q = [os.path.exists(j) for j in g]
                        assert len([x for x in q if x]) == 1, f"err: {g}, {q}"
                        hashes = [hashlib.sha256(("/"+os.path.relpath(j, self.base)).encode("utf-8", ue)).hexdigest() for j in g]
                        r = [h for h, m in zip(hashes, q) if m][0]
                        f = [h for h, m in zip(hashes, q) if not m]
                        hh = [x in self.by_hash for x in hashes]
                        hhc = len([x for x in hh if x])
                        if not hhc: continue

                        img = self.by_hash[r]
                        img["other_hashes"] = f
                        for h in f:
                            self.by_hash[h] = img
                print("done")
                with timing("all_magics"):
                    all_magics = collections.Counter([x.get("magic", x.get("error", "error unknown")) for x in self.get_allfiles()])
                print(all_magics)
                print("image count:")
                print(len(self.all_images))
                with timing("write all_images.msgpack"):
                    with open("all_images.msgpack", "wb") as writer:
                        msgpack.pack(self.all_images, writer, use_bin_type=True, unicode_errors="surrogatepass")
            return self.all_images, self.by_hash

    def get_stat(self, *path, children=False, quick=False):
        joined = os.path.join(*path)
        try:
            s = os.stat(joined)
            isdir = stat.S_ISDIR(s.st_mode)
            result = {
                "size": s.st_size,
                "dir": isdir,
                "mtime": s.st_mtime,
                "ctime": s.st_ctime,
                "name": os.path.basename(joined),
                "path": "/" + os.path.relpath(joined, self.base)
            }
            
            result["hash"] = hashlib.sha256(result["path"].encode("utf-8", ue)).hexdigest()
            if isdir:
                if children:
                    result["children"] = {
                        name: self.get_stat(joined, name)
                        for name
                        in os.listdir(joined)
                    }
                    if os.path.realpath(joined) == os.path.realpath(self.base):
                        result["children"]["_result"] = ({
                            "size": 0,
                            "dir": True,
                            "mtime": time.time(),
                            "ctime": time.time(),
                            "name": "_sorted",
                            "path": "/_sorted"
                        })
                result["magic"] = "directory"
                result["path"] += "/"
            elif result["hash"] in self.magics:
                result["magic"] = self.magics[result["hash"]]
            else:
                result["magic"] = magic.from_file(joined, mime=True)
            if result["hash"] in self.videos:
                result.update(self.videos[result["hash"]])
            elif result["magic"].startswith("video/") and not quick:
                vinfo = {}
                try:
                    videoinfo = subprocess.check_output([
                        "ffprobe",
                        joined,
                        "-loglevel", "quiet",
                        "-print_format", "json",
                        "-show_format",
                        "-show_streams",
                    ])
                    vinfo["video"] = json.loads(videoinfo)
                except (subprocess.CalledProcessError, json.JSONDecodeError)  as e:
                    print("error getting video", e)
                    vinfo["video"] = None
                    vinfo["video_err"] = repr(e)
                try:
                    if result["magic"] != "video/webm":
                        videoinfo = subprocess.check_output([
                            "mp4info",
                            joined,
                        ])
                        vinfo["mp4info"] = videoinfo.decode("utf-8", "backslashreplace")
                except subprocess.CalledProcessError  as e:
                    print("error getting video", e)
                    vinfo["mp4info"] = None
                    vinfo["mp4info_err"] = repr(e)
                self.videos[result["hash"]] = vinfo
                result.update(vinfo)
        except OSError as e:
            result = {
                "error": "oserror",
                "name": os.path.basename(joined)
            }
        except FileNotFoundError as e:
            result = {
                "error": "missing",
                "name": os.path.basename(joined)
            }
        return result

if __name__ == "__main__":
    FilesCache(sys.argv[-1], cache=False)
