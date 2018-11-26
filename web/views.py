import numpy
import urllib.parse
from pyramid.response import Response
from pyramid.request import Request
import traceback
import time
import pyramid.httpexceptions as exc
import json
import random
from pyramid.view import view_config

from .util import timing
from .state import rand_video_fragments
#todo: view sort page for specific image
#todo: directory priors in tree


@view_config(route_name="app")
def app_(request):
    if not request.registry.settings.get("base", "").strip():
        return Response("Edit pyramid.ini and set `base` to the appropriate path.")
    return exc.HTTPFound(location="/_/compare/")

@view_config(route_name="app1")
def app(request):
    if request.path.startswith("/api"):
        return Response(status_code=404)
    elif request.path.startswith("/robots"):
        request.response.status = 404
    return Response("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>app</title>
            <style>
            html, body, #react, #react>div, #react>div>div {
                height: 100% !important;
            }

            .video-react {
                width: 100% !important;
                height: 100% !important;
            }
            </style>
        </head>
        <body>
            <div id="react">
            </div>
            <script src="/static/dist/vendors~entrypoint.js"></script>
            <script src="/static/dist/entrypoint.js"></script>
        </body>
        </html>
    """)

    

@view_config(route_name='api.files.all', request_method="GET", renderer="json")
def allfiles_endpoint(request):
    base = request.registry.settings["base"]

    af = request.files.get_allfiles()
    return af


@view_config(route_name='compare', request_method="GET", renderer="json")
def compare_get(request):
    with timing("compare_get"):
        path = request.matchdict["rest"]
        base = request.registry.settings["base"]
        return compare_inner(request, path, False)


@view_config(route_name='compare', request_method="PUT", renderer="json")
def compare_put(request):
    with timing("compare_put"):
        path = request.matchdict["rest"]
        info = request.json_body
        if len(path) == 2:
            request.state.update(info, path[0], path[1])
        return compare_inner(request, path, True)


def compare_inner(request, path, had_preference):
    with timing("compare_inner"):
        if path and not had_preference and len(path) >= 2 and path[0] in request.state.bh and path[1] in request.state.bh:
            #print(path)
            with timing("compare_inner::current"):
                a_path, b_path = path[:2]
                a = request.state.bh[a_path]
                b = request.state.bh[b_path]
                a, = rand_video_fragments(a, 1)
                b, = rand_video_fragments(b, 1)
                proba, probb = request.state.getinfo(a, b)
        else:
            a, b, proba, probb = request.state.select_next(path)
        a = dict(a)
        b = dict(b)
        a["info"] = proba
        b["info"] = probb

        return {
            "item1": a,
            "item2": b,
            "path": a["hash"] + "/" + b["hash"],
            "replace": len(path) < 2
        }

@view_config(route_name='fileinfo', request_method="GET", renderer="json")
def get_path(request):
    path = request.matchdict["rest"]
    base = request.registry.settings["base"]
    if path == ('_sorted',):
        af, bh = request.files.get_all_images()
        children = {}
        for idx, h in enumerate(request.state.model_.sorted_hashes[::-1]):
            if h not in bh: continue
            if request.state.model_.is_dropped(request.state.stats, h): continue
            d = dict(bh[h])
            d["path"] = "/" + d["parent"] + "/" + d["name"]
            del d["parent"]
            d["name"] = f"{idx:010d}." + d["name"].rpartition(".")[-1]
            d["vpath"] = "/_sorted/" + d["name"]
            children[d["name"]] = d
        res = {
            "size": 0,
            "dir": True,
            "mtime": time.time(),
            "ctime": time.time(),
            "name": "_sorted",
            "path": "/_sorted",
            "children": children
        }
    elif path[:1] == ("_sorted",):
        af, bh = request.files.get_all_images()
        name = path[1]
        idx = int(name.partition(".")[0])
        h = request.state.model_.sorted_hashes[-1 - idx]
        if h not in bh: return exc.HTTPNotFound()
        if request.state.model_.is_dropped(request.state.stats, h): return exc.HTTPNotFound()
        d = dict(bh[h])
        d["path"] = "/" + d["parent"] + "/" + d["name"]
        del d["parent"]
        d["name"] = name
        d["vpath"] = "/_sorted/" + d["name"]
        res = d
    else:
        res = request.files.get_stat(base, *path, children=True, quick=True)
    if res is None:
        return exc.HTTPNotFound()
    res = dict(res)
    if "hash" in res:
        res["info"], = request.state.getinfo(res["hash"])
    return res


@view_config(context=Exception)
def ex(exc, request):
    try: raise
    except: traceback.print_exc()
    if request.path.startswith("/api"): return Response(json={"err": True}, status_code=500)
    return Response(body="err", status_code=500)
