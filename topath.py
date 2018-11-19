#!/usr/bin/env python3

import os
import urllib.parse
import sys
import re
from hypothesis import given
from hypothesis.strategies import from_regex, text

url = " ".join(sys.argv[1:])
z = "^((https?:)?//([a-z0-9.]+)(:[0-9]+)?)(/?files)?/"

@given(from_regex(z, fullmatch=True), text())
def test(fz, fzz):
    assert re.sub(z, "", fz + fzz) == fzz
#test()
#for prefix in prefixes:
#if url.startswith():
#    url = 
#print(urllib.parse.urlparse(sys.argv[-1]))
unquoted = re.sub(z, "", urllib.parse.unquote(sys.argv[-1]))

print(unquoted)
