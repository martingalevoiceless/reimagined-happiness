#!/bin/bash
cd "$(dirname "$(realpath "$0")")"/

if [ ! -d .pyenv ]; then git clone https://github.com/pyenv/pyenv.git .pyenv; fi

export PYENV_ROOT="$PWD/.pyenv"
./.pyenv/bin/pyenv install 3.7.0 -s

if ! [ -d ve/ ] ; then
    ./.pyenv/versions/3.7.0/bin/python -m venv ve/
    ve/bin/pip install --upgrade pip setuptools
fi
ve/bin/pip install -e .

ve/bin/pserve pyramid.ini
