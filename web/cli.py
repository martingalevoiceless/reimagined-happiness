
from .files import FilesCache
from .state import State

def setup(base, tempdir):
    files = FilesCache(base, save=False)
    state = State("preferences.json", files, tempdir, update=False)
    state.read()
    return files, state
