from waitress import serve
from pyramid.config import Configurator
from .files import FilesCache
from .state import State, reload_all
from .util import timing

def make_app(global_config, **kwargs):
    with timing("startup"):
        with Configurator(settings=kwargs) as config:
            settings = config.get_settings()

            tempdir = settings.get("tempdir", "/tmp")
            if settings.get("base"):
                base = settings.get("base")
                config.add_static_view(name="files", path=settings.get("base"))
            else:
                print("\033[31m Please edit pyramid.ini and set `base =` the path that you'd like. \033[m")
                import sys; sys.exit()
            files = None
            state = None
            def get_files(a):
                nonlocal files
                if files is None:
                    files = FilesCache(base, cache2=True)
                return files
            def get_state(a):
                nonlocal state
                reload_all()
                if state is None:
                    state = State("preferences.json", get_files(a), tempdir)
                    state.read()
                return state
            get_state(None)
            config.add_request_method(get_files, "files", reify=True)
            config.add_request_method(get_state, "state", reify=True)
            config.add_static_view(name="static", path='web:static/')

            config.add_route("api.files.all", "/api/allfiles/")

            config.add_route("similarity", "/api/similarity/*rest")
            config.add_route("compare", "/api/compare/*rest")
            config.add_route("history", "/api/history/{hash}")
            config.add_route("fileinfo", "/api/fileinfo/*rest")

            config.add_route("app1", "/_/*rest")
            config.add_route("app", "/")
            config.scan("web.views")
            app = config.make_wsgi_app()
            with open("app_running", "w") as writer:
                writer.write("running")
            return app

def main():
    app = make_app(None)
    serve(app, host="127.0.0.1", port=8080)

if __name__ == "__main__":
    main()
