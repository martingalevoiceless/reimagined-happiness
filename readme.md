In browser image viewer. to set up:

1. `cp pyramid.ini.example pyramid.ini`
2. `vim pyramid.ini`, edit to point to your image database and to a correct tempdir
3. `bash setup_web.sh` -- or literally just `cd web; npm install; node_modules/.bin/webpack`
4. `bash run_pyramid.sh` -- if needed, downloads fully isolated copy of pyenv, uses it to install 3.7.0, then sets up a virtualenv; then runs the server

now open http://localhost:8844 and have fun! make sure to not expose your site to the internet without authentication if you don't want to accidentally offer your image library to the world or have randos providing ranking feedback!



