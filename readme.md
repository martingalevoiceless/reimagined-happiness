In browser image viewer with ranking.

## mac/linux setup (needs node.js installed already):

1. `cp pyramid.ini.example pyramid.ini`
2. `vim pyramid.ini`, edit to point to your image database and to a correct tempdir
3. `bash setup_web.sh` -- or literally just `cd web; npm install; node_modules/.bin/webpack`
4. `bash run_pyramid.sh` -- if needed, downloads fully isolated copy of pyenv, uses it to install 3.7.0, then sets up a virtualenv; then runs the server

## windows:

1. install chocolatey: [http://chocolatey.org](http://chocolatey.org)
2. make a directory somewhere to put your code, open a terminal, `cd C:/Users/username/that/directory`
3. set up:
```
choco install python nodejs git ffmpeg
git clone https://github.com/martingalevoiceless/reimagined-happiness.git
cd reimagined-happiness/web/
npm install
node_modules\.bin\webpack --config webpack.config.js
cd ..
python -m venv ve
ve\Scripts\activate
pip install python-magic-bin
pip install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp37-cp37m-win_amd64.whl
pip install -e .
copy pyramid.ini.example pyramid.ini
```
4. edit pyramid.ini:
```
base = C:/Users/username/path/to/your/files/
tempdir =  C:/Users/username/AppData/Local/Temp/images_app/
```
5. run it: `pserve pyramid.ini`

## usage

now open [http://localhost:8844](http://localhost:8844) and have fun!

make sure to not expose your site to the internet without authentication (you don't by default) if you don't want to accidentally offer your image library to the world or have randos providing ranking feedback!

views:

- compare view: [http://localhost:8844/\_/compare/](http://localhost:8844/_/compare/)
- file browser view: [http://localhost:8844/\_/p/](http://localhost:8844/_/p/)
- history view: `http://localhost:8844/\_/history/<file hash>` (can be opened by shift-j or shift-o in compare)

### keyboard layout in main view:

[link](http://www.keyboard-layout-editor.com/##@_backcolor=%23dbdbdb&name=Apple%20Wireless%20Keyboard&author=Alistair%20Calder&radii=6px%206px%2012px%2012px%20%2F%2F%2018px%2018px%2012px%2012px&css=%2F@import%20url%28http%2F:%2F%2F%2F%2Ffonts.googleapis.com%2F%2Fcss%3Ffamily%2F=Varela+Round%29%2F%3B%0A%0A%0A%23keyboard-bg%20%7B%20%0A%20%20%20%20background-image%2F:%20linear-gradient%28to%20bottom,%20rgba%280,0,0,0.5%29%200%25,%20rgba%280,0,0,0%29%204%25,%20rgba%28255,255,255,0.3%29%206%25,%20rgba%280,0,0,0%29%2010%25%29,%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20linear-gradient%28to%20right,%20rgba%280,0,0,0.1%29%200%25,%20rgba%280,0,0,0%29%20100%25%29%20!important%2F%3B%20%0A%7D%0A%0A.keylabel%20%7B%0A%20%20%20%20font-family%2F:%20'Noto%20Sans',%20sans-serif%2F%3B%0A%20%20%20%20line-height%2F:%200.70%2F%3B%0A%7D%0A%0A%2F%2F*%20Strangely,%20%22Volkswagen%20Serial%22%20doesn't%20have%20a%20tilde%20character%20*%2F%2F%0A.varela%20%7B%20%0A%20%20%20%20font-family%2F:%20'Noto%20Sans',%20sans-serif%2F%3B%0A%0A%20%20%20%20%2F%2F*font-family%2F:%20'EmojiSymbols'%2F%3B%20*%2F%2F%0A%20%20%20%20display%2F:%20inline-block%2F%3B%20%0A%20%20%20%20font-size%2F:%20inherit%2F%3B%20%0A%20%20%20%20text-rendering%2F:%20auto%2F%3B%20%0A%20%20%20%20-webkit-font-smoothing%2F:%20antialiased%2F%3B%20%0A%20%20%20%20-moz-osx-font-smoothing%2F:%20grayscale%2F%3B%0A%20%20%20%20transform%2F:%20translate%280,%200%29%2F%3B%0A%7D%0A.varela-tilde%2F:after%20%7B%20content%2F:%20%22%5C07e%22%2F%3B%20%7D%3B&@_y:0.75&t=%23666666&p=CHICKLET&a:7&f:2&w:1.0357&h:0.75%3B&=&_w:1.0357&h:0.75%3B&=&_w:1.0357&h:0.75%3B&=&_w:1.0357&h:0.75%3B&=&_w:1.0357&h:0.75%3B&=&_w:1.0357&h:0.75%3B&=&_w:1.0357&h:0.75%3B&=&_w:1.0357&h:0.75%3B&=&_w:1.0357&h:0.75%3B&=&_w:1.0357&h:0.75%3B&=&_w:1.0357&h:0.75%3B&=&_w:1.0357&h:0.75%3B&=&_w:1.0357&h:0.75%3B&=&_w:1.0357&h:0.75%3B&=%3B&@_y:-0.25&a:4&f:5&f2:1%3B&=%0A%60&=%0A1&=%0A2&=%0A3&=%0A4&=%0A5&=%0A6&=%0A7&_a:5&fa@:0&:1&:1&:1&:1&:1&:8%3B%3B&=%0Ashow%20later%0A%0A%0A%0A%0A%E2%A7%96&_a:4%3B&=%0A9&_a:5&f2:1%3B&=%0Axtra%20prefer%0A%0A%0A%0A%0A%E2%AC%88%E2%AC%88&_a:4%3B&=%0A-&=%0A%2F=&_t=%23666666%0A%0A%23b0a800&f:2&w:1.5%3B&=%0A%0A%E2%87%A7forward%0Aback%3B&@_t=%23666666&a:7&w:1.5%3B&=&_a:4&f:5&f2:1%3B&=%0Aq&=%0Aw&=%0Ae&=%0Ar&_a:5&fa@:0&:1&:1&:1&:1&:1&:6%3B%3B&=%0Atranspose%0A%0A%0A%0A%0A%2F&nbsp%2F%3B%2F&nbsp%2F%3B%E2%AC%8E%20%2F&nbsp%2F%3B%20%E2%AC%91%2F&nbsp%2F%3B&_a:4%3B&=%0Ay&_a:5&f2:1%3B&=%0Aundo%0A%0A%0A%0A%0A%E2%86%BA&_a:4&fa@:1&:1&:1&:1&:1&:1&:1&:1&:1&:9&:1%3B%3B&=%E2%8C%98%20unlock%0A%0A%0A%0A%0A%0A%0A%0A%0A%E2%89%88%0Aequal&_t=%23777777%0A%0A%23b0a800%3B&=%0A%0A%E2%87%A7rvw%0A%0A%0A%0A%0A%0A%0A%E2%AC%88%0Aprefer&_x:-1&t=%236697ba%0A%0A%23b0a800&f:3&fa@:1%3B&d:true%3B&=%E2%8C%98lck&_t=%23666666&a:5&f:5&fa@:1&:1%3B%3B&=%0Afull%0A%0A%0A%0A%0A%E2%87%B1&_f:2&fa@:1&:0&:0&:0&:0&:0&:6%3B%3B&=%0Aback%0A%0A%0A%0A%0A%E2%86%90&=%0Aforward%0A%0A%0A%0A%0A%E2%86%92&_a:7&f:5%3B&=%3B&@_a:4&f:2&fa@:1%3B&w:1.75%3B&=%3Ci%20class%2F='kb%20kb-Multimedia-Record'%3E%3C%2F%2Fi%3E%0Aescape&_f:5&fa@:1&:1%3B%3B&=%0Aa&=%0As&_a:5%3B&=%0Adispl.%20info%0A%0A%0A%0A%0A%E2%93%98&_a:4&n:true%3B&=%0Af&=%0Ag&_a:5&f:6&fa@:1&:1%3B%3B&=%0Axtra%20prefer%0A%0A%0A%0A%0A%E2%AC%8B%E2%AC%8B&_t=%23777777%0A%0A%23b0a800&a:4&f:5&fa@:1&:1&:1&:0&:0&:0&:0&:0&:0&:9&:1%3B&n:true%3B&=%0A%0A%E2%87%A7rvw%0A%0A%0A%0A%0A%0A%0A%E2%AC%8B%0Aprefer&_x:-1&t=%236697ba&f:3&fa@:1%3B&d:true%3B&=%E2%8C%98lck&_t=%23666666&a:5&f:9&fa@:1&:1%3B%3B&=%0Aincompar.%0A%0A%0A%0A%0A%E2%A6%B8&_t=%23777777&f:7&fa@:1&:1%3B%3B&=%0Aresample%0A%0A%0A%0A%0A%E2%99%BB&_t=%23666666&a:4&f:5&fa@:1&:1%3B%3B&=%0A%2F%3B&=%0A'&_a:7&f:2&w:1.75%3B&=%3B&@_c=%23c2c195&a:4&fa@:7%3B&w:2.25%3B&=%E2%87%A7%0Ashift&_c=%23cccccc&f:5&fa@:7&:1%3B%3B&=%0Az&=%0Ax&=%0Ac&=%0Av&=%0Ab&_a:5%3B&=%0Afull%0A%0A%0A%0A%0A%E2%87%B1&_f:7&fa@:0&:1&:0&:0&:0&:0&:1%3B%3B&=%0Aresample%0A%0A%0A%0A%0A%E2%99%BB&_f:9&fa@:0&:1&:0&:0&:0&:0&:1%3B%3B&=%0Agoes%20well%0A%0A%0A%0A%0A%E2%A6%B9&_a:4&f:5&f2:1%3B&=%0A.&=%0A%2F%2F&_c=%23c2c195&f:2&fa@:0&:1&:7%3B&w:2.25%3B&=%0A%0A%E2%87%A7%0Ashift%3B&@_c=%23cccccc&f:2&h:1.111%3B&=%0Afn&_fa@:8%3B&h:1.111%3B&=%E2%8C%83%0Acontrol&_fa@:4%3B&h:1.111%3B&=%E2%8C%A5%0Aoption%20alt&_c=%2395b4c9&fa@:4&:0&:5%3B&w:1.25&h:1.111%3B&=%0A%0A%E2%8C%98%0Acommand%20super&_c=%23cccccc&a:7&w:5&h:1.111%3B&=&_c=%2395b4c9&a:4&w:1.25&h:1.111%3B&=%E2%8C%98%0Acommand%20super&_c=%23cccccc&fa@:4&:0&:4%3B&h:1.111%3B&=%0A%0A%E2%8C%A5%0Aoption&_x:1&a:7&f:5&h:0.611%3B&=%3B&@_y:-0.5&x:11.5&h:0.6111%3B&=&_h:0.6111%3B&=&_h:0.6111%3B&=)

`0` top/right strongly preferred (use sparingly, normal prefer tracks how long you take to respond to estimate how easy it was to rank)  
`o` top/right preferred  
`i` too close to easily call  
`j` bottom/left preferred  
`h` bottom/left strongly preferred  

`p` view top/right full  
`l` show top/right with a different pairing

`k` incomparable: doesn't even make sense to compare these images  
`,` goes well: they're so incomparable that viewing them together is better than either one on its own

`n` view bottom/left full  
`m` show bottom/left with a different pairing

`Shift+o` view history for top/right  
`Shift+j` view history for bottom/left  

*on windows, Super is the win key, and on mac, it's command.*  
`Super+o` lock to top/right image  
`Super+i` unlock, go back to showing the most useful comparison  
`Super+j` lock to the bottom/left image

`[` alias for browser back  
`]` alias for browser forward  
`backspace` alias for browser back  
`shift-backspace` alias for browser forward

`t` switch between horizontal and vertical view  
`d` show/hide numerical info

### keyboard layout in file browser view

`j` go to previous image  
`i` compare this image  
`o` go to next image

`p` return to last compare  
`n` return to last compare

`d` show/hide numerical info

`[` alias for browser back  
`]` alias for browser forward  
`backspace` alias for browser back  
`shift-backspace` alias for browser forward
