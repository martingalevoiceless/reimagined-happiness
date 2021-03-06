
import React from 'react'
import { Dimensions,Image, Button, Platform, FlatList, StyleSheet, Text, View, TextInput } from 'react-native'
import { Router, Route, Link, with_redirector } from './router-wrapper'
import { AsyncStorage } from "react-native"
import base64 from 'Base64';
import { MaterialIcons, MaterialCommunityIcons } from './icons';
import Video from './video';
import { UtilComponent } from './util';
import keyboard from './keyboard';


class Settings_ extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            username: "",
            password: "",
            url: ""
        };
        this.load();
    }
    async load() {
        var j = await AsyncStorage.getItem("apiinfo");
        if (j) {
            var d = JSON.parse(j);
            this.setState(d);
        }
    }
    async save(args={}) {
        var up = Object.assign({}, this.state);
        up = Object.assign(up, args);
        this.setState(args);
        await AsyncStorage.setItem("apiinfo", JSON.stringify(up));
        await this.load();
    }
    render() {
      return <View>
        <TextInput
          style={styles.apiurl_field}
          placeholder="Enter api url"
          value={this.state.url}
          onChangeText={(url) => this.save({url})}
        />
        <TextInput
          style={styles.apiurl_field}
          placeholder="Enter username"
          value={this.state.username}
          onChangeText={(username) => this.save({username})}
        />
        <TextInput
          style={styles.apiurl_field}
          placeholder="Enter password"
          value={this.state.password}
          onChangeText={(password) => this.save({password})}
        />
        <Link to="/">
          <Text>Done</Text>
        </Link>
      </View>
    }
}
const Settings = with_redirector(Settings_);

class Home_ extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
        };
        this.setup()
    }
    async setup() {
        var au = await AsyncStorage.getItem("apiinfo");
        if (Platform.OS === 'web' && au == undefined) {
            au = "";
            AsyncStorage.setItem('apiinfo', JSON.stringify({url:"", password: "", username: ""}));
        }
        if (au == undefined) {
            this.props.redirect("/_/settings/");
        } else {
            this.props.redirect("/_/p/");
        }
    }
    render() {
      return <View>
        <Text>Loading...</Text>
      </View>
    }
}
const Home = with_redirector(Home_);

async function apiinfo() {
    var i = {url:"",password:"",username:""};
    var j = await AsyncStorage.getItem("apiinfo");
    if (j) {
        i = JSON.parse(j);
    }
    i.inclpw = (i.password && i.url && i.username && i.url.startsWith("https://"))

    return i;
}
async function apiurl(path, usepw=true) {
    var i = await apiinfo();
    if (i.inclpw && usepw) {
        var host = i.url.substring(8);
        i.url = `https://${i.username}:${i.password}@${host}`;
    }
    return i.url + path;
}
async function imgurl(path, usepw=true) {
    var i = await apiinfo();
    if (i.inclpw && usepw) {
        var host = i.url.substring(8);
        i.url = `https://${host}/filesauth/${i.password}`;
    } else {
        i.url = `/files`;
    }
    return i.url + path;
}

async function fetchpath(path, params={}) {
    var i = await apiinfo();
    if (i.inclpw) {
        params.headers = new Headers();
        params.headers.set('Authorization', 'Basic ' +  base64.btoa(i.username.trim() + ":" + i.password.trim()));
    }
    params.credentials = 'include';
    params.cache = 'no-cache';

    return await fetch(await apiurl(path, false), params);
}


const Content = ({item, style, cellwidth, imgurl, onClick, big, in_compare}) => {
    if (imgurl === undefined) {
        return <View/>;
    }
    var src = imgurl + encodeURIComponent(item.path);
    if (item.magic === "image/jpeg" || item.magic === "image/gif" || item.magic == "image/png" || item.magic == "image/webp") {
        // TODO: https://github.com/DuDigital/react-native-zoomable-view
        // TODO: https://github.com/expo/videoplayer
        // TODO: gif and etc
        return <Image 
            style={style}
            source={{uri: src}}
            onClick={onClick}
            />;
    } else if (item.magic === "video/mp4" || item.magic === "video/x-flv" || item.magic == "video/webm" || item.magic == "video/x-m4v" || item.magic == "video/quicktime") {
        return <Video
            autosize={big}
            style={style}
            source={src}
            onClick={onClick}
            controls={big}
            autoplay={big}
            short_controls={in_compare}
            long_controls={!in_compare && big}
            min_time={item.min_time}
            max_time={item.max_time}
            />;
    }
    return <View style={style} onClick={onClick}>
        <MagicIcon magic={item.magic} size={16} color="#ffffff" style={styles.icon}/>
        <Text style={styles.fileentry_title}>{item.name}</Text>
        <Text style={styles.fileentry_magic}>{item.magic}</Text>
    </View>;
};
class FileEntry_ extends UtilComponent {
    constructor(props) {
        super(props);
    }
    render() {
        var hw = {height: this.props.cellwidth, maxWidth: this.props.cellwidth};
        return <View style={[styles.fileentry, hw]}>
            <Link to={this.props.nextpath || ("/_/p" + (this.props.item.vpath || this.props.item.path))}>
                <Content
                    style={[styles.fileentry_content, hw]}
                    item={this.props.item}
                    imgurl={this.props.imgurl}
                    cellwidth={this.props.cellwidth}
                    big={false}
                />
                {this.props.info &&
                    <View 
                        style={styles.info_small}>
                        {this.props.item.info && this.props.item.info.map((val, idx) => (
                            <Text style={this.props.smallinfo ? styles.textstyle_small : styles.textstyle} key={idx}>{val}</Text>
                        ))}
                    </View>
                }
            </Link>
        </View>
    }
}
const FileEntry = with_redirector(FileEntry_);

class SingleFile_ extends UtilComponent {
    constructor(props) {
        super(props);
    }
    render({item, imgurl, next, prev, onClick}) {
        return <View style={styles.singlefile}>
            <Content
                style={styles.singlefile_content}
                item={item}
                imgurl={imgurl}
                onClick={onClick || (() => { if (next && (next.vpath||next.path)) { this.props.redirect("/_/p" + (next.vpath||next.path))}})}
                big={true}
                in_compare={this.props.in_compare}
            />
            <View
                style={this.props.info_top ? styles.info_top : styles.info_bottom}>
            {this.props.info
                ? 
                    <React.Fragment>
                        <View
                            style={this.props.info_top ? styles.info_col_top: styles.info_col_bottom}>
                            <Button
                                onPress={() => this.props.setinfo(false)}
                                title="hide"
                                color="#404040"
                                style={styles.vbutton}
                                accessibilityLabel="hide info"
                                />

                            <Button
                                onPress={() => this.props.redirect("/_/compare/" + this.props.item.hash + "/")}
                                title="compare"
                                style={styles.vbutton}
                                accessibilityLabel="compare"
                                />
                            <Button
                                onPress={() => this.props.redirect("/_/p" + (this.props.item.path))}
                                title="view"
                                style={styles.vbutton}
                                accessibilityLabel="view"
                                />
                            {this.props.children}
                        </View>
                        <View 
                            style={styles.info_main}>
                            {this.props.item.info && this.props.item.info.map((val, idx) => (
                                <Text style={styles.textstyle} key={idx}>{val}</Text>
                            ))}
                        </View>
                    </React.Fragment>

                :
                    <Button
                        onPress={() => this.props.setinfo(true)}
                        title="info"
                        color="#404040"
                        accessibilityLabel="info"
                        />
            }
            </View>

        </View>
    };
}
const SingleFile = with_redirector(SingleFile_);

const sortby = (array, f) => {
    array.sort((a, b) => {
        var as = f(a);
        var bs = f(b);
        if (as > bs) {
            return 1;
        } else if (as < bs) {
            return -1;
        } else {
            return 0;
        }
    });
};

class Directory_ extends UtilComponent {
    constructor(props) {
        super(props);
    }

    rc(x) {
        return <View>
            <Text>{JSON.stringify(x)}</Text>
        </View>
    }
    render() {
        //var chil = [];
        //if (this.state.c) {
        //    for (var x of this.state.c) {
        //        chil.append(this.rc(x));
        //    }
        //}
        var cols = this.prop("cols", 3);
        var {height, width} = Dimensions.get('window');
        return <View style={styles.directory}>
            <FlatList
                windowSize={2}
                initialNumToRender={4}
                key={"" + cols}
                data={this.prop("children", [])}
                style={styles.browserlist}
                renderItem={({item}) => <FileEntry
                    imgurl={this.props.imgurl}
                    item={item}
                    smallinfo={true}
                    info={this.props.info}
                    cellwidth={width/cols}
                />}
                numColumns={cols}
                />
        </View>
    }
}
const Directory = with_redirector(Directory_);

const File = (props) => (
    (props.item || {}).dir ? (
        <Directory
            {...props}
        />
    ) : (
        <SingleFile
            {...props}
        />
    )
);

var magic_icons = {
    "directory": {set: MaterialIcons, name: "folder"},
    "data": {set: MaterialCommunityIcons, name: "file"},
};
const MagicIcon = ({magic, ...props}) => {
    var icon = magic_icons[magic];
    if (icon === undefined) {
        icon = magic_icons["data"];
    }
    var IconSet = icon.set;
    return <IconSet name={icon.name} {...props}/>;
}
var dircache = {};
var comparecache = {};
function convert_c(cc) {
    if (!cc || cc.name === undefined) {
        return;
    }
    var key = cc.name;
    if (cc.parent !== undefined) {
        cc.path = "/" + cc.parent + "/" + cc.name;
    }
    if (cc.path !== undefined) {
        key = cc.path;
    }
    return {
        key,
        ...cc
    }
}
function convert_jc(jc, sortkey=(x => [!(x.dir), x.name])) {
    var c = [];
    if (jc) {
        for (var child in jc) {
            var cc = convert_c(jc[child]);
            if (!cc) {
                continue;
            }
            c.push(cc);
        }
    }
    sortby(c, sortkey);
    return c;
}

class Browser_ extends React.Component {
    constructor(props) {
        super(props);
        var {height, width} = Dimensions.get('window');
        this.state = {
            col_width: width / Math.round(width/128),
            imgurl: undefined,
            info: false,
        };
        this.geturl();
    }
    async geturl() {
        var url = await imgurl("", true);
        this.setState({imgurl: url});
    }
    componentDidMount() {
        this.request();
    }
    rest(props) {
        if (props === undefined) {
            props = this.props;
        }
        return (props.match.params.rest || "")
    }
    subpaths(props) {
        var pc = [""].concat(this.rest().split(/\/+/).filter(x => x));
        var subpath = "";
        var subpaths = [];
        for (var idx = 0; idx < pc.length; idx++) {
            var section = pc[idx];
            var magic = "directory";
            if (idx !== pc.length - 1 || (this.state.item||{}).dir) {
                section += "/";
            }
            if (idx === pc.length - 1) {
                magic = (this.state.item||{}).magic || "directory";
            }
            subpath += section;
            subpaths.push({
                magic, section, subpath
            });
        }
        return subpaths;
    }
    async request() {
        try {
            this.setState({loading: true});
            var path = this.rest();
            var subpaths = this.subpaths();
            var req1 = undefined;
            if (subpaths.length >= 2) {
                var parent = subpaths[subpaths.length-2];
                req1 = this.request_inner(parent.subpath);
            }
            var parent;
            var j = await this.request_inner(path);
            if (path !== this.rest()) {
                return;
            }
            var jc = j.children;
            var c = convert_jc(jc);
            var values = {
                j: JSON.stringify(j),
                item: j,
                children: c,
                parent: parent,
                loaded: path,
                loading: false,
                prev,
                next,
            };
            this.setState(values);

            var next, prev;
            if (req1) {
                parent = await req1;
                var position;
                var pc = convert_jc(parent.children);
                for (var i=0; i<pc.length; i++) {
                    var ch = pc[i];
                    if (ch && ch.name == j.name) {
                        position = i;
                        break;
                    }
                }
                if (position !== undefined) {
                    prev=pc[position-1];
                    next=pc[position+1];
                }
            }

            if (path !== this.rest()) {
                return;
            }
            var jc = j.children;
            var c = convert_jc(jc);
            var values = {
                j: JSON.stringify(j),
                item: j,
                children: c,
                parent: parent,
                loaded: path,
                loading: false,
                prev,
                next,
            };
            this.setState(values);
        } catch (e) {
            console.warn(e);
            if (e.status == 401) {
                this.props.redirect("/_/settings/");
            } else {
                throw e;
            }
        }
    }
    async request_inner(path) {
        var j;
        if (dircache[path] !== undefined) {
            j = dircache[path];
        } else {
            var resp = await fetchpath("/api/fileinfo/" + encodeURIComponent(path));
            j = await resp.json();
        }
        var jc = j.children;
        if (jc) {
            dircache[path] = j;
        }
        return j;
    }
    componentDidUpdate(prevProps, prevState, snapshot) {
        if (this.rest(prevProps) !== this.rest()) {
            this.request();
        }
    }
    render() {
        screen_key_callbacks = {};
        screen_key_callbacks['d'] = () => this.setState({info: !this.state.info, children: this.state.children.slice()});
        screen_key_callbacks['n'] = () => this.props.redirect("/_/compare/");
        screen_key_callbacks['j'] = (() => { if (this.state.prev && (this.state.prev.vpath||this.state.prev.path)) { this.props.redirect("/_/p" + (this.state.prev.vpath||this.state.prev.path))}});
        screen_key_callbacks['i'] = () => this.props.redirect("/_/compare/" + this.state.item.hash + "/");
        screen_key_callbacks['o'] = (() => { if (this.state.next && (this.state.next.vpath||this.state.next.path)) { this.props.redirect("/_/p" + (this.state.next.vpath||this.state.next.path))}});
        screen_key_callbacks['p'] = () => this.props.redirect("/_/compare/");

        //screen_key_callbacks['u'] = () => this.request({undo: true, fast: true});
        //screen_key_callbacks[','] = () => this.request({incomparable: true, goes_well: true, fast: true});

        screen_key_callbacks['['] = () => this.props.history.goBack();
        screen_key_callbacks[']'] = () => this.props.history.goForward();
        screen_key_callbacks['Backspace'] = () => this.props.history.goBack();
        screen_key_callbacks['Shift-Backspace'] = () => this.props.history.goForward();

        if (this.state.imgurl === undefined) {
            return <Text>loading imgurl</Text>;
        }
        var {height, width} = Dimensions.get('window');
        var ww = width/this.state.col_width;
        var cols = Math.round(ww);
        var icols = cols;
        if (!isFinite(cols) || cols == 0) {
            cols = 3;
            
        }
        if (cols > 8) { cols = 6; }
        var breadcrumbs = [];
        var subpaths = this.subpaths();
        for (var subpath of subpaths) {
            if (breadcrumbs.length) {
                breadcrumbs.push(
                    <View style={styles.breadcrumb_separator} key={"sep_" + subpath.subpath}>
                        <Text style={styles.textstyle}> &gt; </Text>
                    </View>
                )
            }
            breadcrumbs.push(
                <View style={styles.breadcrumb} key={"path_" + subpath.subpath}>
                    <Link to={"/_/p" + subpath.subpath}>
                        <View style={styles.breadcrumb_inner}>
                            <MagicIcon magic={subpath.magic} size={16} color="#ffffff" style={styles.icon}/>
                            <Text style={styles.breadcrumb_text}>
                                {subpath.section}
                            </Text>
                        </View>
                    </Link>
                </View>
            )
        }
        //<Text>cols: {cols}, height: {height}, width: {width}, cw: {this.state.col_width}, ww: {ww}, rww:  icols: {icols}</Text>
        return <View style={styles.browser}>
            <View style={styles.breadcrumbs}>
                {breadcrumbs}
            </View>
            {!this.state.loading && 
            <File
                in_compare={false}
                prev={this.state.prev}
                next={this.state.next}
                imgurl={this.state.imgurl}
                item={this.state.item}
                children={this.state.children}
                cols={cols}
                info={this.state.info}
                setinfo={info => this.setState({info})}
            />}
        </View>
    }
}
const Browser = with_redirector(Browser_);

var screen_key_callbacks = {};
keyboard(function (event) {
    var k = event.key;
    if (event.altKey) { k = "Alt-" + k; }
    if (event.ctrlKey) { k = "Ctrl-" + k; }
    if (event.shiftKey) { k = "Shift-" + k; }
    if (event.metaKey) { k = "Meta-" + k; }
    var callback = screen_key_callbacks[k];
    if (callback) {
        event.preventDefault();
    } 
    console.log(k, event);
    if (callback) {
        callback();
    }
});
var global_hor;

class Compare_ extends React.Component {
    constructor(props) {
        super(props);
        this.state = {info: false, hor: global_hor};
        this.geturl();
    }
    async geturl() {
        var url = await imgurl("", true);
        this.setState({imgurl: url});
    }
    componentDidMount() {
        this.request();
    }
    rest(props) {
        if (props === undefined) {
            props = this.props;
        }
        return (props.match.params.rest || "")
    }
    componentDidUpdate(prevprops) {
        if (this.rest() !== this.rest(prevprops) && !this.state.loading && this.rest() !== this.state.loadedpath) {
            this.request();
        }
    }
    async request(preference) {
        try {
            this.setState({loading: true});
            var path = this.rest();
            var j;
            if (false && comparecache[path] !== undefined && preference === undefined) {
                j = comparecache[path];
            } else {
                var extra = {};
                if (preference !== undefined) {
                    extra = {headers: {"content-type": "application/json"}, "method": "PUT", "body": JSON.stringify({
                        viewstart: this.state.viewstart,
                        viewend: +(new Date()),
                        info: this.state.info_,
                        hor: this.state.hor,
                        os: Platform.OS,
                        preference
                    })};
                }
                var resp = await fetchpath("/api/compare/" + path, extra);
                j = await resp.json();
                comparecache[j.path] = j;
            }
            if (path !== this.rest()) {
                return;
            }
            this.setState({derp: !this.state.derp});
            if (j.path && j.path !== this.rest()) {
                console.log(j.path, this.rest(), "redirect");
                this.props.redirect("/_/compare/" + j.path, !(j.replace));
            }
            var values = {
                j: JSON.stringify(j),
                loadedpath: j.path,
                viewstart: +(new Date()),
                item1: convert_c(j.item1),
                item2: convert_c(j.item2),
                info_: j.info,
                loading: false,
            };
            this.setState(values);
        } catch (e) {
            console.warn(e);
            if (e.status == 401) {
                this.props.redirect("/_/settings/");
            } else {
                throw e;
            }
        }
    }
    render() {
        screen_key_callbacks = {};
        screen_key_callbacks['t'] = () => this.setState({hor: !this.state.hor}, () => {global_hor = this.state.hor});
        screen_key_callbacks['d'] = () => this.setState({info: !this.state.info});
        //screen_key_callbacks['Shift-B'] = () => this.setState({info: !this.state.info});

        screen_key_callbacks['m'] = () => this.props.redirect("/_/compare/" + this.state.item2.hash + "/");

        screen_key_callbacks['h'] = () => this.request({prefer: 2, fast: true, strong: 1});
        screen_key_callbacks['n'] = () => this.props.redirect("/_/p" + (this.state.item2.path));
        screen_key_callbacks['j'] = () => this.request({prefer: 2, fast: true});
        screen_key_callbacks['Shift-J'] = () => this.props.redirect("/_/history/" + this.state.item2.hash);

        screen_key_callbacks['i'] = () => this.request({too_close: true, fast: true});
        screen_key_callbacks['8'] = () => this.request({not_sure: true, fast: true});

        screen_key_callbacks['Shift-O'] = () => this.props.redirect("/_/history/" + this.state.item1.hash);
        screen_key_callbacks['o'] = () => this.request({prefer: 1, fast: true});
        screen_key_callbacks['p'] = () => this.props.redirect("/_/p" + (this.state.item1.path));
        screen_key_callbacks['0'] = () => this.request({prefer: 1, fast: true, strong: 1});

        screen_key_callbacks['l'] = () => this.props.redirect("/_/compare/" + this.state.item1.hash + "/");

        screen_key_callbacks['Meta-j'] = () => this.request({lock: this.state.item2.hash});
        screen_key_callbacks['Meta-i'] = () => this.request({lock: null});
        screen_key_callbacks['Meta-o'] = () => this.request({lock: this.state.item1.hash});

        screen_key_callbacks['u'] = () => this.request({undo: true, fast: true});
        screen_key_callbacks['k'] = () => this.request({incomparable: true, fast: true});
        screen_key_callbacks[','] = () => this.request({incomparable: true, goes_well: true, fast: true});

        screen_key_callbacks['['] = () => this.props.history.goBack();
        screen_key_callbacks[']'] = () => this.props.history.goForward();
        screen_key_callbacks['Backspace'] = () => this.props.history.goBack();
        screen_key_callbacks['Shift-Backspace'] = () => this.props.history.goForward();

        var dir = {
            browser: styles.browser,
            compare_pane: styles.compare_pane,
            compare_buttons: styles.compare_buttons,
            too_close: "Too close",
            incomparable:"Incomp.",
            goes: "Goes",
            undo: "Undo",
        };
        if (this.state.hor) {
            dir = {
                browser: styles.browser_hor,
                compare_pane: styles.compare_pane_hor,
                compare_buttons: styles.compare_buttons_hor,
                too_close: "==",
                incomparable:"!=",
                goes: "+=",
                undo: "U",
            };
        }
        return <View style={dir.browser}>
            <View style={dir.compare_pane}>
                {this.state.item1 && 
                <File
                    in_compare={true}
                    imgurl={this.state.imgurl}
                    item={this.state.item1}
                    next={this.state.item1}
                    cols={1}
                    info_top={true}
                    info={this.state.info}
                    setinfo={info => this.setState({info})}
                >
                    <View style={styles.blklst}>
                    <Button
                        onPress={() => this.request({dislike: [1,0]})}
                        title="blklst"
                        color="#be3e2e"
                        accessibilityLabel="blklst"
                        />
                    </View>
                </File>}
                <View style={this.state.derp ? styles.compare_info_top : styles.compare_info_top2}>
                    <Button
                        onPress={() => this.request({prefer: 1})}
                        title="Prefer"
                        color="#999999"
                        accessibilityLabel="Prefer"
                        />
                </View>
            </View>
            <View style={dir.compare_buttons}>
                <Button
                    onPress={() => this.request({lock: null})}
                    title={"Unlk"}
                    color="#404040"
                    style={styles.compare_button}
                    />
                <Button
                    onPress={() => this.request({undo: true})}
                    title={dir.undo}
                    style={styles.compare_button}
                    color="#404040"
                    />
                <Button
                    onPress={() => this.request({incomparable: true, goes_well: true})}
                    title={dir.goes}
                    style={styles.compare_button}
                    color="#404040"
                    />
                <Button
                    onPress={() => this.request({incomparable: true})}
                    title={dir.incomparable}
                    accessibilityLabel="Doesn't make sense to compare"
                    color="#404040"
                    style={styles.compare_button}
                    />
                <Button
                    onPress={() => this.request({too_close: true})}
                    title={dir.too_close}
                    accessibilityLabel="Too close to call"
                    style={styles.compare_button}
                    color="#404040"
                    />
            </View>
            <View style={styles.compare_pane}>
                {this.state.item2 && 
                <File
                    in_compare={true}
                    imgurl={this.state.imgurl}
                    item={this.state.item2}
                    next={this.state.item2}
                    cols={1}
                    info={this.state.info}
                    setinfo={info => this.setState({info})}
                >
                    <View style={styles.blklst}>
                    <Button
                        onPress={() => this.request({dislike: [0,1]})}
                        title="blklst"
                        style={styles.blklst}
                        color="#be3e2e"
                        accessibilityLabel="blklst"
                        />
                    </View>
                </File>}
                
                <View style={this.state.derp ? styles.compare_info_bottom : styles.compare_info_bottom2}>
                    <Button
                        onPress={() => this.request({prefer: 2})}
                        color="#999999"
                        title="Prefer"
                        accessibilityLabel="Prefer"
                        />
                        </View>
            </View>
        </View>;
    }
}
const Compare = with_redirector(Compare_);

class Similarity_ extends React.Component {
    constructor(props) {
        super(props);
        this.state = {info: false, hor: global_hor};
        this.geturl();
    }
    async geturl() {
        var url = await imgurl("", true);
        this.setState({imgurl: url});
    }
    componentDidMount() {
        this.request();
    }
    rest(props) {
        if (props === undefined) {
            props = this.props;
        }
        return (props.match.params.rest || "")
    }
    componentDidUpdate(prevprops) {
        if (this.rest() !== this.rest(prevprops) && !this.state.loading && this.rest() !== this.state.loadedpath) {
            this.request();
        }
    }
    async request(similarity) {
        try {
            this.setState({loading: true});
            var path = this.rest();
            var j;
            if (false && comparecache[path] !== undefined && similarity === undefined) {
                j = comparecache[path];
            } else {
                var extra = {};
                if (similarity !== undefined) {
                    extra = {headers: {"content-type": "application/json"}, "method": "PUT", "body": JSON.stringify({
                        viewstart: this.state.viewstart,
                        viewend: +(new Date()),
                        info: this.state.info_,
                        hor: this.state.hor,
                        os: Platform.OS,
                        similarity
                    })};
                }
                var resp = await fetchpath("/api/similarity/" + path, extra);
                j = await resp.json();
                comparecache[j.path] = j;
            }
            if (path !== this.rest()) {
                return;
            }
            this.setState({derp: !this.state.derp});
            if (j.path && j.path !== this.rest()) {
                console.log(j.path, this.rest(), "redirect");
                this.props.redirect("/_/similarity/" + j.path, !(j.replace));
            }
            var values = {
                j: JSON.stringify(j),
                loadedpath: j.path,
                viewstart: +(new Date()),
                item1: convert_c(j.item1),
                item2: convert_c(j.item2),
                item3: convert_c(j.item3),
                info_: j.info,
                loading: false,
            };
            this.setState(values);
        } catch (e) {
            console.warn(e);
            if (e.status == 401) {
                this.props.redirect("/_/settings/");
            } else {
                throw e;
            }
        }
    }
    render() {
        screen_key_callbacks = {};
        screen_key_callbacks['t'] = () => this.setState({hor: !this.state.hor}, () => {global_hor = this.state.hor});
        screen_key_callbacks['d'] = () => this.setState({info: !this.state.info});
        //screen_key_callbacks['Shift-B'] = () => this.setState({info: !this.state.info});

        //screen_key_callbacks['m'] = () => this.props.redirect("/_/compare/" + this.state.item2.hash + "/");

        //screen_key_callbacks['h'] = () => this.request({prefer: 2, fast: true, strong: 1});
        //screen_key_callbacks['n'] = () => this.props.redirect("/_/p" + (this.state.item2.path));
        screen_key_callbacks['j'] = () => this.request({least_similar: 2, fast: true});
        screen_key_callbacks['Meta-j'] = () => this.request({most_similar: 2, fast: true});
        //screen_key_callbacks['Shift-J'] = () => this.props.redirect("/_/history/" + this.state.item2.hash);

        screen_key_callbacks['k'] = () => this.request({too_close: true, fast: true});
        screen_key_callbacks['8'] = () => this.request({not_sure: true, fast: true});

        //screen_key_callbacks['Shift-O'] = () => this.props.redirect("/_/history/" + this.state.item1.hash);
        screen_key_callbacks['i'] = () => this.request({least_similar: 1, fast: true});
        screen_key_callbacks['Meta-i'] = () => this.request({most_similar: 1, fast: true});
        //screen_key_callbacks['p'] = () => this.props.redirect("/_/p" + (this.state.item1.path));
        //screen_key_callbacks['0'] = () => this.request({prefer: 1, fast: true, strong: 1});

        //screen_key_callbacks['l'] = () => this.props.redirect("/_/compare/" + this.state.item1.hash + "/");

        //screen_key_callbacks['Meta-j'] = () => this.request({lock: this.state.item2.hash});
        //screen_key_callbacks['Meta-i'] = () => this.request({lock: null});
        //screen_key_callbacks['Meta-o'] = () => this.request({lock: this.state.item1.hash});

        //screen_key_callbacks['u'] = () => this.request({undo: true, fast: true});
        screen_key_callbacks['o'] = () => this.request({least_similar: 0, fast: true});
        screen_key_callbacks['Meta-o'] = () => this.request({most_similar: 0, fast: true});
        //screen_key_callbacks[','] = () => this.request({incomparable: true, goes_well: true, fast: true});

        screen_key_callbacks['['] = () => this.props.history.goBack();
        screen_key_callbacks[']'] = () => this.props.history.goForward();
        screen_key_callbacks['Backspace'] = () => this.props.history.goBack();
        screen_key_callbacks['Shift-Backspace'] = () => this.props.history.goForward();

        var dir = {
            browser: styles.browser,
            compare_pane: styles.compare_pane,
            compare_buttons: styles.compare_buttons,
            too_close: "Too Close",
            incomparable:"Incomparable",
            goes: "Goes Well",
            undo: "Undo",
            transpose: "Transpose"
        };
        if (this.state.hor) {
            dir = {
                browser: styles.browser_hor,
                compare_pane: styles.compare_pane_hor,
                compare_buttons: styles.compare_buttons_hor,
                too_close: "==",
                incomparable:"!=",
                goes: "+=",
                undo: "U",
                transpose: "T",
            };
        }
        return <View style={dir.browser}>
            <View style={dir.compare_pane}>
                {this.state.item2 && 
                <File
                    in_compare={true}
                    imgurl={this.state.imgurl}
                    item={this.state.item1}
                    next={this.state.item1}
                    cols={1}
                    info_top={true}
                    info={this.state.info}
                    setinfo={info => this.setState({info})}
                >
                    <View style={styles.blklst}>
                    <Button
                        onPress={() => this.request({dislike: [1,0,0]})}
                        title="blklst"
                        color="#be3e2e"
                        accessibilityLabel="blklst"
                        />
                    </View>
                </File>}
                <View style={this.state.derp ? styles.compare_info_top : styles.compare_info_top2}>
                    <Button
                        onPress={() => this.request({least_similar: 0})}
                        title="Least sim."
                        color="#999999"
                        accessibilityLabel="Least similar image"
                        />
                    <Button
                        onPress={() => this.request({most_similar: 0})}
                        title="Most sim."
                        color="#999999"
                        style={styles.compare_inv_button}
                        accessibilityLabel="Most similar image"
                        />
                </View>
            </View>
            <View style={dir.compare_buttons}>
                <Button
                    onPress={() => this.request({too_close: true})}
                    title={dir.too_close}
                    accessibilityLabel="Too close to call"
                    />
                <Button
                    onPress={() => this.request({undo: true})}
                    title={dir.undo}
                    />
                <Button
                    onPress={() => this.setState({hor: !this.state.hor})}
                    title={dir.transpose}
                    />
            </View>
            <View style={styles.compare_pane}>
                {this.state.item2 && 
                <File
                    in_compare={true}
                    imgurl={this.state.imgurl}
                    item={this.state.item2}
                    next={this.state.item2}
                    cols={1}
                    info={this.state.info}
                    setinfo={info => this.setState({info})}
                >
                    <View style={styles.blklst}>
                    <Button
                        onPress={() => this.request({dislike: [0,1,0]})}
                        title="blklst"
                        style={styles.blklst}
                        color="#be3e2e"
                        accessibilityLabel="blklst"
                        />
                    </View>
                </File>}
                
                <View style={this.state.derp ? styles.compare_info_middle : styles.compare_info_middle2}>
                    <Button
                        onPress={() => this.request({least_similar: 1})}
                        title="Least sim."
                        color="#999999"
                        accessibilityLabel="Least similar image"
                        />
                    <Button
                        onPress={() => this.request({most_similar: 1})}
                        title="Most sim."
                        color="#999999"
                        style={styles.compare_inv_button}
                        accessibilityLabel="Most similar image"
                        />
                </View>
            </View>
            <View style={dir.compare_pane}>
                {this.state.item3 && 
                <File
                    in_compare={true}
                    imgurl={this.state.imgurl}
                    item={this.state.item3}
                    next={this.state.item3}
                    cols={1}
                    info_top={true}
                    info={this.state.info}
                    setinfo={info => this.setState({info})}
                >
                    <View style={styles.blklst}>
                    <Button
                        onPress={() => this.request({dislike: [0,0,1]})}
                        title="blklst"
                        color="#be3e2e"
                        accessibilityLabel="blklst"
                        />
                    </View>
                </File>}
                <View style={this.state.derp ? styles.compare_info_bottom : styles.compare_info_bottom2}>
                    <Button
                        onPress={() => this.request({least_similar: 2})}
                        title="Least sim."
                        color="#999999"
                        accessibilityLabel="Least similar image"
                        />
                    <Button
                        onPress={() => this.request({most_similar: 2})}
                        title="Most sim."
                        color="#999999"
                        style={styles.compare_inv_button}
                        accessibilityLabel="Most similar image"
                        />
                </View>
            </View>
        </View>;
    }
}
const Similarity = with_redirector(Similarity_);

class History_ extends React.Component {
    constructor(props) {
        super(props);
        var {height, width} = Dimensions.get('window');
        this.state = {
            col_width: width / Math.round(width/512),
            imgurl: undefined,
            info: true,
            children: [],
            hor: global_hor,
        };
        this.geturl();
    }
    async geturl() {
        var url = await imgurl("", true);
        this.setState({imgurl: url});
    }
    componentDidMount() {
        this.request();
    }
    async request() {
        try {
            this.setState({loading: true});
            var path = this.rest();
            var resp = await fetchpath("/api/history/" + encodeURIComponent(path));
            var j = await resp.json();
            if (path !== this.rest()) {
                return;
            }
            var jc = j.history.children;
            var c = convert_jc(jc, x => -x.ratio);
            var values = {
                item: j.ref,
                children: c,
                loaded: path,
                loading: false,
            };
            this.setState(values);
        } catch (e) {
            console.warn(e);
            if (e.status == 401) {
                this.props.redirect("/_/settings/");
            } else {
                throw e;
            }
        }
    }
    rest(props) {
        if (props === undefined) {
            props = this.props;
        }
        return (props.match.params.rest || "")
    }
    componentDidUpdate(prevProps, prevState, snapshot) {
        if (this.rest(prevProps) !== this.rest()) {
            this.request();
        }
    }
    render() {
        screen_key_callbacks = {};
        screen_key_callbacks['t'] = () => this.setState({hor: !this.state.hor}, () => {global_hor = this.state.hor});
        screen_key_callbacks['d'] = () => this.setState({info: !this.state.info, children: this.state.children.slice()});
        //screen_key_callbacks['i'] = () => this.props.history.goBack();
        screen_key_callbacks['i'] = () => this.props.redirect("/_/compare/");
        //screen_key_callbacks['n'] = () => this.props.redirect("/_/compare/");
        //screen_key_callbacks['j'] = (() => { if (this.state.prev && (this.state.prev.vpath||this.state.prev.path)) { this.props.redirect("/_/p" + (this.state.prev.vpath||this.state.prev.path))}});
        //screen_key_callbacks['i'] = () => this.props.redirect("/_/compare/" + this.state.item.hash + "/");
        //screen_key_callbacks['o'] = (() => { if (this.state.next && (this.state.next.vpath||this.state.next.path)) { this.props.redirect("/_/p" + (this.state.next.vpath||this.state.next.path))}});
        //screen_key_callbacks['p'] = () => this.props.redirect("/_/compare/");

        //screen_key_callbacks['u'] = () => this.request({undo: true, fast: true});
        //screen_key_callbacks[','] = () => this.request({incomparable: true, goes_well: true, fast: true});

        screen_key_callbacks['['] = () => this.props.history.goBack();
        screen_key_callbacks[']'] = () => this.props.history.goForward();
        screen_key_callbacks['Backspace'] = () => this.props.history.goBack();
        screen_key_callbacks['Shift-Backspace'] = () => this.props.history.goForward();

        var {height, width} = Dimensions.get('window');
        var ww = width/this.state.col_width;
        var cols = Math.round(ww);
        var icols = cols;
        if (!isFinite(cols) || cols == 0) {
            cols = 1;
            
        }
        if (cols > 8) { cols = 6; }

        var dir = {
            browser: styles.browser,
            compare_pane: styles.compare_pane,
            compare_buttons: styles.compare_buttons,
            too_close: "Too Close",
            incomparable:"Incomparable",
            goes: "Goes Well",
            undo: "Undo",
            transpose: "Transpose"
        };
        if (this.state.hor) {
            dir = {
                browser: styles.browser_hor,
                compare_pane: styles.compare_pane_hor,
                compare_buttons: styles.compare_buttons_hor,
                too_close: "==",
                incomparable:"!=",
                goes: "+=",
                undo: "U",
                transpose: "T",
            };
        }
        //<Text>cols: {cols}, height: {height}, width: {width}, cw: {this.state.col_width}, ww: {ww}, rww:  icols: {icols}</Text>
        return <View style={dir.browser}>
            <View style={dir.compare_pane}>
                {!this.state.loading && this.state.children &&
                    <View style={styles.directory}>
                        <FlatList
                            windowSize={2}
                            initialNumToRender={4}
                            key={"" + cols}
                            data={this.state.children}
                            style={styles.browserlist}
                            renderItem={({item}) => <FileEntry
                                imgurl={this.state.imgurl}
                                item={item}
                                info={this.state.info}
                                smallinfo={false}
                                nextpath={"/_/compare/" + this.state.item.hash + "/" + item.hash}
                                cellwidth={width/cols}
                            />}
                            numColumns={cols}
                            />
                    </View>
                }
            </View>
            <View style={dir.compare_pane}>
                {this.state.item && 
                <File
                    in_compare={true}
                    imgurl={this.state.imgurl}
                    item={this.state.item}
                    next={this.state.item}
                    cols={1}
                    info={this.state.info}
                    setinfo={info => this.setState({info})}
                >
                </File>}
            </View>
        </View>
    }
}
const History = with_redirector(History_);

const App = () => (
    <View style={styles.container}>
        <Router>
            <Route exact path="/" component={Home}/>
            <Route exact path="/_/settings/" component={Settings}/>
            <Route sensitive path="/_/compare/:rest*" component={Compare}/>
            <Route sensitive path="/_/history/:rest*" component={History}/>
            <Route sensitive path="/_/similarity/:rest*" component={Similarity
            
            }/>
            <Route sensitive path="/_/p/:rest*" component={Browser}/>
        </Router>
    </View>
)

const styles = StyleSheet.create({
    container: {
        backgroundColor: '#333333',
        height: '100%',
        //padding: 10,
    },
    header: {
        fontSize: 20,
    },
    nav: {
        flexDirection: 'row',
        justifyContent: 'space-around'
    },
    navItem: {
        flex: 1,
        alignItems: 'center',
        padding: 10,
    },
    subNavItem: {
        padding: 5,
    },
    topic: {
        textAlign: 'center',
        fontSize: 15,
    },
    apiurl_field: {
        height: 40,
        color: '#ffffff',
    },
    directory: {
        flex: 1,
    },
    browser: {
        flex: 1,
    },
    browser_hor: {
        flex: 1,
        flexDirection: 'row-reverse',
    },
    compare_pane: {
        flex: 1,
        position: 'relative',
    },
    compare_button: {
        marginLeft: 8,
    },
    compare_pane_hor: {
        flex: 1,
        position: 'relative',
    },
    compare_buttons_hor: {
        width: 24,
        marginLeft: 4,
        marginRight: 4,
        flexDirection: "column",
        position: 'relative',
    },
    compare_buttons: {
        height: 32,
        justifyContent: "space-between",
        marginTop: 4,
        marginBottom: 4,
        flexDirection: "row",
        position: 'relative',
    },
    compare_info_top2: {
        position: 'absolute',
        bottom: 80,
        right: 0,
        flexDirection: "column",
        justifyContent: "space-between",

    },
    compare_info_bottom2: {
        position: 'absolute',
        bottom: 130,
        right: 0,
        flexDirection: "column",
        justifyContent: "space-between",
        alignItems: "flex-end",
    },
    compare_info_middle2: {
        position: 'absolute',
        bottom: 80,
        right: 0,
        flexDirection: "column",
        justifyContent: "space-between",

    },
    compare_info_middle: {
        position: 'absolute',
        bottom: 60,
        right: 0,
        flexDirection: "column",
        justifyContent: "space-between",

    },
    compare_info_top: {
        position: 'absolute',
        bottom: 40,
        right: 0,
        flexDirection: "column",
        justifyContent: "space-between",

    },
    compare_info_bottom: {
        position: 'absolute',
        bottom: 90,
        right: 0,
        flexDirection: "column",
        justifyContent: "space-between",
        alignItems: "flex-end",
    },
    info_top: {
        position: 'absolute',
        top: 0,
        left: 0,
        flexDirection: "row",
        alignItems: "flex-start",
        padding: 5,
    },
    info_bottom: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        padding: 5,
        paddingBottom: 20,
        flexDirection: "row",
        alignItems: "flex-end",
    },
    info_col: {
        flexDirection: "column",
        justifyContent: "space-between",
    },
    info_col_bottom: {
        flexDirection: "column-reverse",
        justifyContent: "space-between",
    },
    info_small: {
        position: 'absolute',
        top: 0,
        left: 0,
        backgroundColor: 'rgba(52, 52, 52, 0.3)',
        flexDirection: "column",
        flexGrow: 1,
        padding: 5,
    },
    info_main: {
        backgroundColor: 'rgba(52, 52, 52, 0.3)',
        flexDirection: "column",
        flexGrow: 1,
        padding: 5,
    },
    vbutton: {
        marginTop: 3,
        marginBottom: 3,
    },
    browserlist: {
    },
    fileentry: {
        flex: 1,
        flexDirection: 'column',
        alignItems: 'stretch',
    }, 
    fileentry_metadata: {
        position: 'absolute',
        top: 0,
        left: 0,
    },
    fileentry_title: {
        fontWeight: "bold",
        fontSize: 15,
        color: '#ffffff',
    },
    fileentry_magic: {
        color: '#ffffff',
    },
    fileentry_content: {
    },
    breadcrumbs: {
        borderWidth: 2,
        borderColor: "#00ff00",
        backgroundColor: '#ff00ff',
        flexDirection: "row",
        flexWrap: "wrap",
    },
    breadcrumb: {
        flexDirection: "row",
    },
    breadcrumb_inner: {
        flexDirection: "row",
    },
    breadcrumb_text: {
        fontWeight: "bold",
        color: '#ffffff',
    },
    breadcrumb_separator: {
    },
    singlefile: {
        flex: 1,
        position: 'relative',
    },
    textstyle: {
        color: '#ffffff',
    },
    textstyle_small: {
        color: '#ffffff',
        fontSize: 10,
    },
    singlefile_content: {
        flex: 1,
        resizeMode: 'contain',
    },
    biglink: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
    },
    blklst: {
        marginTop: 40,
        marginBottom: 40
    },
    compare_inv_button: {
        marginTop: 20
    }

})

export default App;
