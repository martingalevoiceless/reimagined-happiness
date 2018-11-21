
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


const Content = ({item, style, cellwidth, imgurl, onClick, big}) => {
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
            style={style}
            source={src}
            onClick={onClick}
            controls={big}
            autoplay={big}
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
        this.state = {
            info: false
        };
    }
    render() {
        var hw = {height: this.props.cellwidth, maxWidth: this.props.cellwidth};
        return <View style={[styles.fileentry, hw]}>
            <Link to={"/_/p" + (this.props.item.vpath || this.props.item.path)}>
                <Content
                    style={[styles.fileentry_content, hw]}
                    item={this.props.item}
                    imgurl={this.props.imgurl}
                    cellwidth={this.props.cellwidth}
                    big={false}
                />
            </Link>
        </View>
    }
}
const FileEntry = with_redirector(FileEntry_);

class SingleFile_ extends UtilComponent {
    constructor(props) {
        super(props);
        this.state = {
            info: false
        };
    }
    render({item, imgurl, next, onClick}) {
        return <View style={styles.singlefile}>
            <Content
                style={styles.singlefile_content}
                item={item}
                imgurl={imgurl}
                onClick={onClick}
                big={true}
            />
            {next && 
            <Link to={"/_/p" + (next||{}).path} style={styles.biglink}>
            </Link>
            }
            <View
                style={this.props.info_top ? styles.info_top : styles.info_bottom}>
            {this.state.info
                ? 
                        <React.Fragment>
                            <Button
                                onPress={() => this.setState({info: false})}
                                title="hide"
                                color="#999999"
                                accessibilityLabel="hide info"
                                />
                            {this.props.children}

                            <Button
                                onPress={() => this.props.redirect("/_/compare/" + this.props.item.hash + "/")}
                                title="compare"
                                accessibilityLabel="compare"
                                />
                            <Button
                                onPress={() => this.props.redirect("/_/p" + (this.props.item.path))}
                                title="view"
                                accessibilityLabel="view"
                                />
                        </React.Fragment>

                :
                    <Button
                        onPress={() => this.setState({info: true})}
                        title="info"
                        color="#999999"
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
                key={"" + cols}
                data={this.prop("children", [])}
                style={styles.browserlist}
                renderItem={({item}) => <FileEntry
                    imgurl={this.props.imgurl}
                    item={item}
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
function convert_jc(jc) {
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
    sortby(c, x => [!(x.dir), x.name]);
    return c;
}

class Browser extends React.Component {
    constructor(props) {
        super(props);
        var {height, width} = Dimensions.get('window');
        this.state = {
            col_width: width / Math.round(width/128),
            imgurl: undefined,
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
        var breadcrumbs = [];
        var subpaths = this.subpaths();
        for (var subpath of subpaths) {
            if (breadcrumbs.length) {
                breadcrumbs.push(
                    <View style={styles.breadcrumb_separator} key={"sep_" + subpath.subpath}>
                        <Text style={styles.breadcrumb_separator_text}> &gt; </Text>
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
                prev={this.state.prev}
                next={this.state.next}
                imgurl={this.state.imgurl}
                item={this.state.item}
                children={this.state.children}
                cols={cols}
            />}
        </View>
    }
}

var compare_callbacks = {};
keyboard(function (event) {
    var callback = compare_callbacks[event.key];
    if (callback) {
        event.preventDefault();
    }
    if (callback && !event.altKey) {
        callback();
    }
});

class Compare_ extends React.Component {
    constructor(props) {
        super(props);
        this.state = {};
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
            if (comparecache[path] !== undefined && preference === undefined) {
                j = comparecache[path];
            } else {
                var extra = {};
                if (preference !== undefined) {
                    extra = {headers: {"content-type": "application/json"}, "method": "PUT", "body": JSON.stringify({
                        viewstart: this.state.viewstart,
                        viewend: +(new Date()),
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
            if (j.path !== this.rest()) {
                console.log(j.path, this.rest(), "redirect");
                this.props.redirect("/_/compare/" + j.path);
            }
            var values = {
                j: JSON.stringify(j),
                loadedpath: j.path,
                viewstart: +(new Date()),
                item1: convert_c(j.item1),
                prob1: j.proba,
                item2: convert_c(j.item2),
                prob2: j.probb,
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
        compare_callbacks['u'] = () => this.request({incomparable: true});
        compare_callbacks['i'] = () => this.request({prefer: 1});

        compare_callbacks['j'] = () => this.request({prefer: 2});

        compare_callbacks['o'] = () => this.request({too_close: true});
        compare_callbacks['h'] = () => this.request({incomparable: true, goes_well: true});

        return <View style={styles.browser}>
            <View style={styles.compare_pane}>
                {this.state.item1 && 
                <File
                    imgurl={this.state.imgurl}
                    item={this.state.item1}
                    cols={1}
                    info_top={true}
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
                <View style={styles.compare_info_top}>
                    <Button
                        onPress={() => this.request({prefer: 1})}
                        title="Prefer"
                        accessibilityLabel="Prefer"
                        />
                </View>
            </View>
            <View style={styles.compare_buttons}>
                <Button
                    onPress={() => this.request({too_close: true})}
                    title="Too Close"
                    accessibilityLabel="Too close to call"
                    />
                <Button
                    onPress={() => this.request({incomparable: true})}
                    title="Incomparable"
                    accessibilityLabel="Doesn't make sense to compare"
                    />
                <Button
                    onPress={() => this.request({incomparable: true, goes_well: true})}
                    title="goes well"
                    />
            </View>
            <View style={styles.compare_pane}>
                {this.state.item2 && 
                <File
                    imgurl={this.state.imgurl}
                    item={this.state.item2}
                    cols={1}
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
                <View style={styles.compare_info_bottom}>
                    <Button
                        onPress={() => this.request({prefer: 2})}
                        title="Prefer"
                        accessibilityLabel="Prefer"
                        />
                </View>
            </View>
        </View>;
    }
}
const Compare = with_redirector(Compare_);


const App = () => (
    <View style={styles.container}>
        <Router>
            <Route exact path="/" component={Home}/>
            <Route exact path="/_/settings/" component={Settings}/>
            <Route sensitive path="/_/compare/:rest*" component={Compare}/>
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
        paddingTop: 25,
        flex: 1,
    },
    compare_pane: {
        flex: 1,
        position: 'relative',
    },
    compare_buttons: {
        height: 32,
        marginTop: 4,
        marginBottom: 4,
        flexDirection: "row",
        position: 'relative',
    },
    compare_info_top: {
        position: 'absolute',
        top: 0,
        right: 0,
        flexDirection: "row",
        justifyContent: "space-between",

    },
    compare_info_bottom: {
        position: 'absolute',
        bottom: 0,
        right: 0,
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "flex-end",
    },
    info_top: {
        position: 'absolute',
        top: 0,
        left: 0,
        flexDirection: "column",
        justifyContent: "space-between",

    },
    info_bottom: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        flexDirection: "column-reverse",
        justifyContent: "space-between",
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
    },
    breadcrumb: {
        flex: 1,
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
        flex: 1,
        width: 10,
    },
    breadcrumb_separator_text: {
        color: '#ffffff',
    },
    singlefile: {
        flex: 1,
        position: 'relative',
    },
    singlefile_title: {
        color: '#ffffff',
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
    }

})

export default App;
