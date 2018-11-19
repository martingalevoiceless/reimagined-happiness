import React from 'react'

export class UtilComponent extends React.Component {
    constructor(props) {
        super(props);
        var orig_render
        if (!this.render) {
            console.log("missing render method, replacing with error placeholder", this);
            orig_render = () => <div> error - no render method on {"" + this} </div>;
        } else {
            orig_render = this.render.bind(this);
        }
        this.render = () => orig_render(this.props, this.state);

        this.onunmount = [];
        this.dying = false;
        var orig_componentWillUnmount;
        if (this.componentWillUnmount) {
            orig_componentWillUnmount = this.componentWillUnmount.bind(this);
        }
        this.componentWillUnmount = (a, b, c, d, e, f) => {
            this.dying = true;
            if (orig_componentWillUnmount) {
                orig_componentWillUnmount(a, b, c, d, e, f);
            }
            for (var hook of this.onunmount) {
                hook();
            }
        }
    }
    prop(name, def) {
        var prop = this.props[name];
        if (prop === undefined || prop === null) {
            return def;
        }
        return prop;
    }
}
