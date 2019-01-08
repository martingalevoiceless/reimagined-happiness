import React from 'react';

import {Platform, StyleSheet, View} from 'react-native';

import {
    Route as ORoute,
    Redirect as ORedirect,
    NativeRouter as ORouter1,
    Link as OLink,
    BackButton,
    BrowserRouter as ORouter2,
    withRouter as withRouterReact,
    NavLink as ONavLink,
    Switch as OSwitch,
} from "react-router-native";

var ORouter;
if (ORouter1 !== undefined) {
    ORouter = ORouter1;
} else {
    ORouter = ORouter2;
}

export const Ctx = React.createContext({
    history: {},
    location: null,
    match: null
})

export function with_router(Component) {
    return function RoutedComponent(props) {
        return (
            <Ctx.Consumer>
                {extra_props => {
                    var props_inner = Object.assign({}, extra_props);
                    props_inner = Object.assign(props_inner, props);
                    if (Platform.OS == 'web' && (typeof props_inner.style) === (typeof 1)) {
                        props_inner.style = StyleSheet.flatten(props_inner.style);
                    }
                    return <Component {...props_inner}/>
                }}
            </Ctx.Consumer>
        );
    }
}
export const withRouter = with_router;

function CtxProvider(props) {
    return <Ctx.Provider value={{match: props.match, location: props.location, history:props.history}}>
        {props.children}
    </Ctx.Provider>
}
const Provider = withRouterReact(CtxProvider);

export function Router(props) {
    if (BackButton !== undefined) {
        return <ORouter>
            <BackButton>
                <Provider>
                    {props.children}
                </Provider>
            </BackButton>
        </ORouter>;
    } else {
        return <ORouter>
            <Provider>
                {props.children}
            </Provider>
        </ORouter>;
    }
}
export function with_redirector(Component) {
    class RedirectableComponentBase extends React.Component {
        constructor(props) {
            super(props);
            this.state = {
                redirect_to: null
            };
        }
        go(path, push=true) {
            this.setState({redirect_to: path, push});
        }
        render() {
            var redirect = null;
            if (this.state.redirect_to != null) {
                redirect = <Redirect push={this.state.push !== undefined ? this.state.push : true} to={this.state.redirect_to}/>;
            }
            return <React.Fragment>
                {redirect}
                <Component {...this.props} redirect={(path, push=true) => this.go(path, push)} location={this.props.location} match={this.props.match}/>
            </React.Fragment>
        }
        static getDerivedStateFromProps(nextProps, prevState) {
            if (prevState.redirect_to != null && nextProps.location !== prevState.location) {
                return {location: nextProps.location, redirect_to: null};
            }
            return {location: nextProps.location};
        }
    }
    return with_router(RedirectableComponentBase);
}

export const wrap_children = Elem => (({children, ...props}) => (
    <Elem {...props}>
        <View style={{flex: 1}}>
            {children}
        </View>
    </Elem>
));

export const Route    = with_router(ORoute);
export const Link     = wrap_children(with_router(OLink));
export const NavLink  = wrap_children(with_router(ONavLink));
export const Switch   = with_router(OSwitch);
export const Redirect = with_router(ORedirect);

