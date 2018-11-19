const path = require('path');
const webpack = require('webpack');

module.exports = {
    entry: {
        entrypoint: ["babel-polyfill", './js/entrypoint.js'],
    },
    optimization: {
        splitChunks: {
            chunks: "initial",
        },
    },
    output: {
        filename: '[name].js',
        path: path.resolve(__dirname, 'static/dist')
    },
    module: {
        rules: [
            {
                test: /\.ttf$/,
                loader: "url-loader", // or directly file-loader
                include: path.resolve(__dirname, "node_modules/react-native-vector-icons"),
            },
            {
                test: /\.js$/,
                exclude: /node_modules/,
                loader: "babel-loader",
            },
            {
                test: /\.css$/,
                use: [ 'style-loader', 'css-loader' ]
            }
        ]
    },
    resolve: {
        modules: [path.resolve(__dirname, "./js"), "node_modules"],
        extensions: ['.js', '.css'],
        symlinks: false,
        alias: {
            'react-native$': 'react-native-web',
            'react-router-native$': 'react-router-dom',
        }
    }
};
