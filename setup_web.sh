#!/bin/bash

cd "$(dirname "$(realpath "$0")")"/web/

npm install
./node_modules/.bin/webpack --config webpack.config.js
