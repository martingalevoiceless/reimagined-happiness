import "babel-polyfill";

import App from './core';
import React from 'react';
import { AppRegistry, StyleSheet, Text, View } from 'react-native-web';

AppRegistry.registerComponent('App', () => App);
AppRegistry.runApplication('App', { rootTag: document.getElementById('react') });
console.log("enter");
