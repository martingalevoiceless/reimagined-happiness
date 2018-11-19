

//export {default as FontAwesome} from 'react-native-vector-icons/FontAwesome';
export {default as MaterialIcons} from 'react-native-vector-icons/dist/MaterialIcons';
export {default as MaterialCommunityIcons} from 'react-native-vector-icons/dist/MaterialCommunityIcons';

import materialIconsFont from 'react-native-vector-icons/Fonts/MaterialIcons.ttf';
import materialCommunityIconsFont from 'react-native-vector-icons/Fonts/MaterialCommunityIcons.ttf';
var fonturls = [
    ["MaterialIcons", materialIconsFont],
    ["MaterialCommunityIcons", materialCommunityIconsFont],
];
var icon_font_styles = "";
for (var fonturl of fonturls) {
    icon_font_styles += `@font-face {
    src: url(${fonturl[1]});
    font-family: ${fonturl[0]};
    }\n`;
}
 
// Create stylesheet
const style = document.createElement('style');
style.type = 'text/css';
if (style.styleSheet) {
  style.styleSheet.cssText = icon_font_styles;
} else {
  style.appendChild(document.createTextNode(icon_font_styles));
}
 
// Inject stylesheet
document.head.appendChild(style);
