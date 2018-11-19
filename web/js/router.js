import { Platform } from 'react-native';

const isWeb = Platform.OS === 'web';
const RouterPackage = isWeb ? require('react-router-dom') : require('react-router-native');
const Router = isWeb  ? RouterPackage.BrowserRouter  : 
RouterPackage.NativeRouter;
export Router;
export default RouterPackage;
