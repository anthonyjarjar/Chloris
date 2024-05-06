import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import Home from './Home'; 
import ImageUpload from './ImageUpload';
import LocationPrediction from './LocationPrediction';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={Home} />
        <Stack.Screen name="ImageUpload" component={ImageUpload} />
        <Stack.Screen name="LocationPrediction" component={LocationPrediction} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
