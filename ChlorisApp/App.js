import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import Home from './Home'; // Import the Home component
import ImageUpload from './ImageUpload'; // Import the ImageUpload component
import LocationPrediction from './LocationPrediction'; // Correct casing

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
