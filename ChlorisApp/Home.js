import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

export default function Home({ navigation }) {
  const goToImageUpload = () => {
    navigation.navigate('ImageUpload');
  };

  const goToLocationPrediction = () => {
    navigation.navigate('LocationPrediction');
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Chloris</Text>
      <TouchableOpacity style={styles.button} onPress={goToImageUpload}>
        <Text style={styles.buttonText}>Go to Image Upload</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.button} onPress={goToLocationPrediction}>
        <Text style={styles.buttonText}>Go to Location Prediction</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#072A0C',
    alignItems: 'center',
    paddingTop: 150,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#27ae60',
  },
  button: {
    backgroundColor: '#27ae60',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
    width: '100%',
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});
