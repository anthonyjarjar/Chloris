import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Platform, ActivityIndicator } from 'react-native';
import { ScrollView, GestureHandlerRootView } from 'react-native-gesture-handler';
import * as Location from 'expo-location';
import { Picker } from '@react-native-picker/picker';
import { StatusBar } from 'expo-status-bar';

export default function LocationPrediction({ navigation }) {
  const [loading, setLoading] = useState(false);
  const [locationPrediction, setLocationPrediction] = useState(null);
  const [locationProbability, setLocationProbability] = useState(null);
  const [birdSpecies, setBirdSpecies] = useState('Dark-eyed Junco');

  const getLocation = async () => {
    setLoading(true);
    try {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        alert('Permission to access location was denied');
        return;
      }

      let location = await Location.getCurrentPositionAsync({});
      predictLocation(location.coords.longitude, location.coords.latitude);
    } catch (error) {
      console.error('Error getting location:', error);
    } finally {
      setLoading(false);
    }
  };

  const predictLocation = async (longitude, latitude) => {
    setLoading(true);
    try {
      const response = await fetch('http://192.168.1.101:8000/predict_location', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          longitude,
          latitude,
          birdSpecies,
        }),
      });
  
      const result = await response.json();
      setLocationPrediction(result.prediction === 1 ? 'New bird sighting' : 'Not a new bird sighting');
      setLocationProbability(result.probability);
    } catch (error) {
      console.error('Error predicting location:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Sighting Probability</Text>
  
        <Picker
          selectedValue={birdSpecies}
          style={styles.picker}
          onValueChange={(itemValue, itemIndex) => setBirdSpecies(itemValue)}
        >
          <Picker.Item label="American Robin" value="American Robin" />
          <Picker.Item label="American Goldfinch" value="American Goldfinch" />
          <Picker.Item label="Blue Jay" value="Blue Jay" />
          <Picker.Item label="Cardinal" value="Cardinal" />
          <Picker.Item label="Dark-eyed Junco" value="Dark-eyed Junco" />
          <Picker.Item label="Downy Woodpecker" value="Downy Woodpecker" />
        </Picker>

        <TouchableOpacity style={styles.predictButton} onPress={getLocation}>
          <Text style={styles.buttonText}>Calculate Probability</Text>
        </TouchableOpacity>
  
        {loading ? (
          <ActivityIndicator size="large" color="#3498db" />
        ) : (
          locationPrediction && (
            <View>
              <Text style={styles.locationPrediction}>{locationPrediction} (Probability {locationProbability}%) </Text>
            </View>
          )
        )}
  
        <StatusBar style="auto" />
      </ScrollView>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: '#072A0C',
    alignItems: 'center',
    padding: 20,
    paddingTop: Platform.OS === 'ios' ? 80 : 20,
    paddingBottom: Platform.OS === 'ios' ? 20 : 20,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#27ae60',
  },
  predictButton: {
    backgroundColor: '#27ae60',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  locationPrediction: {
    marginTop: 20,
    fontSize: 20,
    fontWeight: 'bold',
    color: '#27ae60',
  },
  divider: {
    borderBottomColor: '#27ae60',
    borderBottomWidth: 3,
    width: '108%',
    marginVertical: 20,
  },
  picker: {
    width: '100%',
    backgroundColor: '#27ae60',
    color: '#fff',
    marginBottom: 20,
    borderRadius: 10,
  },
});
