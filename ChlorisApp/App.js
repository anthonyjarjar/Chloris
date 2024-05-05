import React, { useState, useEffect } from 'react';
import { View, Text, Image, TouchableOpacity, StyleSheet, Platform, ActivityIndicator } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';
import { ScrollView, GestureHandlerRootView } from 'react-native-gesture-handler';
import * as Location from 'expo-location';
import { Picker } from '@react-native-picker/picker';
import birdLabels from './bird_names_dict.js';

export default function App() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [locationPrediction, setLocationPrediction] = useState(null);
  // Location state
  const [longitude, setLongitude] = useState(null);
  const [latitude, setLatitude] = useState(null);

  // Bird species state
  const [birdSpecies, setBirdSpecies] = useState('Black-capped Chickadee'); // Default to Black-capped Chickadee

  useEffect(() => {
    (async () => {
      if (Platform.OS !== 'web') {
        // Request media library permissions
        const mediaPermission = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (mediaPermission.status !== 'granted') {
          alert('Sorry, we need camera roll permissions to make this work!');
        }

        // Request location permissions
        const locationPermission = await Location.requestForegroundPermissionsAsync();
        if (locationPermission.status !== 'granted') {
          alert('Sorry, we need location permissions to make this work!');
        } else {
          // Fetch location data
          getLocation();
        }
      }
    })();
  }, []);

  // Function to pick image from gallery
  const pickImage = async () => {
    console.log('Uploading image...');
    setLoading(true);
  
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
    });  

    // Check if the action was cancelled by the user
    if (!result.cancelled && result.assets && result.assets.length > 0) {
      const imageUri = result.assets[0].uri;  // Correctly access the URI from the assets array
      console.log('Image URI:', imageUri);
      setImage(imageUri);  // Set the image URI to state
      predictImage(imageUri);  // Pass the URI to the predictImage function
    } else {
      setLoading(false);
    }
  };

  // Function to predict image
  const predictImage = async (selectedImageURI) => {
    console.log('Predicting image...', selectedImageURI);
    const formData = new FormData();
    formData.append('image', { uri: selectedImageURI, name: 'image.jpg', type: 'image/jpeg' });

    try {
      const response = await fetch('http://192.168.1.101:8000/predict', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const result = await response.json();
      console.log('Server Response:', result);

      const predictionLabel = result;
      const birdPredict = birdLabels[predictionLabel + 1];

      console.log(predictionLabel, birdPredict);
      setPrediction(birdPredict);
    } catch (error) {
      console.error('Error predicting image:', error);
    } finally {
      setLoading(false);
    }
  };

  // Function to get current location
  const getLocation = async () => {
    setLoading(true);
    try {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        alert('Permission to access location was denied');
        return;
      }

      let location = await Location.getCurrentPositionAsync({});
      setLongitude(location.coords.longitude);
      setLatitude(location.coords.latitude);
    } catch (error) {
      console.error('Error getting location:', error);
    } finally {
      setLoading(false);
    }
  };

  // Function to predict location
  const predictLocation = async () => {
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

      console.log(longitude, latitude, birdSpecies)
      const result = await response.json();
      setLocationPrediction(result.prediction === 1 ? 'New bird sighting' : 'Not a new bird sighting');
    } catch (error) {
      console.error('Error predicting location:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Chloris</Text>
  
        <View style={styles.divider}></View>

        <View style={styles.imageUploadContainer}>
          <TouchableOpacity style={styles.uploadButton} onPress={pickImage}>
            <Text style={styles.buttonText}>Upload Image</Text>
          </TouchableOpacity>
  
          {image && (
            <View style={styles.imageContainer}>
              <Image source={{ uri: image }} style={styles.image} resizeMode="contain" />
              {prediction && <Text style={styles.prediction}>{prediction}</Text>}
            </View>
          )}
        </View>
        

        <View style={styles.divider}></View>


        <Picker
          selectedValue={birdSpecies}
          style={styles.picker}
          onValueChange={(itemValue, itemIndex) => setBirdSpecies(itemValue)}
        >
          <Picker.Item label="Black-Eyed Junco" value="Black-Eyed Junco" />
          <Picker.Item label="American Robin" value="American Robin" />
          <Picker.Item label="American Goldfinch" value="American Goldfinch" />
          <Picker.Item label="Blue Jay" value="Blue Jay" />
          <Picker.Item label="Cardinal" value="Cardinal" />
          <Picker.Item label="Downy Woodpecker" value="Downy Woodpecker" />
          {/* Add more bird species here */}
        </Picker>
  
        <TouchableOpacity style={styles.predictButton} onPress={predictLocation}>
          <Text style={styles.buttonText}>Predict Location</Text>
        </TouchableOpacity>
  
        {loading ? (
          <ActivityIndicator size="large" color="#3498db" />
        ) : (
          locationPrediction && <Text style={styles.locationPrediction}>{locationPrediction}</Text>
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
    color: '#27ae60', // Blue color
  },
  imageUploadContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  uploadButton: {
    backgroundColor: '#27ae60', // Green color
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  imageContainer: {
    alignItems: 'center',
  },
  image: {
    width: 300,
    height: 200,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#27ae60', // Green color
  },
  prediction: {
    marginTop: 10,
    fontSize: 18,
    fontWeight: 'bold',
    color: '#27ae60', // Green color
  },
  picker: {
    width: '100%',
    backgroundColor: '#27ae60', // Green color
    color: '#fff',
    marginBottom: 20,
    borderRadius: 10,
  },
  predictButton: {
    backgroundColor: '#27ae60', // Green color
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
  },
  locationPrediction: {
    marginTop: 20,
    fontSize: 20,
    fontWeight: 'bold',
    color: '#27ae60', // Green color
  },
  divider: {
    borderBottomColor: '#27ae60',
    borderBottomWidth: 3,
    width: '108%',
    marginVertical: 20,
  },
});
