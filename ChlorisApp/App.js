import React, { useState, useEffect } from 'react';
import { View, Text, Image, TouchableOpacity, StyleSheet, Platform, ActivityIndicator } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';
import { ScrollView, GestureHandlerRootView } from 'react-native-gesture-handler';
import birdLabels from './bird_names_dict.js';


export default function App() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    (async () => {
      if (Platform.OS !== 'web') {
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== 'granted') {
          alert('Sorry, we need camera roll permissions to make this work!');
        }
      }
    })();
  }, []);


  const pickImage = async () => {
    console.log('Uploading image...');
    setLoading(true);
  
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
    });  
    // Check if the action was cancelled by the user
    if (!result.canceled && result.assets && result.assets.length > 0) {
      const imageUri = result.assets[0].uri;  // Correctly access the URI from the assets array
      console.log('Image URI:', imageUri);
      setImage(imageUri);  // Set the image URI to state
      predictImage(imageUri);  // Pass the URI to the predictImage function
    } else {
      setLoading(false);
    }
  };
  

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

return (
  <GestureHandlerRootView style={{ flex: 1 }}>
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Chloris</Text>
      <TouchableOpacity style={styles.uploadButton} onPress={pickImage}>
        <Text style={styles.buttonText}>Upload Image</Text>
      </TouchableOpacity>
      {loading ? (
        <ActivityIndicator size="large" color="#3498db" />
      ) : (
        image && (
          <View style={styles.imageContainer}>
            <Image source={{ uri: image }} style={styles.image} resizeMode="contain" />
            {prediction && <Text style={styles.prediction}>{prediction}</Text>}
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
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#27ae60', // Blue color
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
    marginTop: 20,
  },
  image: {
    width: 300,
    height: 200,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#27ae60', // Green color
  },
  prediction: {
    marginTop: 20,
    fontSize: 20,
    fontWeight: 'bold',
    color: '#27ae60', // Green color
  },
});
