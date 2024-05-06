import React, { useState, useEffect } from 'react';
import { View, Text, Image, TouchableOpacity, StyleSheet, Platform, ActivityIndicator } from 'react-native';
import { ScrollView, GestureHandlerRootView } from 'react-native-gesture-handler';
import * as ImagePicker from 'expo-image-picker';
import { StatusBar } from 'expo-status-bar';
import birdLabels from './bird_names_dict.js';

export default function ImageUpload({ navigation }) {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const pickImage = async () => {
    console.log('Uploading image...');
    setLoading(true);
  
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
    });  

    if (!result.cancelled && result.assets && result.assets.length > 0) {
      const imageUri = result.assets[0].uri; 
      console.log('Image URI:', imageUri);
      setImage(imageUri);  
      predictImage(imageUri);  
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
      setPrediction(birdLabels[predictionLabel + 1]);
    } catch (error) {
      console.error('Error predicting image:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Upload Bird Image</Text>
  
        <TouchableOpacity style={styles.uploadButton} onPress={pickImage}>
          <Text style={styles.buttonText}>Upload Image</Text>
        </TouchableOpacity>
  
        {image && (
          <View style={styles.imageContainer}>
            <Image source={{ uri: image }} style={styles.image} resizeMode="contain" />
            {loading ? (
              <ActivityIndicator size="large" color="#3498db" />
            ) : (
              prediction && <Text style={styles.prediction}>{prediction}</Text>
            )}
          </View>
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
  uploadButton: {
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
  imageContainer: {
    alignItems: 'center',
  },
  image: {
    width: 300,
    height: 200,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#27ae60', 
  },
  prediction: {
    marginTop: 10,
    fontSize: 18,
    fontWeight: 'bold',
    color: '#27ae60', 
  },
  divider: {
    borderBottomColor: '#27ae60',
    borderBottomWidth: 3,
    width: '108%',
    marginVertical: 20,
  },
});
