# Chloris

## Introduction
The Hiking Companion app is designed to enhance the hiking experience by providing users with valuable information about the flora and fauna they encounter on their hikes. Leveraging advanced technologies such as machine learning, GIS satellite data, and real-time monitoring, the app offers a range of features to assist users in identifying birds, flowers, and environmental conditions along hiking trails.

## Methodology

### Data Collection
- **Birds and Flowers Dataset**: Curated datasets containing images and information about various bird species and flower types are used to train the identification models.
- **GIS Satellite Data**: Real-time weather and environmental data are obtained from GIS satellite sources to provide users with up-to-date information about hiking conditions.

### Machine Learning Models
- **ResNet18 Model**: A pre-trained ResNet18 neural network is fine-tuned using transfer learning to classify bird species and flower types based on user-submitted images.
- **Laplace Smoothing**: Probability estimation techniques, such as Laplace smoothing, are applied to predict the likelihood of encountering birds and flowers along hiking trails.

### Alert System
- **User Preferences**: Users can set up personalized alerts for bird sightings and flower bloomings based on their preferences.
- **Real-time Monitoring**: The app continuously monitors environmental data and user contributions to detect the first bloom of the season for various flower species.
- **Notification System**: Users receive push notifications for significant events, such as the first bloom of the season or rare bird sightings.

## Features

### Identification Features
- **Bird Identification**: Users can take photos of birds and receive instant identification results along with additional information about the species.
- **Flower Identification**: Similar to bird identification, users can identify flowers by capturing photos using the app.

### Environmental Information
- **Real-time Weather Updates**: Users can access real-time weather forecasts and conditions for hiking locations.
- **Seasonal Tracking**: The app provides information about the blooming seasons of different flower species, helping users plan their hikes accordingly.

### Community Engagement
- **User Contributions**: Users can contribute to the app by reporting bird sightings, flower blooms, and other observations along hiking trails.
- **Interactive Map**: An interactive map allows users to explore hiking trails, record sightings, and access community-contributed data.

## Privacy and Data Security
- **Opt-in/Opt-out**: Users have control over the types of notifications they receive and can opt in or out of sharing their location data.
- **Data Encryption**: User data is encrypted to ensure privacy and security.

## Conclusion
The Hiking Companion app combines advanced technology with user-friendly features to create an immersive and informative hiking experience. By leveraging machine learning, GIS satellite data, and community engagement, the app empowers users to explore nature with confidence and appreciation.
