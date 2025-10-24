"""
Feature Extraction Module for Fish Health Detector
This module extracts relevant features from preprocessed image and audio data.
"""

import cv2
import numpy as np
import librosa
from typing import Dict, List, Tuple
from sklearn.feature_extraction import image as sk_image
import scipy.stats as stats

class ImageFeatureExtractor:
    """Extracts features from fish images for health analysis."""
    
    def __init__(self):
        self.color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255])
        }
    
    def extract_morphological_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features from fish image.
        
        Args:
            image: Preprocessed fish image
            
        Returns:
            Dictionary of morphological features
        """
        features = {}
        
        # Convert to grayscale for contour detection
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (assumed to be the fish)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Area and perimeter
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            features['area'] = area
            features['perimeter'] = perimeter
            features['area_to_perimeter_ratio'] = area / (perimeter + 1e-6)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['width'] = w
            features['height'] = h
            features['aspect_ratio'] = w / (h + 1e-6)
            
            # Convex hull
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            features['solidity'] = area / (hull_area + 1e-6)
            
            # Moments for shape analysis
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                features['centroid_x'] = moments['m10'] / moments['m00']
                features['centroid_y'] = moments['m01'] / moments['m00']
            else:
                features['centroid_x'] = 0
                features['centroid_y'] = 0
            
            # Ellipse fitting
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                features['ellipse_major_axis'] = max(ellipse[1])
                features['ellipse_minor_axis'] = min(ellipse[1])
                features['ellipse_eccentricity'] = np.sqrt(1 - (features['ellipse_minor_axis'] / features['ellipse_major_axis'])**2)
        else:
            # Default values if no contours found
            for key in ['area', 'perimeter', 'area_to_perimeter_ratio', 'width', 'height', 
                       'aspect_ratio', 'solidity', 'centroid_x', 'centroid_y', 
                       'ellipse_major_axis', 'ellipse_minor_axis', 'ellipse_eccentricity']:
                features[key] = 0.0
        
        return features
    
    def extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract color-based features from fish image.
        
        Args:
            image: Preprocessed fish image
            
        Returns:
            Dictionary of color features
        """
        features = {}
        
        # Convert to different color spaces
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # RGB statistics
        for i, channel in enumerate(['red', 'green', 'blue']):
            channel_data = image[:, :, i].flatten()
            features[f'{channel}_mean'] = np.mean(channel_data)
            features[f'{channel}_std'] = np.std(channel_data)
            features[f'{channel}_skewness'] = stats.skew(channel_data)
            features[f'{channel}_kurtosis'] = stats.kurtosis(channel_data)
        
        # HSV statistics
        for i, channel in enumerate(['hue', 'saturation', 'value']):
            channel_data = hsv[:, :, i].flatten()
            features[f'{channel}_mean'] = np.mean(channel_data)
            features[f'{channel}_std'] = np.std(channel_data)
        
        # Color distribution analysis
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            color_percentage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
            features[f'{color_name}_percentage'] = color_percentage
        
        # Dominant color analysis
        pixels = image.reshape(-1, 3)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        dominant_colors = kmeans.cluster_centers_
        color_percentages = np.bincount(kmeans.labels_) / len(kmeans.labels_)
        
        for i, (color, percentage) in enumerate(zip(dominant_colors, color_percentages)):
            features[f'dominant_color_{i}_r'] = color[0]
            features[f'dominant_color_{i}_g'] = color[1]
            features[f'dominant_color_{i}_b'] = color[2]
            features[f'dominant_color_{i}_percentage'] = percentage
        
        return features
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features from fish image.
        
        Args:
            image: Preprocessed fish image
            
        Returns:
            Dictionary of texture features
        """
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Local Binary Pattern (LBP)
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        
        for i, val in enumerate(lbp_hist):
            features[f'lbp_bin_{i}'] = val
        
        # Gray Level Co-occurrence Matrix (GLCM) features
        from skimage.feature import graycomatrix, graycoprops
        
        # Compute GLCM
        glcm = graycomatrix(gray, distances=[1], angles=[0, 45, 90, 135], 
                           levels=256, symmetric=True, normed=True)
        
        # Extract GLCM properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
        for prop in properties:
            prop_values = graycoprops(glcm, prop)
            features[f'glcm_{prop}_mean'] = np.mean(prop_values)
            features[f'glcm_{prop}_std'] = np.std(prop_values)
        
        # Gabor filter responses
        from skimage.filters import gabor
        
        gabor_responses = []
        for theta in [0, 45, 90, 135]:
            for frequency in [0.1, 0.3, 0.5]:
                real, _ = gabor(gray, frequency=frequency, theta=np.deg2rad(theta))
                gabor_responses.append(real)
        
        for i, response in enumerate(gabor_responses):
            features[f'gabor_response_{i}_mean'] = np.mean(response)
            features[f'gabor_response_{i}_std'] = np.std(response)
        
        return features
    
    def detect_lesions_and_abnormalities(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect potential lesions and abnormalities in fish image.
        
        Args:
            image: Preprocessed fish image
            
        Returns:
            Dictionary of abnormality features
        """
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Edge detection for abnormal patterns
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features['edge_density'] = edge_density
        
        # Blob detection for lesions
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 1000
        params.filterByCircularity = True
        params.minCircularity = 0.1
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        
        features['blob_count'] = len(keypoints)
        if keypoints:
            blob_sizes = [kp.size for kp in keypoints]
            features['avg_blob_size'] = np.mean(blob_sizes)
            features['max_blob_size'] = np.max(blob_sizes)
        else:
            features['avg_blob_size'] = 0
            features['max_blob_size'] = 0
        
        # Intensity variation analysis
        features['intensity_variance'] = np.var(gray)
        features['intensity_range'] = np.max(gray) - np.min(gray)
        
        return features

class AudioFeatureExtractor:
    """Extracts features from audio data for fish health analysis."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features from audio signal.
        
        Args:
            audio: Preprocessed audio signal
            
        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i}_std'] = np.std(mfcc[i])
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
        
        return features
    
    def extract_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features from audio signal.
        
        Args:
            audio: Preprocessed audio signal
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sample_rate)
        features['onset_count'] = len(onset_frames)
        features['onset_rate'] = len(onset_frames) / (len(audio) / self.sample_rate)
        
        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        features['tempo'] = tempo
        
        # Statistical features
        features['audio_mean'] = np.mean(audio)
        features['audio_std'] = np.std(audio)
        features['audio_skewness'] = stats.skew(audio)
        features['audio_kurtosis'] = stats.kurtosis(audio)
        features['audio_max'] = np.max(audio)
        features['audio_min'] = np.min(audio)
        
        return features
    
    def extract_frequency_domain_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features from audio signal.
        
        Args:
            audio: Preprocessed audio signal
            
        Returns:
            Dictionary of frequency domain features
        """
        features = {}
        
        # Compute FFT
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        
        # Only consider positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(positive_magnitude)
        features['dominant_frequency'] = positive_freqs[dominant_freq_idx]
        features['dominant_magnitude'] = positive_magnitude[dominant_freq_idx]
        
        # Frequency bands analysis
        bands = {
            'low': (0, 500),
            'mid': (500, 2000),
            'high': (2000, 8000),
            'very_high': (8000, self.sample_rate//2)
        }
        
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (positive_freqs >= low_freq) & (positive_freqs <= high_freq)
            band_energy = np.sum(positive_magnitude[band_mask]**2)
            features[f'{band_name}_band_energy'] = band_energy
        
        # Spectral features
        features['spectral_energy'] = np.sum(positive_magnitude**2)
        features['spectral_entropy'] = -np.sum((positive_magnitude**2) * np.log(positive_magnitude**2 + 1e-10))
        
        return features

class MultimodalFeatureExtractor:
    """Combines image and audio features for multimodal analysis."""
    
    def __init__(self):
        self.image_extractor = ImageFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
    
    def extract_combined_features(self, image: np.ndarray, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract combined features from both image and audio data.
        
        Args:
            image: Preprocessed fish image
            audio: Preprocessed audio signal
            
        Returns:
            Dictionary of combined features
        """
        features = {}
        
        # Extract image features
        morphological_features = self.image_extractor.extract_morphological_features(image)
        color_features = self.image_extractor.extract_color_features(image)
        texture_features = self.image_extractor.extract_texture_features(image)
        abnormality_features = self.image_extractor.detect_lesions_and_abnormalities(image)
        
        # Extract audio features
        spectral_features = self.audio_extractor.extract_spectral_features(audio)
        temporal_features = self.audio_extractor.extract_temporal_features(audio)
        frequency_features = self.audio_extractor.extract_frequency_domain_features(audio)
        
        # Combine all features
        features.update(morphological_features)
        features.update(color_features)
        features.update(texture_features)
        features.update(abnormality_features)
        features.update(spectral_features)
        features.update(temporal_features)
        features.update(frequency_features)
        
        # Cross-modal features (correlations between visual and audio features)
        features['visual_audio_correlation'] = np.corrcoef(
            list(morphological_features.values())[:5],
            list(spectral_features.values())[:5]
        )[0, 1] if len(morphological_features) >= 5 and len(spectral_features) >= 5 else 0
        
        return features

