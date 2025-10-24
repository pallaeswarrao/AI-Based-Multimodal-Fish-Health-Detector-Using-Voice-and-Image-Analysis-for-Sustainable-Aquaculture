"""
Data Preprocessing Module for Fish Health Detector
This module handles preprocessing of both image and audio data.
"""

import cv2
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List, Optional
import os
from pathlib import Path

class ImagePreprocessor:
    """Handles preprocessing of fish images for health analysis."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess a single fish image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Apply histogram equalization for better contrast
        image = self._enhance_contrast(image)
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast enhancement to the image."""
        # Convert to LAB color space
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced.astype(np.float32) / 255.0
    
    def extract_fish_region(self, image: np.ndarray) -> np.ndarray:
        """
        Extract fish region from the image using background subtraction.
        
        Args:
            image: Input image
            
        Returns:
            Masked image with fish region highlighted
        """
        # Convert to grayscale
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to create binary mask
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        masked_image = image.copy()
        mask_3d = np.stack([mask/255.0] * 3, axis=-1)
        masked_image = masked_image * mask_3d
        
        return masked_image

class AudioPreprocessor:
    """Handles preprocessing of audio data for fish health analysis."""
    
    def __init__(self, sample_rate: int = 22050, duration: float = 5.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Preprocess audio file for analysis.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Preprocessed audio signal
        """
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Normalize audio length
        audio = self._normalize_length(audio)
        
        # Apply noise reduction
        audio = self._reduce_noise(audio)
        
        # Normalize amplitude
        audio = librosa.util.normalize(audio)
        
        return audio
    
    def _normalize_length(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target length."""
        if len(audio) > self.target_length:
            # Truncate if too long
            audio = audio[:self.target_length]
        elif len(audio) < self.target_length:
            # Pad if too short
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        return audio
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction using spectral gating."""
        # Compute short-time Fourier transform
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Estimate noise floor (bottom 10% of magnitudes)
        noise_floor = np.percentile(magnitude, 10)
        
        # Apply spectral gating
        mask = magnitude > (noise_floor * 2)
        stft_denoised = stft * mask
        
        # Convert back to time domain
        audio_denoised = librosa.istft(stft_denoised)
        
        return audio_denoised
    
    def extract_features(self, audio: np.ndarray) -> dict:
        """
        Extract audio features for machine learning.
        
        Args:
            audio: Preprocessed audio signal
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Spectral features
        features['mfcc'] = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)
        
        # Temporal features
        features['rms'] = librosa.feature.rms(y=audio)
        
        # Harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(audio)
        features['harmonic_mean'] = np.mean(harmonic)
        features['percussive_mean'] = np.mean(percussive)
        
        return features

class DataAugmentor:
    """Handles data augmentation for both image and audio data."""
    
    @staticmethod
    def augment_image(image: np.ndarray) -> List[np.ndarray]:
        """
        Apply data augmentation to images.
        
        Args:
            image: Input image
            
        Returns:
            List of augmented images
        """
        augmented_images = [image]  # Original image
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)
        
        # Rotation
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        for angle in [-15, 15]:
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
            augmented_images.append(rotated)
        
        # Brightness adjustment
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
        augmented_images.extend([bright, dark])
        
        return augmented_images
    
    @staticmethod
    def augment_audio(audio: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """
        Apply data augmentation to audio.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate of the audio
            
        Returns:
            List of augmented audio signals
        """
        augmented_audio = [audio]  # Original audio
        
        # Time stretching
        stretched_fast = librosa.effects.time_stretch(audio, rate=1.1)
        stretched_slow = librosa.effects.time_stretch(audio, rate=0.9)
        augmented_audio.extend([stretched_fast, stretched_slow])
        
        # Pitch shifting
        pitched_up = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=2)
        pitched_down = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=-2)
        augmented_audio.extend([pitched_up, pitched_down])
        
        # Add noise
        noise = np.random.normal(0, 0.005, audio.shape)
        noisy_audio = audio + noise
        augmented_audio.append(noisy_audio)
        
        return augmented_audio

