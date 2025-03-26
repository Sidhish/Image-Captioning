# Image Caption Generator - Flickr Dataset

## Overview
This project implements an **Image Caption Generator** using a **CNN-LSTM model** trained on the Flickr dataset. It utilizes **VGG16** for feature extraction and an **LSTM-based decoder** to generate textual descriptions of images.

## Features
- Extracts image features using **VGG16**.
- Processes captions using **Natural Language Processing (NLP)** techniques.
- Trains a **CNN-LSTM** model for generating image captions.
- Evaluates the model using **BLEU scores**.
- Provides visualization of actual vs. predicted captions.

## Dataset
- **Images**: Extracted from Flickr dataset.
- **Captions**: Available in `captions.txt`, mapped to respective image IDs.

## Installation
1. Mount Google Drive in **Google Colab**.
2. Extract the dataset and load images.
3. Install necessary dependencies:
   ```sh
   pip install tensorflow numpy pandas matplotlib nltk
   ```

## Model Architecture
- **Encoder**: Uses **VGG16** to extract image features.
- **Decoder**: LSTM-based model with **embedding layers** to process text.
- **Final Model**: Merges extracted image features with text sequences for caption generation.

## Training
- The dataset is split into **90% training** and **10% testing**.
- Uses a **batch generator** to avoid memory overflow.
- Trained using **categorical cross-entropy loss** and **Adam optimizer**.
- **Epochs**: 20
- **Batch Size**: 32

## Evaluation
- Uses **BLEU scores** to evaluate caption quality:
   ```python
   print("BLEU-1:", corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
   print("BLEU-2:", corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
   ```

## Usage
### Predict Caption for an Image
1. Load the trained model.
2. Extract image features.
3. Generate captions:
   ```python
   generate_caption("image_name.jpg")
   ```

### Test on a Real Image
1. Load an external image.
2. Extract features using VGG16.
3. Predict caption using the trained model.
   ```python
   predict_caption(model, feature, tokenizer, max_length)
   ```

## Results
- The model successfully generates relevant captions for images.
- BLEU scores indicate the quality of generated captions.
- Predictions align well with actual captions.

## Technologies Used
- **TensorFlow & Keras** (Deep Learning)
- **VGG16** (Feature Extraction)
- **LSTM** (Sequence Generation)
- **NLTK** (Text Processing & BLEU Scores)
- **Matplotlib & PIL** (Visualization)
- **Google Colab** (Training & Execution)

## Future Improvements
- Fine-tune the model with a larger dataset.
- Experiment with **Transformer-based** models like **GPT-4 Vision**.
- Enhance text preprocessing for better predictions.

## Author
- Developed as part of a **Deep Learning** project focusing on **Image Captioning**.
- Contact: [Your Email or GitHub Link]

