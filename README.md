# Mushroom Toxicity Predictor

**Author:** SaranBalaji Rajesh

## Overview
This is a computer vision project I built to classify mushrooms as either edible or poisonous. Instead of just training a model in a notebook, my goal was to build an end-to-end pipeline—from data preprocessing to deploying a basic web interface where users can upload their own images.

## Tech Stack
*   **Model Architecture:** ConvNeXt-Small
*   **Web UI:** Streamlit
*   **Techniques Used:** Data Augmentation (Rotation, Color Jitter), AdamW optimizer, Cosine Annealing

## My Learning Journey
I started this project to get practical, hands-on experience with deep learning outside of standard university coursework. 

Working with a dataset of just over 3,400 images, I quickly ran into the challenge of making the model generalize well. Implementing a robust data augmentation pipeline (specifically using rotation and color jitter) was a great lesson in preparing data for real-world inputs. By tuning the AdamW optimizer with Cosine Annealing, I was able to push the model's validation accuracy to 90.11%.

One of the biggest learning curves was stepping out of the training environment and getting the Streamlit web application running to handle real-time classifications. 

## Known Limitations & Next Steps
As a student project, this is a work in progress. A few things I want to improve:
*   **Input Validation:** Right now, the Streamlit app assumes every uploaded image is a mushroom. If you upload a picture of a car, it will still try to classify it. Adding a preliminary check or error handling is a future goal.
*   **Dataset Limitations:** [Insert a quick note here about something the model struggles with, e.g., "The model sometimes struggles with highly blurry images or rare mushroom variants not heavily represented in the 3,400 images."]

## How to Run Locally
1. Clone the repository: `git clone [your-repo-link]`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the UI: `streamlit run app.py`