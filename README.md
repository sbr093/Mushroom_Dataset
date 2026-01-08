# Mushroom Classifier

Mushroom Classifier is a web-based application that classifies mushroom images as edible or poisonous using a trained image-classification model. The app is designed for simplicity and safety: upload a photo, click "Classify", and receive a prediction with a confidence score and a clear safety warning.

> ⚠️ Important: This project is for educational and research purposes only. Do NOT consume mushrooms based solely on this tool's output. Always consult a qualified mycologist or local expert before making decisions about wild mushrooms.

Short project blurb (suitable for GitHub repo description)
Mushroom Classifier — A Streamlit-based image classification app that analyzes photos of mushrooms and predicts whether a specimen is edible or poisonous. Provides clear confidence scores and safety-first recommendations for demonstrative and research use.

Demo / Screenshots
Place the UI screenshots in the repository at `assets/screenshots/` with the filenames below. The README references these files so they display on GitHub.

- assets/screenshots/upload_view.png — Upload UI (drag & drop)
- assets/screenshots/landing_view.png — App header & file selector
- assets/screenshots/prediction_view.png — Prediction result, confidence and warning

Insert the images into this README as follows:

![Upload view — choose an image to classify](assets/screenshots/upload_view.png)
*Upload view — drag-and-drop or browse files.*

![Landing view — app header and uploader](assets/screenshots/landing_view.png)
*Clean, dark-themed interface and upload control.*

![Prediction view — label, confidence, and safety warning](assets/screenshots/prediction_view.png)
*Prediction output with confidence percentage and a prominent safety message.*

Key features
- Simple image upload: drag-and-drop or file-browser support for common image formats (JPG, JPEG, PNG).
- Real-time inference with a single click to trigger classification.
- Clear results: predicted label (Edible / Poisonous) with a confidence score.
- Safety-first messaging: explicit warning when a specimen is predicted poisonous.
- Lightweight, mobile-friendly UI with accessible controls.

Technical overview
- Frontend / UI: Streamlit (inferred from UI components and Streamlit-specific warnings).
- Model: A serialized image-classification model (e.g., TensorFlow/Keras `.h5`, PyTorch `.pt`, or similar). Update the path below to your actual model filename.
- Inference flow: uploaded image -> preprocessing (resize / normalize) -> model inference -> confidence calculation -> UI display.

Repository layout (suggested)
- app.py (or main.py) — Streamlit app that implements the upload, inference, and result UI.
- models/
  - mushroom_classifier.h5 (or .pt) — serialized trained model
- assets/
  - screenshots/upload_view.png
  - screenshots/landing_view.png
  - screenshots/prediction_view.png
- requirements.txt — Python dependencies
- README.md — this file

Quick start (example)
1. Clone the repo
```bash
git clone https://github.com/sbr093/Mushroom_Dataset.git
cd Mushroom_Dataset
```

2. Create & activate a virtual environment
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Add screenshots
Create the folder and copy screenshots (from your local machine) into:
```
assets/screenshots/upload_view.png
assets/screenshots/landing_view.png
assets/screenshots/prediction_view.png
```

5. Run the app
```bash
streamlit run app.py
```
Open the local URL shown in the terminal (usually http://localhost:8501).

Model and training (update with exact info)
- Model file: `models/mushroom_classifier.h5` (replace with your actual filename and format).
- Preprocessing: resize to model input size (e.g., 224×224), normalize pixel values, ensure proper channel order.
- Training data: add dataset description (source, number of images, class balance, augmentation strategy).
- Training script: include `train.py` or training notes if you want others to reproduce your model.

Safety, limitations, and disclaimers
- Predictions are probabilistic and may be incorrect. Confidence percentages are model outputs, not guarantees.
- Accuracy depends on image quality, occlusion, lighting, and species representation in training data.
- This tool is for demonstration and educational use only — not a substitute for expert identification.
- Consider adding a multi-class taxonomy or "unknown" threshold to avoid overconfident predictions when the image is out-of-distribution.

Deployment recommendations
- Streamlit Community Cloud — quick, direct deployment from GitHub.
- Containerize with Docker + Gunicorn + Nginx for a more production-ready setup.
- Protect model files and limit upload size / rate to prevent abuse.

Contributing
- Improvements welcome: expand dataset, retrain with more samples, add batch uploads, add localization, or improve UI accessibility.
- Please open an issue or PR with reproducible changes and tests where appropriate.

License & Attribution
- Add your chosen license (e.g., MIT) and dataset citations if required by the image sources.

Contact
- Maintainer: sbr093 (GitHub)
