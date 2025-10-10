# ♻️ Waste Classifier

A deep learning system that automatically classifies garbage images into categories:
**cardboard, glass, metal, organic, paper, plastic**.

Built with **PyTorch** and **Streamlit**, using **transfer learning** on
[Garbage Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification).

---

## 🚀 Features

- **Data preparation**: resize, normalize, augment, and split images  
- **Model training**: EfficientNet-B0, MobileNetV2, ResNet50 via transfer learning  
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
- **Interactive UI**: Streamlit app for image upload & prediction  
- **Deployment ready**: Streamlit Cloud or local run

---

## 🧰 Installation

```bash
git clone https://github.com/Hritvik16000/waste-classifier.git
cd waste-classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_data.py --persist_resized --img_size 224 --trash_to_organic
python src/train_model.py --model efficientnet_b0 --epochs 5
python src/evaluate_model.py

streamlit run src/app_streamlit.py

cp models/best_model_efficientnet_b0.pth models/best_model.pth

waste-classifier/
├── src/
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── app_streamlit.py
│   └── utils.py
├── data/splits/
├── models/
├── reports/
└── requirements.txt

MIT License © 2025 Hritvik Dadhich
