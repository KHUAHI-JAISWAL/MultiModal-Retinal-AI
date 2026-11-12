# 🩺 MultiModal-Retinal-AI

### Multimodal Deep Learning for Retinal Findings Identification

## 📌 About Project
MultiModal-Retinal-AI is a deep learning project that combines retinal fundus images and clinical reports to identify retinal abnormalities using multimodal learning.

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- EfficientNetB0
- NumPy, Pandas
- Matplotlib
- Scikit-learn
- Google Colab

## 🧠 Model Architecture
- **Image Processing:** EfficientNetB0
- **Report Processing:** Dense Neural Network
- **Feature Fusion:** Concatenation
- **Output:** Multi-label classification

## 🔍 Retinal Findings
- Hemorrhage
- Hard Exudates
- Cotton Wool Spots
- Retinal Detachment

## 📊 Results

| Metric | Result |
|---|---:|
| DR Image-only Accuracy | 46.43% |
| DR Fusion Accuracy | 53.57% |
| Findings Binary Accuracy | 90.18% |
| Findings Macro F1-score | 0.59 |

*Results are preliminary and based on a small test dataset.*

## 📂 Files
- `retinal_fusion_model.keras` – Trained model
- `model_config.json` – Model configuration

## 🚀 Future Scope
- Improve model performance.
- Use advanced NLP techniques.
- Test on larger datasets.
- Develop an interactive web application.

## ⚠️ Disclaimer
This project is for educational and research purposes only. It is not intended for clinical diagnosis.

## 👩‍💻 Author
**Khushi Jaiswal**

Model development and multimodal deep learning.

[GitHub Profile](https://github.com/KHUAHI-JAISWAL)

**Reena Soni**

Streamlit UI development and application integration.

[GitHub Profile](https://github.com/reenasoni1757)
