# Advanced-Task-3

## 🧠 Task 3: Multimodal Machine Learning – Housing Price Prediction Using Images and Tabular Data

### 🎯 Objective

Develop a machine learning model that predicts housing prices by leveraging **both structured/tabular data and property images**. This task demonstrates the use of **multimodal learning** by integrating computer vision and traditional regression modeling.

---

## 📊 Dataset

- **Structured Data**: Housing sales dataset containing features such as location, size, number of rooms, etc.
- **Image Data**: A set of house images (can be your own dataset or sourced from any public domain).

---

## 🛠️ Implementation Steps

### 1. Image Feature Extraction
- Utilize **Convolutional Neural Networks (CNNs)** (e.g., pre-trained models like VGG, ResNet) to extract meaningful visual features from property images.

### 2. Data Fusion
- Combine the CNN-based image features with structured data features using appropriate concatenation or fusion techniques.

### 3. Model Training
- Train a **regression model** on the combined dataset (image + tabular) to predict house prices.

### 4. Model Evaluation
- Measure performance using:
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Squared Error)

---

## 🧰 Tools & Libraries

- Python  
- TensorFlow or PyTorch  
- OpenCV / PIL for image handling  
- pandas, numpy for tabular data  
- scikit-learn for evaluation and traditional ML  
- Matplotlib / Seaborn for visualizations  

---

## 💡 Skills You'll Gain

- Multimodal machine learning (combining image and structured data)
- Deep learning with **Convolutional Neural Networks (CNNs)**
- Feature fusion techniques
- Regression modeling and performance evaluation

---

## 📌 Notes

- Ensure proper preprocessing of both image and tabular data before fusion.
- You can experiment with different CNN architectures and fusion strategies to improve performance.
- Optional: Implement early and late fusion techniques for experimentation.

---

## 📁 Suggested Folder Structure

