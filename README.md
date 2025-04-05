## 🔬 Research Contribution: Multi-Object Tracking for Precision Poultry Farming

**Role:** Research Assistant  
**Institution:** University of Georgia  
**Project Title:** Enhancing Multi-Object Tracking of Broiler Chickens using Deep Learning, Machine Learning, and Computer Vision  

---

### 🧠 Overview

Contributed to the development of a robust, real-time, identity-preserving AI tracking system for broiler chickens in commercial poultry farms. The goal was to improve behavior analysis, tracking reliability, and animal welfare using modern deep learning and ML pipelines.

---

### 🚀 Technical Highlights

#### 1. Object Detection & Optimization

- Trained and benchmarked **10 YOLO variants**
- **Best model:** `YOLOv11x`
  - **Precision:** 0.968  
  - **Recall:** 0.960  
  - **mAP@50:** 0.986  
  - **mAP@50–95:** 0.805  
- Applied **L1 unstructured pruning** for latency reduction
  - **Inference Speed:** Improved from 46.5 FPS → 60 FPS  
  - **Pruning Ratio:** 0.09  

---

#### 2. Deep Feature Extraction & Re-Identification

Designed a **hybrid deep feature extractor** using:

- Vision Transformer (ViT)  
- ResNet152  
- DenseNet201  

**Embedding Evaluation Metrics:**

- **Cosine Similarity:** 0.956 ± 0.032  
- **Euclidean Distance:** 0.020 ± 0.007  

---

#### 3. Kinematics-Aware Identity Classification

Developed classifiers using features like **velocity**, **acceleration**, and **displacement**. Benchmarked **15 ML models**, including:

- Logistic Regression, Random Forest, Extra Trees Classifier (Best)
- Gradient Boosting, XGBoost, LightGBM, CatBoost, AdaBoost
- K-Nearest Neighbors (KNN), Support Vector Machine (SVM)
- Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA)
- Decision Tree, Naive Bayes, Multilayer Perceptron (MLP)

**Top Performer:** `Extra Trees Classifier`

- **Accuracy:** 0.917  
- **Precision:** 0.958  
- **Recall:** 0.920  
- **F1 Score:** 0.939  

---

#### 4. Multi-Object Tracking System

Evaluated and optimized **6 tracking algorithms**:

- DeepSORT, StrongSORT, SMILEtrack, OC-SORT, ByteTrack, Modified ByteTrack  

**Final Pipeline Metrics:**

- **MOTA:** 0.904 ± 0.073  
- **MOTP:** 0.953 ± 0.057  
- **Tracking Speed:** 30.1 ± 3.3 FPS  
- **Continuous Duration:** Up to 17.3 minutes  

---

### 📈 Impact & Deployment

Tracked over **5,700 broiler chickens** under diverse real-world conditions including:

- Lighting variability  
- Occlusions  
- Region-specific zones (feeder, drinker, open floor)

Enabled:

- Long-term identity preservation  
- Automated behavior monitoring  
- Precision livestock farming integrations  

This project bridged **Computer Vision**, **ML**, and **Precision Agriculture**, delivering a high-accuracy, scalable pipeline to advance smart farming and animal welfare monitoring systems.

