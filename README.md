🔬 Research Contribution: Multi-Object Tracking for Precision Poultry Farming
Role: Research Assistant
Institution: University of Georgia
Project: Enhancing Multi-Object Tracking of Broiler Chickens Using Deep Learning, ML & Computer Vision

As part of an interdisciplinary research team, I developed a robust AI pipeline for real-time, identity-preserving tracking of broiler chickens in dense commercial environments. The work spanned object detection, re-identification, behavior modeling, and tracking performance optimization.

🚀 Key Technical Contributions
1. Object Detection & Optimization
Trained and evaluated 10 YOLO variants.

Selected YOLOv11x for deployment based on high performance:

Precision: 0.968

Recall: 0.960

mAP@50: 0.986

mAP@50–95: 0.805

Applied L1 unstructured pruning to reduce latency and model size:

Inference Speed: Improved from 46.5 FPS → 60 FPS at a 0.09 pruning ratio

2. Deep Feature Extraction & Re-Identification
Designed a hybrid deep feature extractor using:

Vision Transformer (ViT)

ResNet152

DenseNet201

Achieved highly discriminative embedding performance:

Cosine Similarity: 0.956 ± 0.032

Euclidean Distance: 0.020 ± 0.007

3. Kinematics-Aware ML Classifier for Identity Matching
Engineered a kinematics-aware classifier based on velocity, acceleration, and displacement features.

Benchmarked 15 machine learning models:

Logistic Regression

Random Forest

Extra Trees Classifier

Gradient Boosting Classifier

XGBoost

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Linear Discriminant Analysis (LDA)

Quadratic Discriminant Analysis (QDA)

Decision Tree

Naive Bayes

AdaBoost

LightGBM

CatBoost

Multilayer Perceptron (MLP)

Top Performer: Extra Trees Classifier

Accuracy: 0.917

Precision: 0.958

Recall: 0.920

F1 Score: 0.939

4. Multi-Object Tracking System
Evaluated and tuned six tracking algorithms:

DeepSORT, StrongSORT, SMILEtrack, OC-SORT, ByteTrack, and Modified ByteTrack

Final pipeline performance:

MOTA: 0.904 ± 0.073

MOTP: 0.953 ± 0.057

Tracking Speed: 30.1 ± 3.3 FPS

Longest Continuous Tracking Duration: 17.3 mins

📊 Impact & Deployment
Successfully tracked 5,700+ broilers across challenging real-world conditions:

Varied lighting and occlusion

High-density environments

Region-specific zones (feeder, drinker, open)

Enabled long-term identity preservation for behavior monitoring

Strong potential for:

Precision livestock farming

Automated welfare monitoring

Behavioral diagnostics at scale

This research bridged the gap between computer vision, deep learning, and precision agriculture, resulting in a high-accuracy, scalable tracking system tailored for modern poultry farms.



