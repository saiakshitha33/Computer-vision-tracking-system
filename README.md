🔬 Research Assistant | University of Georgia
Project: Enhancing Multi-Object Tracking of Broiler Chickens Using Deep Learning & ML, Computer Vision

As part of an interdisciplinary research team, I developed a robust AI pipeline for real-time, identity-preserving tracking of broiler chickens in dense commercial environments. The work spanned object detection, re-identification, behavior modeling, and tracking performance optimization.

🚀 Key Technical Contributions:
🔍 1. Object Detection & Optimization:

Trained and tested 10 YOLO variants; selected YOLOv11x for its superior metrics:

Precision: 0.968, Recall: 0.960, mAP@50: 0.986, mAP@50–95: 0.805

Pruned the YOLOv11x model with L1 unstructured pruning to reduce latency:

Achieved 60 FPS vs. original 46.5 FPS at a pruning ratio of 0.09.

📦 2. Feature Extraction & Re-Identification:

Designed a hybrid feature extractor using ViT, ResNet152, and DenseNet201.

Obtained highly discriminative embeddings with:

Cosine Similarity: 0.956±0.032, Euclidean Distance: 0.020±0.007

🧠 3. ML-Based Kinematics-Aware Classifier:

Built and benchmarked 15 machine learning models for re-ID classification from velocity, acceleration, and displacement features.

Evaluated Models:

Logistic Regression

Random Forest

Extra Trees Classifier

Gradient Boosting Classifier

XGBoost

K-Nearest Neighbors

Support Vector Machine (SVM)

Linear Discriminant Analysis (LDA)

Quadratic Discriminant Analysis (QDA)

Decision Tree Classifier

Naive Bayes

AdaBoost

LightGBM

CatBoost

Multilayer Perceptron (MLP)

Best performer: Extra Trees Classifier
Accuracy: 0.917, Precision: 0.958, Recall: 0.920, F1 Score: 0.939

🎯 4. Multi-Object Tracking System:

Compared and tuned 6 tracking algorithms: DeepSORT, StrongSORT, SMILEtrack, OC-SORT, ByteTrack, and Modified ByteTrack.

Final system delivered:

MOTA: 0.904±0.073, MOTP: 0.953±0.057

Tracking Speed: 30.1±3.3 FPS, Duration: up to 17.3 mins continuously

📊 Impact:
Accurately tracked over 5,700 broilers across multiple camera setups, lighting conditions, and occlusion-heavy environments.

Enabled long-term identity preservation for behavior analysis and monitoring in precision livestock farming.

This research combined computer vision, ML, and deep learning into a scalable pipeline for the poultry industry—contributing to smarter animal welfare, precision agriculture, and real-time behavioral diagnostics.

