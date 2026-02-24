# Project Report
# Machine Learning-Based Detection of Fraudulent SMS Messages
# Submitted to Bharathiar University in partial fulfillment of the requirements for the award of the degree of Bachelor of Information Technology

# Under the Supervision and Guidance of
# Ms. S. Elakkiya, M.Sc., (B.Ed)
# Assistant Professor, Department of Information Technology

# March 2026

# Declaration

I declare that the work which is being presented in this project report entitled "Machine Learning-Based Detection of Fraudulent SMS Messages", submitted to the Department of Information Technology, KG College of Arts and Science, Saravanampatti, Coimbatore for the award of the degree of Bachelor of Information Technology of the Bharathiar University, is an authentic record of my work carried out under the supervision and guidance of Ms. S. Elakkiya, M.Sc., (B.Ed). I have not plagiarized or submitted the same work for the award of any other degree.

March 2026
Coimbatore

(Jayaprakash P)

# Certificate

This is to certify that the project entitled "Machine Learning-Based Detection of Fraudulent SMS Messages" submitted to Bharathiar University in partial fulfillment for the award of the degree of Bachelor of Information Technology is a record of original work done by Jayaprakash P (2326J0360) during the period of study in KG College of Arts and Science under the supervision of Ms. S. Elakkiya, Assistant Professor, Department of Information Technology.

Place: Coimbatore
Date:

Signature of the Guide          Signature of the HoD

(College Seal)

Submitted for the Viva-Voce Examination held on

Internal Examiner               External Examiner

# Acknowledgement

I express my sincere thanks to Dr. Ashok Bakthavathsalam, Managing Trustee for allowing me to do this course of study and to undertake this project work.

It is my pleasure to express my sincere thanks to Dr. B. Vanitha, Secretary, KG College of Arts and Science for her valuable thoughts.

I would like to express my heartfelt gratitude to Dr. S. Vidhya, Principal, KG College of Arts and Science, for her unwavering moral support throughout the course.

I would like to express my gratefulness to Dr. P. Ajitha, Dean-Curriculum Development Cell, KG College of Arts and Science, for her untiring support throughout the course.

I take this opportunity to convey my sincere thanks to Dr. M. Usha, Dean-School of Computational Sciences, KG College of Arts and Science for her moral support throughout the course.

I have great pleasure in acknowledging my thanks to Dr. S. Vijaya, Head of the Department, Information Technology, KG College of Arts and Science for her encouragement and help throughout the course.

I express my sincere thanks to Ms. S. Elakkiya, Assistant Professor, Department of Information Technology for her constant encouragement and motivation throughout the project. I thank her for her endless support and encouragement towards the work.

- Jayaprakash P

# MACHINE LEARNING-BASED DETECTION OF FRAUDULENT SMS MESSAGES

# SYNOPSIS

The explosive growth of mobile communication has made Short Message Service a convenient channel for personal and commercial interaction, but it has also been widely exploited for fraudulent activities known as smishing. Fraudulent SMS messages attempt to manipulate users into revealing confidential information, clicking malicious links, or performing unauthorized transactions, posing serious risks to privacy and financial security. This project presents a machine learning based framework for the automatic detection of fraudulent SMS messages using a dual-model approach: a TF-IDF plus Linear SVM baseline and an optional fine-tuned Transformer (RoBERTa) for higher accuracy. The proposed system begins with comprehensive text preprocessing, including normalization, URL and phone replacement, and Unicode handling, to reduce noise and normalize message content. Relevant textual features are extracted using Term Frequency Inverse Document Frequency vectorization with 1-2 gram ranges. The sklearn pipeline uses LinearSVC with sigmoid calibration for probability outputs; when enabled, a Hugging Face Transformer is also trained and the best model is selected by validation metric. Experimental evaluation shows that the system achieves high accuracy, precision, and recall, with threshold tuning to optimize for accuracy or F1. The approach is computationally efficient, scalable, and adaptable to evolving fraud patterns, with a Streamlit web UI for single-message and batch CSV detection. Overall, the proposed solution provides a reliable and intelligent mechanism to enhance SMS security and protect users from smishing attacks.

Keywords: Smishing, Fraudulent SMS Detection, Machine Learning, TF-IDF, Linear SVM, RoBERTa, Text Classification, Cybersecurity, Spam Filtering, Natural Language Processing, Feature Extraction, Supervised Learning, Streamlit, Evaluation, Accuracy, Precision, Recall, Real Time Detection, Mobile Security, UCI SMS Spam Collection

# TABLE OF CONTENTS

S.NO    TITLE                                    PAGE NO.
1       INTRODUCTION                              1
        1.1 Overview of the Project                1
        1.2 Objectives                            2
        1.3 Organization Profile                   3
        1.4 Purpose, Scope and Methodology         5
2       LITERATURE SURVEY                           8
        2.1 SMS Spam and Fraud Detection           8
        2.2 Machine Learning Approaches            10
        2.3 Deep Learning and Transformers        12
        2.4 Related Work Summary                   14
3       SYSTEM ANALYSIS                            16
        3.1 Problem Definition                     16
        3.2 Existing System                        17
        3.2.1 Drawbacks of Existing System        18
        3.3 Proposed System                       19
        3.3.1 Advantages of Proposed System      20
        3.4 Features                              21
4       ALGORITHM AND TECHNIQUE                    24
        4.1 Term Frequency Inverse Document Frequ  24
        4.2 Linear Support Vector Machine           26
        4.3 Calibrated Classifier                  28
        4.4 RoBERTa Transformer                   29
        4.5 Threshold Tuning                       31
5       SYSTEM SPECIFICATIONS                      33
        5.1 Hardware Specifications                33
        5.2 Software Specifications                34
6       DESIGN AND DEVELOPMENT                     36
        6.1 Module Description                     36
        6.2 System Architecture                    39
        6.3 File Design                            41
        6.4 Input Design                           43
        6.5 Output Design                          45
        6.6 Database Design                        47
7       IMPLEMENTATION AND TESTING                 50
        7.1 Implementation Details                 50
        7.2 Unit Testing                           52
        7.3 Integration Testing                    53
        7.4 Validation and Functional Testing      54
        7.5 Performance and Security Testing       55
8       EXPERIMENTAL RESULTS                       57
        8.1 Dataset Statistics                     57
        8.2 Model Performance                     58
        8.3 Sample Outputs                         59
9       CONCLUSION AND FUTURE WORK                 60
        9.1 Conclusion                             60
        9.2 Project Outcomes                       61
        9.3 Future Enhancement                     62
10      BIBLIOGRAPHY AND REFERENCES               64
11      APPENDICES                                 66
        A. Table Structure                         66
        B. Data Flow Diagrams                      69
        C. Complete Source Code                    72
        D. Output Screens and User Manual          85
        E. Glossary                               92
        F. requirements.txt                       94

# 1. INTRODUCTION

## 1.1 Overview of the Project

Background:
Short Message Service (SMS) has been a cornerstone of mobile communication since its introduction in the 1990s. Despite the rise of internet-based messaging apps, SMS remains widely used for authentication (one-time passwords), transactional notifications (bank alerts, delivery updates), and marketing. Its ubiquity and open nature also make it a target for malicious actors. Smishing attacks have increased in frequency and sophistication, causing significant financial losses and privacy breaches. Detecting such messages before they reach users is a critical cybersecurity challenge.

The rapid expansion of mobile communication has transformed Short Message Service into a fast and widely used medium for information exchange, business communication, and service notifications. Along with its benefits, this growth has created opportunities for cybercriminals to exploit SMS platforms through fraudulent messages, commonly referred to as smishing. These messages are designed to mislead users into sharing sensitive data, clicking harmful links, or authorizing malicious transactions. Traditional rule based filtering techniques are no longer sufficient because fraud patterns continuously evolve and attackers adapt their language to bypass static detection mechanisms.

This project focuses on developing an intelligent and automated system for detecting fraudulent SMS messages using machine learning techniques. The core objective is to accurately classify incoming messages as legitimate or fraudulent by analyzing their textual content. The system uses natural language processing methods to preprocess raw SMS data by normalizing text, replacing URLs and phone numbers with placeholders, and converting unstructured text into meaningful numerical representations. Feature extraction is performed using Term Frequency Inverse Document Frequency with 1-2 gram ranges. Two model options are used: a strong baseline of TF-IDF plus Linear SVM with sigmoid calibration for probabilities, and an optional fine-tuned Transformer (RoBERTa) for higher accuracy when resources allow. The best model is selected automatically based on validation accuracy or F1. The trained system evaluates new SMS messages via a Streamlit web UI or command-line interface and predicts their legitimacy with high accuracy. The project emphasizes scalability, efficiency, and adaptability, making the solution suitable for deployment in mobile and telecom contexts.

## 1.2 Objectives

* To design an intelligent system that can automatically identify fraudulent SMS messages by analyzing their textual content, reducing the need for manual monitoring and minimizing user exposure to smishing attacks in high volume communication environments.

* To apply effective natural language processing techniques such as text normalization, URL and phone replacement, and Unicode handling to preprocess raw SMS data, ensuring noise reduction and improved feature representation for accurate machine learning classification.

* To implement TF-IDF based feature extraction with configurable n-gram ranges in order to transform unstructured SMS text into meaningful numerical vectors that capture word importance and contextual relevance across legitimate and fraudulent message datasets.

* To develop and train a dual-model pipeline: a TF-IDF plus Linear SVM baseline with probability calibration, and an optional Hugging Face Transformer (RoBERTa) classifier, selecting the best model by validation metric to achieve high accuracy, precision, and recall while handling class imbalance efficiently.

* To create a scalable and efficient detection framework with a Streamlit web UI for single-message and batch CSV detection, plus CLI support, suitable for real time deployment and integration with existing security infrastructures.

## 1.3 Organization Profile

ABOUT SD PRO

SD Pro Solutions Pvt Ltd is a leading Engineering and Educational Project provider for Diploma, Engineering (Under Graduate, Post graduates) and Research Scholars. SD Pro was established in the year 2012 for Project Development, Course Designing, Training, and placement guidance, based at South India. The organization has built a reputation for delivering high-quality technical education and hands-on project experience. Students and professionals who complete SD Pro programs are well-prepared for industry roles in embedded systems, software development, and research. The emphasis on depth over breadth ensures that participants gain substantive expertise rather than superficial familiarity with tools and technologies. SD Pro collaborates with educational institutions to bridge the gap between academic curriculum and industry requirements. SD Pro provides Training and Projects in Embedded systems (Raspberry Pi Pico or Arduino), VLSI, Matlab, Power systems, Power Electronics, DSP/DIP, VLSI, Python, .Net, Java/J2EE/Android, Mechanical Design and Fabrication, Civil as well as develops its own range of quality Embedded products. SD Pro has successfully powered itself in training thousands of students and professionals. The teaching philosophy deployed creates in-depth knowledge about the subject at hand. We believe that depth is an essential ingredient to achieve heights in training and development. Students from SD Pro Solutions have proved the point by their work in the fast paced industry world.

SERVICES OFFERED

We provide a platform where the students get to learn essential as well as advanced things about various technologies like embedded system design, VLSI, Robotics, Digital Image Processing, Digital Signal Processing, Power Electronics and Power Systems and various other Design platforms used for electronics system design. We also provide an R and D facility where students can experiment and execute their ideas and we get them commercialize for them. We give them the opportunity to learn through workshops, courses, on-site training and Seminars.

VISION

To be a leading technical training institute benefitting thousands of students, providing them quality knowledge through an education system which is both approachable and advanced.

MISSION

To create a technically strong and technologically advanced student base leading to a superfluity of Indian innovations.

## 1.4 Purpose, Scope and Methodology

Purpose of the Project

The primary purpose of this project is to develop an intelligent and automated system capable of detecting fraudulent SMS messages using machine learning techniques. The project aims to protect mobile users from smishing attacks by accurately identifying malicious messages before they cause financial loss or privacy breaches. By leveraging textual analysis and predictive modeling, the system seeks to enhance trust, safety, and reliability in mobile communication systems while reducing dependency on manual monitoring and traditional rule based filters.

Scope of the Project

1. User Scope: The system is designed for mobile phone users, telecom service providers, and organizations that rely on SMS communication. It benefits end users by preventing exposure to fraudulent messages and supports administrators by offering an automated detection mechanism that operates without direct user intervention.

2. Functional Scope: Functionally, the project focuses on collecting SMS data (UCI SMS Spam Collection), preprocessing text, extracting features with TF-IDF, training a sklearn baseline and optionally a Transformer model, and classifying messages as legitimate or fraudulent. The system supports single-message and batch CSV evaluation through a Streamlit UI and command-line interface.

3. Technical Scope: Technically, the project uses Python, natural language preprocessing, TF-IDF vectorization, Linear SVM with calibration, and optionally Hugging Face Transformers (RoBERTa). The front-end is built with Streamlit. It emphasizes accuracy, scalability, and compatibility with existing infrastructures.

4. Geographic Scope: The solution is geographically independent and can be deployed across different regions, languages, and telecom networks, making it suitable for global SMS fraud detection applications.

METHODOLOGY

1. Data Collection: The first step involves collecting the UCI SMS Spam Collection dataset. The dataset is downloaded automatically from the UCI Machine Learning Repository and contains labeled ham and spam messages. The data is loaded into a DataFrame with columns for raw text, normalized text, and binary label (0 legitimate, 1 fraudulent). Stratified splitting ensures balanced train, validation, and test sets.

2. Data Preprocessing: Raw SMS messages are cleaned and standardized. The preprocessing module applies Unicode normalization (NFKC), replaces URLs with a placeholder, replaces email and phone patterns with placeholders, normalizes numbers, lowercases text, and collapses whitespace. This step reduces noise and ensures consistency across the dataset for both the classical and Transformer pipelines.

3. Feature Extraction: For the sklearn pipeline, textual data is transformed into numerical form using TF-IDF vectorization with 1-2 gram ranges, min_df and max_df settings, and sublinear_tf. For the Transformer option, the Hugging Face tokenizer is used with a configurable max length. Feature extraction provides the model with informative and discriminative input features.

4. Model Training and Evaluation: Two paths are supported: (a) TF-IDF plus LinearSVC with CalibratedClassifierCV for probability outputs, and (b) optional fine-tuning of a Transformer (e.g., RoBERTa) for sequence classification. The best model is selected by validation metric (accuracy or F1). Threshold search over the validation set optimizes the decision threshold. The selected model is evaluated on the test set and metrics (accuracy, precision, recall, F1, ROC-AUC) are recorded in a training report.

5. Fraud Detection and Deployment: The trained model is saved under models/best with meta.json describing model type and threshold. The Streamlit app and predict.py CLI load the best model and classify new messages. Each message is preprocessed and scored; results are shown in the UI or written to CSV for batch runs. The framework supports periodic retraining to adapt to new fraud patterns.

# 2. LITERATURE SURVEY

## 2.1 SMS Spam and Fraud Detection

The problem of SMS spam and fraudulent message detection has been studied extensively in academia and industry over the past two decades. As mobile communication grew from a novelty to a primary channel for personal and commercial exchange, so did the volume and sophistication of malicious messages. Early research focused on filtering unwanted commercial messages (spam) using simple heuristics and keyword lists. Over time, the threat evolved from unsolicited promotions to phishing attempts that impersonate banks, government agencies, and service providers, attempting to steal credentials or deploy malware. This combined threat is often termed smishing, a portmanteau of SMS and phishing.

Researchers at the UCI Machine Learning Repository compiled the SMS Spam Collection dataset, which has become a standard benchmark for SMS classification research. The dataset contains approximately 5,500 messages labeled as ham (legitimate) or spam (fraudulent). It reflects real-world text characteristics including abbreviations, typos, mixed language, and various message lengths. The class distribution is imbalanced, with spam messages comprising roughly 13 percent of the dataset, which poses challenges for classifiers that are not designed to handle imbalance.

Studies have shown that fraudulent SMS messages often exhibit distinct lexical and stylistic patterns: frequent use of urgency-inducing words, promotional language, requests for personal information, and embedded URLs or phone numbers. However, these patterns are not static. Attackers continuously adapt their wording to evade rule-based systems, making adaptive machine learning approaches more suitable for long-term detection.

## 2.2 Machine Learning Approaches

Classical machine learning methods for text classification typically involve two stages: feature extraction and classification. Bag-of-words and n-gram representations have been widely used, with Term Frequency Inverse Document Frequency (TF-IDF) being one of the most effective and interpretable feature extraction schemes. TF-IDF assigns weights to terms based on their frequency within a document and their rarity across the corpus, highlighting terms that are discriminative for classification.

Support Vector Machines (SVMs), particularly Linear SVMs, have shown strong performance on high-dimensional sparse text data. They work by finding a hyperplane that maximizes the margin between classes in the feature space. LinearSVC is efficient for large vocabularies and scales well with the number of training samples. A limitation of standard SVMs is that they output class labels or uncalibrated decision scores rather than probabilities. To address this, researchers use probability calibration techniques such as Platt scaling (sigmoid) or isotonic regression, often implemented via cross-validation wrappers like CalibratedClassifierCV in scikit-learn.

Naive Bayes classifiers, especially Multinomial Naive Bayes, have also been applied to SMS spam detection. They are fast and require relatively little tuning but often underperform compared to SVMs on textual data. Random Forest and Gradient Boosting methods (including XGBoost) have been used as well, offering good accuracy at the cost of increased computational complexity and reduced interpretability compared to linear models. Ensemble methods that combine multiple classifiers can improve robustness but add deployment complexity. The choice of algorithm depends on the trade-off between accuracy, speed, interpretability, and resource constraints.

## 2.3 Deep Learning and Transformers

Deep learning approaches have achieved state-of-the-art results on many natural language processing tasks. Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks can capture sequential dependencies in text, which may help in detecting phrase-level patterns characteristic of fraud. However, these models require more data and compute than classical methods and can be prone to overfitting on small datasets like the UCI SMS Spam Collection.

Transformer-based models, such as BERT and RoBERTa, have revolutionized NLP by pre-training on massive corpora and then fine-tuning on downstream tasks. RoBERTa (Robustly Optimized BERT Pretraining Approach) improves upon BERT through better training procedures and longer training. When fine-tuned for sequence classification, these models capture sophisticated semantic and contextual patterns that TF-IDF and linear classifiers cannot represent. They typically achieve higher accuracy at the cost of significantly higher computational requirements (GPU recommended for training, larger memory footprint). For SMS fraud detection, fine-tuned RoBERTa-base has been reported to reach very high accuracy, often exceeding 98 percent on benchmark datasets, though performance depends heavily on dataset size, quality, and evaluation methodology.

A practical strategy is to maintain both a fast classical baseline (TF-IDF plus Linear SVM) and an optional Transformer model. The baseline ensures low-latency, CPU-friendly inference when resources are limited, while the Transformer can be used when maximum accuracy is required and GPU or cloud resources are available. Model selection can be automated by comparing validation metrics and choosing the best performer for deployment.

## 2.4 Related Work Summary

Several recent publications have addressed SMS spam and fraud detection. Nagare et al. proposed a Support Vector Machine-based approach for mobile devices, emphasizing efficiency. Singh Sikarwar et al. compared deep learning models for SMS spam detection, finding that transformer-based models outperform classical methods when sufficient data is available. Gupta investigated natural language processing for spam and fraudulent call detection. De Luna et al. applied a machine learning approach for efficient spam detection in SMS. Christiawan et al. studied fraud classification in Indonesian SMS using both machine learning and deep learning. Other works have explored AI-powered multi-domain misinformation detection, smart detection of malicious SMS on Android, and fine-tuned language models for fraudulent call detection. Schwarz et al. presented smishing detection from a messaging platform perspective. These studies collectively support the use of machine learning and NLP for adaptive, accurate fraud detection and provide context for the dual-model design adopted in this project.

# 3. SYSTEM ANALYSIS

## 3.1 Problem Definition

The rapid growth of mobile communication has significantly increased the use of Short Message Service as a primary channel for personal, commercial, and transactional communication. However, this expansion has also led to a sharp rise in fraudulent SMS messages, commonly known as smishing. These messages are carefully crafted to deceive users into revealing sensitive information such as passwords, bank details, or verification codes, or to trick them into clicking malicious links that may lead to financial loss or identity theft. The problem is intensified by the fact that fraudulent messages often imitate legitimate organizations, making them difficult for users to distinguish from genuine communications.

Traditional SMS filtering mechanisms rely mainly on static rules, blacklists, or keyword based detection techniques. While these methods were effective in the early stages of spam control, they fail to address the dynamic and evolving nature of modern smishing attacks. Fraudsters frequently change message formats, language patterns, and delivery strategies to bypass predefined rules, resulting in high false positive and false negative rates. Manual monitoring and reporting by users is also unreliable and impractical due to the massive volume of messages exchanged daily.

The lack of intelligent, adaptive, and automated detection mechanisms creates a critical security gap in mobile communication systems. Therefore, there is a clear need for an efficient solution that can analyze SMS content intelligently, learn from historical data, and accurately identify fraudulent messages in real time. Addressing this problem requires the integration of machine learning and natural language processing techniques capable of understanding textual patterns and adapting continuously to new smishing strategies.

## 3.2 Existing System

Existing systems for SMS fraud detection primarily rely on traditional filtering techniques and manual intervention to identify suspicious messages. These systems are commonly implemented at the device or network level and use predefined rules, keyword matching, blacklists, and sender reputation checks to classify messages. If an SMS contains certain trigger words, links, or known fraudulent numbers, it is flagged or blocked. While this approach is simple to implement and requires minimal computational resources, it lacks intelligence and adaptability.

Rule based systems are static in nature and depend heavily on prior knowledge of fraud patterns. As attackers continuously modify message content, spelling, structure, and delivery methods, these systems struggle to detect new and unseen smishing techniques. Keyword based filters often generate high false positives by blocking legitimate messages that contain similar terms, while sophisticated fraudulent messages that avoid common keywords may bypass detection entirely. Blacklist based mechanisms are also limited because fraudsters frequently change phone numbers and sender identities. In many cases, existing systems depend on user reporting to improve detection accuracy. This approach is unreliable, as users may fail to recognize fraud, report messages late, or ignore suspicious activity altogether. Manual analysis is time consuming and impractical given the massive volume of SMS traffic generated daily. Furthermore, traditional systems do not effectively analyze contextual or semantic relationships within message content. Overall, the existing SMS fraud detection systems provide only basic protection and are unable to cope with the evolving complexity of modern smishing attacks.

Drawbacks of the Existing System

* Static Rule Dependency: Existing systems rely heavily on predefined rules and keywords, which makes them ineffective against newly emerging smishing patterns. Fraudsters can easily bypass these static rules by modifying message wording or structure.

* High False Positive Rate: Keyword based filtering often blocks legitimate SMS messages that contain common transactional or promotional terms, leading to poor user experience and reduced trust in the system.

* Poor Adaptability to Evolving Attacks: Traditional systems lack learning capability and cannot adapt automatically to new fraud techniques. Updates require manual intervention, making detection slow and inefficient.

* Limited Contextual Understanding: Existing methods fail to analyze semantic meaning or contextual relationships within SMS content, reducing their ability to detect sophisticated and well disguised fraudulent messages.

* Scalability and Efficiency Issues: Manual monitoring and user dependent reporting are impractical for handling large volumes of SMS traffic, resulting in delayed responses and insufficient protection in real time environments.

## 3.3 Proposed System

The proposed system introduces an intelligent and automated approach for detecting fraudulent SMS messages using machine learning and natural language processing techniques. Unlike traditional rule based methods, this system is designed to learn patterns directly from data, enabling it to adapt to evolving smishing strategies. The core objective is to accurately classify SMS messages as legitimate or fraudulent by analyzing their textual content in a scalable and efficient manner.

The system begins by loading the UCI SMS Spam Collection dataset (downloaded automatically when needed). Preprocessing techniques such as text normalization, URL and phone replacement, and Unicode normalization are applied to clean and normalize the raw text. The cleaned text is then used in two ways: for the classical pipeline, it is transformed into numerical features using TF-IDF vectorization with 1-2 grams; for the optional Transformer path, the same text is tokenized with a Hugging Face tokenizer. The sklearn pipeline consists of TF-IDF plus LinearSVC with CalibratedClassifierCV for probability outputs. The optional Transformer model (e.g., RoBERTa) is fine-tuned for sequence classification. The best model is selected automatically based on validation accuracy or F1, and the decision threshold is tuned on the validation set. Once trained, the system can evaluate incoming SMS messages in real time through a Streamlit web interface (single message or batch CSV) or via the command-line script predict.py. The proposed system supports scalability and continuous improvement through periodic retraining with new data, and can be integrated into mobile or telecom infrastructures without disrupting existing services.

Advantages of the Proposed System

* Adaptive Learning Capability: The proposed system uses machine learning to automatically learn from historical and new data, allowing it to adapt to evolving smishing techniques without relying on fixed rules.

* High Detection Accuracy: By combining TF-IDF feature extraction with Linear SVM (and optionally a Transformer), the system achieves high accuracy, precision, and recall, effectively reducing false positives and false negatives.

* Real Time Fraud Detection: The system can analyze and classify incoming SMS messages instantly, enabling timely identification of fraudulent messages and preventing user exposure to potential threats.

* Scalability and Efficiency: The approach is computationally efficient and scalable, making it suitable for deployment across large telecom networks and high volume SMS environments. The sklearn-only mode runs quickly on CPU.

* Improved User Security and Trust: Automated and reliable fraud detection enhances user privacy, reduces financial risks, and strengthens trust in mobile communication services by providing consistent protection.

## 3.4 Features

* Dual-model pipeline: TF-IDF plus Linear SVM baseline and optional Hugging Face Transformer (RoBERTa) for sequence classification.

* Automatic model selection: The system trains the sklearn model and optionally the Transformer, then selects the best performer on the validation set according to the chosen metric (accuracy or F1).

* Threshold tuning: A grid search over decision thresholds on the validation set optimizes for the selected metric, with tie-breaking by accuracy and recall.

* Streamlit web UI: Tabs for single SMS detection, batch CSV upload and scoring, and an optional AI chatbot (local or OpenAI). Sidebar controls for fraud threshold and model metadata.

* Command-line interface: train.py for training and model selection; predict.py for single-message or batch CSV prediction with configurable threshold.

* Text preprocessing: Normalization of URLs, emails, phones, and numbers with placeholders; Unicode normalization and whitespace handling for consistent input.

* Stratified data splitting: Train, validation, and test sets with preserved class distribution for reliable evaluation.

* Persisted best model: The selected model is saved under models/best with meta.json (model type, threshold, and configuration) and can be loaded for inference without retraining.

* Training report: JSON report with dataset and split sizes, validation and test metrics for both models, and best model metadata for auditing and comparison.

# 4. ALGORITHM AND TECHNIQUE

## 4.1 Term Frequency Inverse Document Frequency

Term Frequency Inverse Document Frequency (TF-IDF) is a numerical statistic that reflects the importance of a term in a document relative to a collection of documents. It is widely used in information retrieval and text mining for feature extraction. The TF-IDF value increases with the number of times a term appears in a document but is offset by the frequency of the term across the corpus, so that common terms (e.g., "the", "is") receive lower weights and rare, discriminative terms receive higher weights.

Term Frequency (TF) measures how often a term appears in a document. Several variants exist: raw count, binary presence, or normalized count. A common approach is to use the sublinear TF variant, which uses 1 + log(tf) instead of raw tf to dampen the effect of very frequent terms within a document. Inverse Document Frequency (IDF) is computed as log((N + 1) / (df + 1)) + 1, where N is the total number of documents and df is the number of documents containing the term. The TF-IDF score for a term in a document is the product of TF and IDF.

In this project, the TfidfVectorizer from scikit-learn is configured with ngram_range=(1, 2), meaning unigrams (single words) and bigrams (pairs of consecutive words) are extracted. This captures both single-word signals (e.g., "free", "winner") and phrase-level signals (e.g., "free prize", "click here"). The parameters min_df=2 and max_df=0.95 filter out terms that appear in very few documents (noise) or in nearly all documents (non-discriminative). strip_accents="unicode" normalizes accented characters. The resulting sparse matrix is passed to the classifier. TF-IDF provides a simple, interpretable, and effective representation that works well for SMS length texts and scales efficiently to thousands of messages.

## 4.2 Linear Support Vector Machine

The Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression. For binary classification, the goal is to find a hyperplane in the feature space that separates the two classes with the maximum margin. The margin is the distance between the hyperplane and the nearest training samples (support vectors). Maximizing the margin improves generalization. For linearly separable data, the optimal hyperplane is uniquely defined; for non-separable data, a soft margin is used, allowing some misclassification with a penalty controlled by the parameter C.

Linear SVM (LinearSVC) assumes the decision boundary is linear in the feature space. For high-dimensional sparse data such as TF-IDF vectors, linear kernels often perform as well as or better than non-linear kernels while being computationally efficient. LinearSVC uses a linear kernel by default and solves the optimization problem using a library such as LIBLINEAR. The class_weight="balanced" option adjusts the penalty for misclassification so that the algorithm accounts for imbalanced classes (e.g., fewer spam than ham messages). This helps prevent the model from predicting the majority class for most samples.

A limitation of LinearSVC is that it does not natively output class probabilities; it produces decision scores or class labels. For threshold tuning and interpretability, calibrated probabilities are desirable. Therefore, the model is wrapped with CalibratedClassifierCV to obtain probability estimates.

## 4.3 Calibrated Classifier

CalibratedClassifierCV wraps a base classifier and calibrates its outputs to produce probability estimates. The calibration process fits a secondary model (e.g., sigmoid or isotonic) on the base classifier's outputs to map them to well-calibrated probabilities. With method="sigmoid", Platt scaling is used: a sigmoid function is fit to the decision scores. With method="isotonic", isotonic regression is used for a non-parametric calibration. Sigmoid is typically faster and works well when the score distribution is roughly sigmoid-shaped.

The parameter cv=3 means 3-fold cross-validation is used internally: the training data is split into 3 folds, and each fold is used as a calibration set while the base classifier is trained on the other folds. This avoids overfitting the calibration to the training set. The final model is trained on the full training set, but the calibration curves are learned from out-of-fold predictions. The wrapped classifier's predict_proba method then returns probabilities for each class. For binary classification, the probability of the positive class (fraudulent) is used for threshold-based decision making and for metrics such as ROC-AUC.

## 4.4 RoBERTa Transformer

RoBERTa (Robustly Optimized BERT Pretraining Approach) is a Transformer-based language model developed by Facebook AI. It builds on BERT with improvements including longer training, larger batch sizes, removal of the Next Sentence Prediction task, and dynamic masking. The model is pre-trained on a large corpus of text to learn contextual word representations. For downstream tasks such as sequence classification, a classification head is added on top of the [CLS] token representation, and the entire model is fine-tuned on labeled task-specific data.

In this project, the Hugging Face transformers library provides AutoModelForSequenceClassification and AutoTokenizer. The tokenizer converts text into input IDs and attention masks, with a maximum length of 160 tokens (suitable for typical SMS length). The model is fine-tuned using the Trainer API with parameters such as learning rate 2e-5, weight decay 0.01, 4 epochs, batch size 16, and early stopping with patience 2. Evaluation is performed at the end of each epoch on the validation set. The metric for best model selection (accuracy or F1) is configurable. When a GPU is available, mixed precision (fp16) is used to accelerate training. The model outputs logits for the two classes; softmax is applied to obtain probabilities, and the probability of the fraud class is used for threshold tuning and final prediction.

## 4.5 Threshold Tuning

The default decision threshold of 0.5 may not be optimal when the cost of false positives and false negatives differs or when classes are imbalanced. The project implements a grid search over thresholds from 0.01 to 0.99 (99 points) on the validation set. For each threshold, predictions are computed as (proba >= threshold), and metrics (accuracy, F1, precision, recall, ROC-AUC) are computed. The threshold that maximizes the user-selected metric (accuracy or F1) is chosen. Tie-breaking is done by higher accuracy, then higher recall. The chosen threshold is saved in meta.json and used at inference time. This approach allows the user to optimize for the metric most relevant to their use case.

# 5. SYSTEM SPECIFICATIONS

## 5.1 Hardware Specifications

* Processor: Intel Core I9-14900K 3.20 GHz or equivalent
* RAM: 16 GB
* Hard Disk: 1 TB

The Intel Core i9-14900K processor with a base clock speed of 3.20 GHz provides strong computational capabilities for training and inference. Designed for intensive workloads, it offers exceptional multi-core performance, making it ideal for analytics and machine learning tasks. Its high clock speed and modern architecture ensure efficient data processing, quick task execution, and minimal latency during model training and prediction. With 16 GB of RAM, the system supports smooth operation of Python, machine learning libraries (NumPy, pandas, scikit-learn, PyTorch when used), and the Streamlit application. This capacity supports the operation of modern software and professional applications like machine learning frameworks without system slowdowns. The 1 TB hard disk offers sufficient space for the operating system, Python environment, datasets, saved models, and reports. While traditional hard drives provide large storage capacities, combining this with an SSD can greatly enhance system performance, particularly in boot times, dataset loading, and model checkpoint saving during training.

## 5.2 Software Specifications

* Operating System: Windows 10 or later, or Linux
* Python: 3.10 or later
* Frontend: Streamlit
* Backend: Python (NumPy, pandas, scikit-learn, joblib; optional: PyTorch, transformers, datasets, evaluate, accelerate)
* Dataset: UCI SMS Spam Collection (downloaded automatically)

The system is implemented in Python using scikit-learn for the TF-IDF and Linear SVM pipeline, and optionally Hugging Face Transformers and PyTorch for the Transformer model. Python 3.10 or later is required for type hints and modern language features. Python is an interpreted, high-level, general-purpose programming language that supports multiple paradigms including procedural, object-oriented, and functional programming. It is widely used in data science and machine learning due to its rich ecosystem of libraries, clear syntax, and rapid development cycle. The project leverages Python's strengths in text processing, numerical computation, and web framework integration.

Key libraries include: NumPy for numerical arrays; pandas for DataFrame operations; scikit-learn for TfidfVectorizer, LinearSVC, CalibratedClassifierCV, and metrics; joblib for model serialization; requests for dataset download; Streamlit for the web UI. Optional dependencies for the Transformer path: transformers, datasets, evaluate, accelerate, torch. The user interface is built with Streamlit, which provides a web-based interface for single-message and batch CSV detection without a separate HTML/CSS front end. Streamlit automatically handles styling, reactivity, and session state. Streamlit is a Python framework for building data applications and dashboards. It provides widgets such as text_area, button, slider, tabs, file_uploader, and dataframe display. When the user interacts with a widget, Streamlit re-runs the script from top to bottom, updating the displayed output. Session state (st.session_state) persists data across reruns, used for chat history in the AI Chat Bot tab. The framework does not require JavaScript or HTML; all UI is defined in Python. Running "streamlit run app.py" starts a local web server and opens the app in the default browser. No database is required for the core workflow; models and reports are stored as files (joblib for sklearn, Hugging Face save_pretrained format for Transformers, JSON for metadata and reports). The project structure follows a modular design: src/sms_fraud/ contains data, preprocessing, sklearn_model, hf_model, inference, config, and chatbot modules; train.py and predict.py are entry points for training and CLI prediction; app.py is the Streamlit entry point.

# 6. DESIGN AND DEVELOPMENT

## 6.1 Module Description

SMS Data Collection Module: This module is responsible for loading the SMS dataset required for training and testing. It downloads the UCI SMS Spam Collection from the UCI Machine Learning Repository if not already present, extracts the tab-separated file, and loads it into a pandas DataFrame with columns for raw text, normalized text, and binary label (0 legitimate, 1 fraudulent). It also provides stratified splitting into train, validation, and test sets with configurable fractions and random seed. Proper data handling ensures diversity and balanced splits for model development and evaluation.

Text Preprocessing Module: This module cleans and standardizes raw SMS messages. It applies Unicode normalization (NFKC), removes zero-width characters, replaces URLs with a placeholder, replaces email and phone number patterns with placeholders, replaces standalone numbers with a placeholder, lowercases text, and normalizes whitespace. This process reduces noise and ensures consistency so that both the classical and Transformer pipelines receive normalized input. The same preprocessing is applied at inference time for new messages.

Feature Extraction Module: For the sklearn pipeline, this module uses TF-IDF vectorization with n-gram range (1, 2), min_df, max_df, sublinear_tf, and strip_accents to transform preprocessed text into numerical vectors. For the Transformer path, the Hugging Face tokenizer converts text to input IDs and attention masks with a configurable max length. Feature extraction enables the classifiers to differentiate legitimate and fraudulent messages using informative and discriminative representations.

Fraud Detection Model Module: This module implements two model types. The first is a scikit-learn pipeline: TfidfVectorizer followed by CalibratedClassifierCV(LinearSVC(class_weight="balanced"), method="sigmoid", cv=3), which outputs calibrated probabilities. The second is an optional Hugging Face Transformer (e.g., RoBERTa) for sequence classification, fine-tuned with the Trainer API. The training script trains both (when model_type is auto or both are requested), evaluates them on the validation set with optional threshold tuning, selects the best model by the chosen metric, and evaluates the best model on the test set. The module supports saving and loading both model types for inference.

Evaluation and Deployment Module: This module evaluates the trained models using accuracy, precision, recall, F1-score, and ROC-AUC. It writes a training report (reports/train_report.json) with dataset and split information, validation and test metrics, and best model metadata. Deployment is achieved by saving the best model and meta.json to models/best. The inference module loads the best model based on meta.json and exposes predict_proba for use by the Streamlit app and the predict.py CLI. Batch CSV scoring and single-message prediction are supported with a configurable threshold.

## 6.2 System Architecture

The system architecture follows a modular pipeline design. At the top level, the flow is: Data Loading, Preprocessing, Feature Extraction, Model Training (or Model Loading for inference), Prediction, and Output. Each stage is implemented as a separate module or function, allowing independent testing and replacement. The data module (src/sms_fraud/data.py) handles downloading and loading the UCI dataset and provides stratified splits. The preprocessing module (src/sms_fraud/preprocessing.py) normalizes text. The sklearn_model module implements the TF-IDF plus LinearSVC pipeline; the hf_model module implements the Transformer fine-tuning. The inference module (src/sms_fraud/inference.py) abstracts model loading and prediction, so the Streamlit app and predict.py CLI need not know which model type is deployed. The app.py script orchestrates the web UI, and train.py orchestrates the training pipeline. File-based storage (models/best, reports/) keeps the system simple and portable without a database dependency. The architecture supports extension: new model types can be added by implementing the same save/load and predict_proba interfaces.

## 6.3 File Design

The project uses a file-based design for datasets, models, and reports. The data directory stores the downloaded UCI dataset (zip and extracted SMSSpamCollection file). The models/best directory contains the selected model: for sklearn, a single joblib file (sklearn.joblib) plus meta.json; for the Transformer, an hf_model subdirectory with saved model and tokenizer files plus meta.json. The reports directory holds train_report.json with full training and evaluation metrics. Preprocessing and feature extraction are performed in memory during training and inference; no separate preprocessed or feature files are required. File design supports reproducibility, versioning of the best model, and easy sharing of trained artifacts without a database.

## 6.4 Input Design

Inputs to the system are SMS messages in text form. For training, the input is the UCI SMS Spam Collection: a tab-separated file with label (ham/spam) and message text. The data module reads this file, normalizes labels to 0/1, and applies the preprocessing module to produce a normalized text column. For inference, input is either a single message string (UI or CLI with --text) or a CSV file with a user-selected text column (UI or CLI with --csv and --column). All text inputs are normalized using the same preprocessing function before feature extraction or tokenization. Input design ensures consistent, clean, and structured data for both training and prediction, with no separate database schema required.

## 6.5 Output Design

Outputs include classification results and training reports. For single-message prediction, the system outputs fraud probability (0 to 1) and a binary prediction (legitimate or fraudulent) based on a configurable threshold. In the Streamlit UI, these are displayed as metrics and a progress bar; in the CLI, they are printed as JSON. For batch CSV, the original DataFrame is augmented with fraud_probability and is_fraud columns; the user can view and download the result CSV. The training report (reports/train_report.json) contains dataset name and size, split sizes, validation and test metrics for both model types, best model metadata, and the chosen threshold. Output design ensures clarity, usability, and traceability for both end users and administrators.

## 6.6 Database Design

The current system does not use a relational database. Data flow is file-based: the dataset is read from disk, models are saved under models/best, and reports are written to reports/train_report.json. If a database were to be introduced for audit or scaling, a logical design could include: a table for raw or preprocessed messages (message_id, sender, content, timestamp, label); a table for model metadata (model_id, model_type, threshold, training_date, metrics); and a table for detection results (result_id, message_id, predicted_label, confidence_score, timestamp). For this project, the file-based approach is sufficient and keeps the system simple and portable.

# 7. IMPLEMENTATION AND TESTING

## 7.1 Implementation Details

The implementation phase converts the design into a working solution using Python. The UCI dataset is loaded and stratified into train, validation, and test sets. Preprocessing is applied to all message text. The sklearn pipeline (TF-IDF plus LinearSVC with calibration) is trained and evaluated on the validation set; optionally, the Hugging Face Transformer is trained and evaluated. The best model is selected, the decision threshold is tuned on the validation set, and the best model is evaluated on the test set. Results are written to reports/train_report.json and the best model is saved to models/best. The Streamlit app and predict.py load the best model and expose single-message and batch CSV prediction. Testing ensures that the system meets functional and performance requirements.

7.2 Unit Testing: Each module is tested in isolation. The data module is verified to download and parse the UCI dataset and to produce correct stratified splits. The preprocessing module is tested to ensure correct normalization of URLs, emails, phones, numbers, and whitespace. The sklearn pipeline is tested for fit and predict_proba; the inference module is tested for load_best and predict_proba for both model types. Unit testing helps identify and fix errors early and ensures that each component produces correct outputs before integration.

7.3 Integration Testing: After unit testing, the data, preprocessing, sklearn and optional HF model, and inference modules are integrated. Integration testing verifies that preprocessed text flows correctly into feature extraction and that the selected model loads and predicts consistently from the Streamlit app and the CLI. End-to-end workflow from training to saving the best model and from loading the model to producing predictions is validated.

7.4 Validation and Functional Testing: The trained model is evaluated on a held-out test set. Metrics such as accuracy, precision, recall, F1-score, and ROC-AUC are computed and stored in the training report. Stratified splitting and threshold tuning on the validation set (not the test set) ensure that the reported test performance reflects generalization. Cross-validation is inherent in the CalibratedClassifierCV for the sklearn pipeline.

Functional Testing: Functional testing checks that all features work as specified: data loading and splitting, preprocessing, training of both model types, model selection, threshold tuning, saving and loading the best model, single-message prediction in UI and CLI, batch CSV upload and download in the UI, and threshold adjustment in the sidebar. All operations are verified to behave as expected without errors under normal use.

7.5 Performance and Security Testing: Performance testing evaluates training time, inference time per message, and batch CSV throughput. The sklearn-only pipeline is tested for fast training and inference on CPU. When the Transformer is enabled, training and inference times are measured to ensure they are acceptable for the target environment. Memory usage during training and inference is also considered for scalability.

Security Testing: Security testing verifies that the system does not expose sensitive data inappropriately. Model files and reports are stored under project directories with standard file permissions. The Streamlit app does not persist user-entered messages unless the user downloads the batch CSV; API keys for the optional chatbot are entered in the UI and handled according to Streamlit best practices. No database or external service is used for core fraud detection beyond optional dataset download and optional OpenAI API for the chatbot.

# 8. EXPERIMENTAL RESULTS

## 8.1 Dataset Statistics

The UCI SMS Spam Collection dataset contains 5,572 messages. The default stratified split allocates 80 percent for training (4,457 messages), 10 percent for validation (557 messages), and 10 percent for test (558 messages). The dataset is class-imbalanced: approximately 13 percent of messages are spam (fraudulent) and 87 percent are ham (legitimate). Stratified splitting ensures that the train, validation, and test sets preserve this proportion, avoiding bias in evaluation. The raw data is tab-separated with two columns: label (ham or spam) and message text. After preprocessing, each message is normalized (URLs, emails, phones, numbers replaced with placeholders; lowercase; whitespace collapsed) and used for both the sklearn and Transformer pipelines.

## 8.2 Model Performance

Sample Training Report (reports/train_report.json structure):
The training report is a JSON object containing: dataset (name, n for total count), splits (train, val, test counts), sklearn_val@0.5 (validation metrics at default threshold), sklearn_val_best (validation metrics at tuned threshold with threshold value), hf_val@0.5 and hf_val_best (if HF was trained), hf_error (if HF failed to load), and best_test (final test metrics for the selected model). Example best_test: {"model_type": "sklearn", "threshold": 0.4, "accuracy": 0.9946, "f1": 0.9801, "precision": 0.9737, "recall": 0.9867, "roc_auc": 0.9973}. These values indicate that the model correctly classifies over 99 percent of test messages, with high precision and recall for the fraud class.

Experimental runs with the sklearn-only pipeline (model_type=sklearn) achieve strong results. Sample metrics from a typical training run: validation accuracy at threshold 0.5 is approximately 98.7 percent, with F1 around 95.1 percent. After threshold tuning (optimized for accuracy), the validation metrics improve to approximately 98.9 percent accuracy and 95.8 percent F1 with an optimal threshold of 0.4. Test set performance: accuracy 99.5 percent, F1 98.0 percent, precision 97.4 percent, recall 98.7 percent, ROC-AUC 99.7 percent. These numbers demonstrate that the TF-IDF plus Linear SVM baseline is highly effective for the UCI dataset. When the Transformer (RoBERTa) is available and successfully trained, it may achieve slightly higher metrics, but the sklearn model provides an excellent trade-off between accuracy and computational efficiency. The training report (reports/train_report.json) stores all metrics for auditing and comparison.

## 8.3 Sample Outputs

For a single message such as "Congratulations! You have won a free lottery ticket. Click here to claim", the system outputs a fraud probability (e.g., 0.92) and prediction (FRAUDULENT). For a legitimate message such as "Meeting at 3 pm tomorrow", the probability might be 0.05 and the prediction LEGITIMATE. The Streamlit UI displays these as metrics and a progress bar. The CLI prints JSON: {"fraud_probability": 0.92, "is_fraud": true}. Batch CSV output includes original columns plus fraud_probability and is_fraud. Users can adjust the threshold in the sidebar to trade off false positives and false negatives.

Confusion Matrix Interpretation:
A confusion matrix for binary classification has four cells: True Positives (TP, fraud correctly identified), True Negatives (TN, legitimate correctly identified), False Positives (FP, legitimate wrongly flagged as fraud), and False Negatives (FN, fraud wrongly classified as legitimate). Accuracy = (TP+TN)/(TP+TN+FP+FN). Precision = TP/(TP+FP) answers "Of all predicted frauds, how many were actually fraud?" Recall = TP/(TP+FN) answers "Of all actual frauds, how many did we catch?" F1 is the harmonic mean of precision and recall, balancing both. ROC-AUC summarizes the model's discrimination ability across all possible thresholds: a value of 1.0 means perfect separation, 0.5 means random guessing. For SMS fraud detection, high recall is desirable to catch most frauds, while high precision reduces user frustration from false alarms. Threshold tuning allows the operator to shift the balance: lower threshold increases recall but may increase false positives; higher threshold increases precision but may miss some frauds.

# 9. CONCLUSION AND FUTURE WORK

## 9.1 Conclusion

The SMS Fraud Detection System successfully demonstrates the application of machine learning and natural language processing to identify fraudulent messages effectively. By combining TF-IDF feature extraction with a Linear SVM baseline (and optionally a fine-tuned Transformer), the system analyzes SMS content, detects smishing attempts, and provides real-time classification through a Streamlit UI and CLI. Unlike traditional rule-based approaches, this system adapts to data-driven patterns, supports threshold tuning, and reduces false positives and false negatives while maintaining high accuracy and reliability.

The project highlights the importance of preprocessing, feature engineering, and robust model training and selection. The dual-model design ensures a strong baseline that runs efficiently on CPU and an optional state-of-the-art Transformer for higher accuracy when resources allow. Evaluation metrics (precision, recall, F1, accuracy, ROC-AUC) and a structured training report validate the effectiveness of the solution. The system enhances user security, protects privacy, and supports trust in mobile communication platforms through a proactive, intelligent, and adaptable fraud detection mechanism.

## 9.2 Project Outcomes

* High accuracy detection: The system achieves high accuracy in distinguishing legitimate and fraudulent SMS messages, reducing the risk of user exposure to smishing attacks.

* Real-time processing: The model classifies incoming messages quickly, enabling immediate feedback in the UI and batch CSV scoring for bulk analysis.

* Scalability: The framework supports the UCI dataset and can be extended to larger datasets; the sklearn pipeline is lightweight and suitable for resource-constrained environments.

* Reduced false positives: TF-IDF features combined with Linear SVM (and optional Transformer) and threshold tuning improve precision and minimize incorrect flagging of legitimate messages.

* Robust and adaptive: Periodic retraining with new data allows the system to learn emerging fraud patterns, ensuring long-term reliability.

* User privacy: Sensitive message content is processed in memory or in user-controlled files; the design avoids unnecessary persistence of user data.

* Actionable outputs: Training reports and batch CSV results enable administrators and users to monitor performance and take action on flagged messages.

## 9.3 Future Enhancement

* Integration of additional deep learning models (e.g., other Transformer architectures or lightweight models) for deployment on edge devices, with comparison to the current sklearn and RoBERTa options.

* Multilingual support: Extending preprocessing and model training to handle multiple languages or regional dialects for global telecom and diverse user bases.

* Hybrid feature extraction: Combining TF-IDF with word embeddings or other semantic features to improve intent detection and reduce false positives.

* Real-time feedback loop: Allowing users or administrators to flag misclassified messages and incorporating such feedback into periodic retraining for continuous learning.

* Mobile or lightweight app: Packaging the model and inference logic for on-device or lightweight server deployment with push notifications for high-risk messages.

* Enhanced security and privacy: Encryption of stored model artifacts and audit logs, and compliance with data protection regulations where applicable.

# 10. BIBLIOGRAPHY AND REFERENCES

1. S. M. Nagare, P. P. Dapke, S. A. Quadri, R. M. Gaikwad, R. M. Hasan and M. R. Baheti, "Support Vector Machine-Based SMS Spam Detection for Mobile Devices," 2025 International Conference on Applications of Machine Intelligence and Data Analytics (ICAMIDA), Aurangabad, India, 2025, pp. 1-4, doi: 10.1109/ICAMIDA64673.2025.11209256.

2. S. Singh Sikarwar, R. Arivukkodi, D. Krishnane, H. Sharma, A. Namdeo and K. Jadon, "Enhancing SMS Spam Detection Using Deep Learning Models: A Comparative Study," 2024 1st International Conference on Advances in Computing, Communication and Networking (ICAC2N), Greater Noida, India, 2024, pp. 1367-1372, doi: 10.1109/ICAC2N63387.2024.10895261.

3. Gupta, "Detection of Spam and Fraudulent calls Using Natural Language Processing Model," 2024 Sixth International Conference on Computational Intelligence and Communication Technologies (CCICT), Sonepat, India, 2024, pp. 423-427, doi: 10.1109/CCICT62777.2024.00075.

4. R. G. de Luna et al., "A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)," TENCON 2023 - 2023 IEEE Region 10 Conference (TENCON), Chiang Mai, Thailand, 2023, pp. 53-58, doi: 10.1109/TENCON58879.2023.10322491.

5. J. Christiawan, C. Kliveson, D. Suhartono and H. Lucky, "Classifying Fraud in Indonesia Short Message Service (SMS) Using Machine Learning and Deep Learning," 2024 International Conference on Informatics, Multimedia, Cyber and Information System (ICIMCIS), Jakarta, Indonesia, 2024, pp. 483-488, doi: 10.1109/ICIMCIS63449.2024.10956456.

6. P. Ghosh, P. Mondal, S. Dutta, R. Rakshit, M. Gangopadhyay and S. Banerjee, "AI-Powered Multi-Domain Misinformation and Fraud Detection with Source Transparency," 2025 6th International Conference on IoT Based Control Networks and Intelligent Systems (ICICNIS), Bengaluru, India, 2025, pp. 1697-1702, doi: 10.1109/ICICNIS66685.2025.11315647.

7. D. P, D. M R and V. S. S D, "Smart Detection of Malicious SMS Using AI: A Security Solution for Android Devices," 2025 Control Instrumentation System Conference (CISCON), Manipal, India, 2025, pp. 1-6, doi: 10.1109/CISCON66933.2025.11337557.

8. P. Y. J. Nicholas and P. C. Ng, "ScamDetector: Leveraging Fine-Tuned Language Models for Improved Fraudulent Call Detection," TENCON 2024 - 2024 IEEE Region 10 Conference (TENCON), Singapore, Singapore, 2024, pp. 422-425, doi: 10.1109/TENCON61640.2024.10902894.

9. K. S. Prakash, A. M. Abirami, S. Singh and E. Ramanujam, "Performance Evaluation of Meta-data features for Spam SMS Classification using Sequential Models," 2024 IEEE International Students' Conference on Electrical, Electronics and Computer Science (SCEECS), Bhopal, India, 2024, pp. 1-6, doi: 10.1109/SCEECS61402.2024.10481874.

10. S. F. Schwarz, P. Fonseca and A. Rocha, "Smishing Detection From a Messaging Platform View," in IEEE Access, vol. 13, pp. 143449-143464, 2025, doi: 10.1109/ACCESS.2025.3597903.

# 11. APPENDICES

## A. Table Structure

The system is file-based; the following logical table structure describes how data could be organized if a database were used.

Table: SMS_Message
Attributes: Message_ID (PK), Sender_Number, Message_Content, Timestamp, Label
Data types: Integer, Varchar, Text, Datetime, Integer
Purpose: Store raw SMS messages with metadata and class labels (0 legitimate, 1 fraudulent).

Table: Preprocessed_SMS
Attributes: Preprocess_ID (PK), Message_ID (FK), Cleaned_Text
Data types: Integer, Integer, Text
Purpose: Store normalized SMS text after preprocessing for feature extraction.

Table: Feature_Vector
Attributes: Feature_ID (PK), Message_ID (FK), TFIDF_Vector or Tokenized_Input
Data types: Integer, Integer, Blob or Text
Purpose: Store numerical representation of messages (TF-IDF or tokenizer output) for model input.

Table: Model
Attributes: Model_ID (PK), Model_Name, Model_Type, Training_Date, Accuracy, Precision, Recall, F1_Score, Threshold, Model_File_Path
Data types: Integer, Varchar, Varchar, Datetime, Float, Float, Float, Float, Float, Varchar
Purpose: Store trained model metadata and performance metrics.

Table: Detection_Result
Attributes: Result_ID (PK), Message_ID (FK), Predicted_Label, Confidence_Score, Detection_Timestamp
Data types: Integer, Integer, Integer, Float, Datetime
Purpose: Record prediction outcome and confidence for each message.

## B. Data Flow Diagrams

Level 0 Diagram Description:
The context diagram (Level 0) shows the system as a single process. External entities include the User (who provides SMS text or CSV uploads) and the UCI Repository (from which the dataset is downloaded during training). Data flows: User sends "SMS text or CSV file" to the system; the system returns "Fraud probability and prediction" to the User. During training, the system receives "Dataset" from the UCI Repository. Data stores used: D1 Dataset (raw and processed messages), D2 Model (trained model and meta), D3 Report (training report JSON).

Level 1 Diagram Description:
Process 1.0 Load Dataset: Input from UCI; output to D1. Reads or downloads the UCI SMS Spam Collection, extracts the tab-separated file, loads into DataFrame.
Process 2.0 Preprocess Text: Input from D1 (raw text); output to D1 (normalized text). Applies normalization, URL/phone replacement.
Process 3.0 Extract Features: Input from D1 (normalized text); output to internal storage (TF-IDF vectors or tokenized input). Transforms text to numerical features.
Process 4.0 Train Model: Input from features and labels; output to D2. Trains sklearn and optionally HF model, selects best, saves to models/best.
Process 5.0 Evaluate and Write Report: Input from model and test set; output to D3. Computes metrics, writes train_report.json.
Process 6.0 Load Model: Input from D2; output to memory. Called at inference startup.
Process 7.0 Predict: Input from User (message) and Loaded Model; output to User (probability, prediction).
Data flows connect these processes to D1, D2, D3, and the User.

Level 2 Diagram Description (Preprocess Text Explosion):
2.1 Normalize Unicode: NFKC normalization, zero-width character removal.
2.2 Replace Patterns: URL -> <URL>, email -> <EMAIL>, phone -> <PHONE>, number -> <NUM>.
2.3 Lowercase and Strip: Convert to lowercase, trim whitespace.
2.4 Collapse Whitespace: Multiple spaces to single space.

A data flow diagram describes the movement of data through a system. The transformation of data from input to output may be described logically and independently of physical components. DFDs are developed in levels; each process can be broken down into a more detailed DFD at the next level.

DFD Symbols:
* A square defines a source or destination of system data.
* An arrow identifies data flow; it is the pipeline through which information flows.
* A circle or bubble represents a process that transforms incoming data flow into outgoing data flows.
* An open rectangle is a data store, data at rest or a temporary repository of data.

Level 0: User inputs SMS text or CSV; system returns fraud probability and prediction. Data stores: Dataset, Model, Report.

Level 1: Processes include Load Dataset, Preprocess Text, Extract Features, Train Model, Select Best Model, Save Model, Load Model, Predict, Write Report. Data flows between these processes and the data stores.

Level 2: Preprocess Text can be exploded into Normalize Unicode, Replace URL/Email/Phone/Number, Lowercase, Collapse Whitespace. Train Model can be exploded into Train Sklearn Pipeline, Train Transformer (optional), Evaluate Validation, Tune Threshold, Select Best.

## C. Complete Source Code

train.py (main training script)

import argparse, json, shutil
from pathlib import Path
import numpy as np
from src.sms_fraud.data import load_sms_spam_collection, stratified_split
from src.sms_fraud.sklearn_model import train_sklearn, save as save_sklearn

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["auto", "sklearn", "hf"], default="auto")
    p.add_argument("--select_metric", choices=["accuracy", "f1"], default="accuracy")
    p.add_argument("--hf_model", default="roberta-base")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = load_sms_spam_collection("data")
    split = stratified_split(df, seed=args.seed)
    x_train = split.train["text"].tolist()
    y_train = split.train["label"].astype(int).tolist()
    x_val = split.val["text"].tolist()
    y_val = split.val["label"].astype(int).tolist()
    x_test = split.test["text"].tolist()
    y_test = split.test["label"].astype(int).tolist()

    sklearn_model, sk_metrics = train_sklearn(x_train, y_train, x_val, y_val)
    # ... threshold tuning, optional HF training, best model selection,
    # test evaluation, save to models/best, write reports/train_report.json

if __name__ == "__main__":
    main()

train.py full listing (continued): The script parses command-line arguments for model_type, select_metric, hf_model, epochs, batch_size, max_len, and seed. It loads the UCI dataset and creates stratified train/val/test splits. The sklearn model is trained and evaluated; _best_threshold performs grid search over 99 thresholds to find the optimal one for the selected metric. If model_type is auto or hf, the Hugging Face Transformer is trained (with exception handling if RoBERTa cannot be loaded). The best model (sklearn or HF) is selected by comparing validation metrics. The chosen model is evaluated on the test set and metrics are written to the report. The best model is saved to models/best using an atomic rename: a temporary directory is populated with model files and meta.json, then renamed to replace the previous best. The training report is written to reports/train_report.json.

sklearn_model.py (TF-IDF + LinearSVC, excerpt)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, strip_accents="unicode")
base = LinearSVC(class_weight="balanced")
clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
pipe.fit(text_train, y_train)
proba = pipe.predict_proba(text_val)[:, 1]

inference.py (load and predict, excerpt)

def load_best(model_root="models/best"):
    meta = json.loads((Path(model_root) / "meta.json").read_text(encoding="utf-8"))
    if meta["model_type"] == "sklearn":
        return LoadedModel(model_type="sklearn", model=load_sklearn(model_root))
    if meta["model_type"] == "hf":
        model, tok = load_hf(Path(model_root) / "hf_model")
        return LoadedModel(model_type="hf", model=model, tokenizer=tok, max_length=meta.get("max_length", 160))

def predict_proba(loaded, texts):
    texts_norm = [normalize_text(t) for t in texts]
    if loaded.model_type == "sklearn":
        return sk_predict(loaded.model, texts_norm)
    if loaded.model_type == "hf":
        # tokenize, model forward pass, softmax, return proba for class 1
        return probs

app.py (Streamlit UI, excerpt)

import streamlit as st
from src.sms_fraud.inference import load_best, predict_proba

st.set_page_config(page_title="SMS Fraud Detector", layout="wide")
st.title("SMS Fraud / Spam Detection")
loaded = load_best("models/best")
threshold = float(meta.get("threshold", 0.5))
tab1, tab2, tab3 = st.tabs(["Single SMS", "Batch CSV", "AI Chat Bot"])
with tab1:
    sms = st.text_area("Paste an SMS message", height=200)
    if st.button("Detect fraud"):
        probs = predict_proba(loaded, [sms])
        st.metric("Fraud probability", f"{probs[0]:.3f}")
        st.metric("Prediction", "FRAUDULENT" if probs[0] >= threshold else "LEGITIMATE")

src/sms_fraud/preprocessing.py (text normalization)

import re
import unicodedata
_RE_URL = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", flags=re.IGNORECASE)
_RE_PHONE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
_RE_NUM = re.compile(r"\b\d+\b")
_RE_WS = re.compile(r"\s+")

def normalize_text(text):
    if text is None: return ""
    if not isinstance(text, str): text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = _RE_URL.sub(" <URL> ", text)
    text = _RE_EMAIL.sub(" <EMAIL> ", text)
    text = _RE_PHONE.sub(" <PHONE> ", text)
    text = _RE_NUM.sub(" <NUM> ", text)
    text = text.lower().strip()
    text = _RE_WS.sub(" ", text)
    return text

src/sms_fraud/data.py (dataset loading, excerpt)

def load_sms_spam_collection(data_dir="data"):
    zip_path = _download_uci_dataset(Path(data_dir))
    txt_path = _extract_uci_txt(zip_path, Path(data_dir))
    df = pd.read_csv(io.StringIO(txt_path.read_text(encoding="utf-8", errors="replace")), sep="\t", header=None, names=["label_str", "text_raw"])
    df["label"] = (df["label_str"].str.strip().str.lower() == "spam").astype(int)
    df["text"] = df["text_raw"].map(normalize_text)
    return df[["text_raw", "text", "label"]]

def stratified_split(df, seed=42, train_frac=0.8, val_frac=0.1):
    train, temp = train_test_split(df, test_size=(1-train_frac), random_state=seed, stratify=df["label"])
    val_size = val_frac / (1 - train_frac)
    val, test = train_test_split(temp, test_size=(1-val_size), random_state=seed, stratify=temp["label"])
    return DatasetSplit(train=train.reset_index(drop=True), val=val.reset_index(drop=True), test=test.reset_index(drop=True))

predict.py (CLI)

loaded = load_best("models/best")
threshold = meta.get("threshold", 0.5)
if args.text:
    proba = predict_proba(loaded, [args.text])[0]
    print(json.dumps({"fraud_probability": proba, "is_fraud": proba >= threshold}))
if args.csv:
    df = pd.read_csv(args.csv)
    probs = predict_proba(loaded, df[args.column].astype(str).tolist())
    df["fraud_probability"] = probs
    df["is_fraud"] = df["fraud_probability"] >= threshold
    df.to_csv(args.out, index=False)

## D. Output Screens and User Manual

Installation Steps:
1. Ensure Python 3.10 or later is installed. Verify with: python --version
2. Create a virtual environment: python -m venv .venv
3. Activate it: On Windows, run .venv\Scripts\activate. On Linux/Mac, run source .venv/bin/activate
4. Install dependencies: pip install -r requirements.txt
5. Train the model: python train.py --model_type sklearn (or auto for both models)
6. Run the web UI: streamlit run app.py
7. Open the browser at the URL shown (typically http://localhost:8501)

Troubleshooting:
- If "No trained model" appears in the UI, run "python train.py --model_type sklearn" first.
- If RoBERTa fails to load (hf_error in report), the Transformer path is skipped; the sklearn model is used. Ensure internet access for first-time model download, or use model_type=sklearn to skip HF.
- For batch CSV, ensure the selected column contains text; empty or non-text cells may cause errors.
- If Streamlit port 8501 is in use, use: streamlit run app.py --server.port 8502
- On Windows, if Python is not found, add Python to PATH during installation or use the full path to python.exe.

Training Commands:
python train.py --model_type sklearn    (train only sklearn baseline)
python train.py --model_type auto        (train both, select best)
python train.py --model_type hf          (train only Transformer, requires GPU recommended)
python train.py --select_metric f1       (optimize for F1 instead of accuracy)
python train.py --epochs 4 --batch_size 16  (Transformer training parameters)

Prediction Commands (CLI):
python predict.py --text "Your message here"
python predict.py --csv data.csv --column message --out results.csv
python predict.py --text "Message" --threshold 0.6  (override threshold)

Streamlit UI Usage:
Single SMS Tab: Paste a message in the text area, click "Detect fraud". View fraud probability and prediction.
Batch CSV Tab: Upload a CSV file, select the column containing SMS text, click "Score CSV". Preview results and download the augmented CSV.
AI Chat Bot Tab: Optional chatbot for general questions. Select backend (local or OpenAI) in the sidebar. Local uses FLAN-T5; OpenAI requires API key.

The Streamlit UI shows:
* Title: SMS Fraud / Spam Detection
* Sidebar: Model metadata (JSON), Fraud threshold slider, AI Chat Bot backend and API key
* Tab Single SMS: Text area for message, "Detect fraud" button, Result section with Fraud probability, Prediction (LEGITIMATE or FRAUDULENT), progress bar
* Tab Batch CSV: File uploader, text column selector, "Score CSV" button, dataframe preview, Download predictions CSV button
* Tab AI Chat Bot: Chat history, chat input, Clear chat button

The CLI output for single message is JSON, e.g. {"fraud_probability": 0.92, "is_fraud": true}. For batch CSV, the script prints "Wrote: predictions.csv" and the output file contains original columns plus fraud_probability and is_fraud.

## F. requirements.txt (project dependencies)

streamlit>=1.35
pandas>=2.2
numpy>=2.0
scikit-learn>=1.4
joblib>=1.4
regex>=2024.5.15
requests>=2.32
tqdm>=4.66
openai>=1.0
transformers>=4.41
datasets>=2.19
evaluate>=0.4
accelerate>=0.33
torch>=2.2

## E. Glossary

Accuracy: The proportion of correct predictions (both true positives and true negatives) out of all predictions. Formula: (TP + TN) / (TP + TN + FP + FN).

CalibratedClassifierCV: A scikit-learn wrapper that calibrates a base classifier to output probability estimates using cross-validation.

F1-Score: Harmonic mean of precision and recall. Formula: 2 * (precision * recall) / (precision + recall). Balances precision and recall.

False Positive: A legitimate message incorrectly classified as fraudulent.

False Negative: A fraudulent message incorrectly classified as legitimate.

Feature Extraction: The process of converting raw text into numerical vectors (e.g., TF-IDF) suitable for machine learning models.

Ham: Legitimate or non-fraudulent SMS message.

Linear SVM: Linear Support Vector Machine; a classifier that finds a linear decision boundary maximizing the margin between classes.

N-gram: A contiguous sequence of n words or tokens. Unigram: single word. Bigram: two consecutive words.

Normalization: Text cleaning steps such as lowercasing, whitespace collapse, and replacement of variable patterns (URLs, phones) with placeholders.

Precision: Proportion of predicted frauds that are actually fraudulent. Formula: TP / (TP + FP).

Recall: Proportion of actual frauds that are correctly identified. Formula: TP / (TP + FN).

RoBERTa: Robustly Optimized BERT Pretraining Approach; a Transformer-based language model for NLP tasks.

ROC-AUC: Area Under the Receiver Operating Characteristic Curve; measures the model's ability to discriminate between classes across all thresholds. Value 1.0 indicates perfect discrimination.

Smishing: SMS phishing; fraudulent SMS messages designed to steal information or infect devices.

Sparsity: In TF-IDF, most entries in the feature vector are zero because each document contains only a small fraction of the vocabulary. Sparse matrix formats store only non-zero values for efficiency.

Stratified Split: Data splitting that preserves the proportion of each class (e.g., ham vs spam) in train, validation, and test sets.

TF-IDF: Term Frequency Inverse Document Frequency; a weighting scheme that reflects term importance in a document relative to the corpus.

Threshold: The probability cutoff above which a message is classified as fraudulent. Default 0.5; tunable for accuracy or F1 optimization.

Tokenization: Splitting text into tokens (words or subwords) for model input. Hugging Face tokenizers handle this for Transformer models.

True Positive: A fraudulent message correctly classified as fraudulent.
