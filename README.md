ASSIGNMENT 4 REPORT
Software Development: Network Traffic to Image-Based CNN for DDoS Detection
1. Objective

The objective of Assignment 4 was to design and implement a complete software pipeline that:

Processes a large-scale network traffic dataset (Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv)

Converts structured network flow records into image representations

Trains a Convolutional Neural Network (CNN) on these images

Enables automatic detection of network attacks (DDoS) versus benign traffic

The solution was required to be executable, modular, and independent of the original dataset file, which exceeds 100 MB in size.


2. Dataset Description

The dataset used is:

CICIDS 2017 – Friday Working Hours Afternoon DDoS Traffic

Characteristics:

Flow-based network traffic data

Each row represents a summarized network flow

Contains numerical features such as packet counts, byte counts, flow duration, flags, and statistical measures

Includes a Label field identifying traffic as BENIGN or DDoS



3. Software Architecture Overview



3.1 Project Structure
<img width="372" height="667" alt="image" src="https://github.com/user-attachments/assets/371d0256-678b-4733-a696-166a4cc374ec" />

 



4. Data Processing Pipeline
4.1 Label Inspection (inspect_labels.py)

Reads the CSV file in chunks to avoid memory overflow

Identifies the correct label column (handling leading spaces)

Computes label distributions

Confirms presence of both BENIGN and DDoS records

This step prevents false assumptions caused by partial file loading.

4.2 Feature Preprocessing (preprocess.py)

Key operations:

Removal of non-numeric and identifier fields

Handling missing and infinite values

Feature scaling using StandardScaler

Conversion of labels to binary numeric format:

benign → 0

ddos → 1

The processed data is stored as NumPy arrays for efficient downstream processing.


5. Network Traffic to Image Conversion
5.1 Rationale

CNNs operate on spatial data. To leverage their feature extraction capabilities, each network flow is converted into a 2D grayscale image, where:

Each pixel represents a normalized network feature

Spatial proximity encodes feature relationships

5.2 Image Construction (make_images.py)

Process:

Each feature vector is normalized to [0,1]

Features are reshaped into a fixed-size matrix (e.g., 10 × 10)

The matrix is saved as a grayscale PNG image

Images are stored into class-specific directories:

benign/

ddos/

Images are split into:

Training set

Validation set

Test set

This structure is fully compatible with PyTorch’s ImageFolder loader.


6. CNN Model Architecture
6.1 Model Design (train_cnn.py)

The CNN architecture consists of:

Three convolutional blocks:

Conv2D → ReLU → Pooling

Adaptive average pooling to support flexible image sizes

Fully connected classifier head

The model processes single-channel grayscale images.

6.2 Training Configuration

Loss function: CrossEntropyLoss with class weighting

Optimizer: Adam

Learning rate: 1e-3

Device: CPU (with optional CUDA support)

Batch sizes optimized for Windows I/O constraints
<img width="1868" height="1021" alt="image" src="https://github.com/user-attachments/assets/a8d1cef5-a8ac-4a10-a579-42ad6db98e76" />


7. Evaluation and Outputs

The model is evaluated using:

Accuracy

Confusion Matrix

Precision, Recall, F1-score

Outputs generated:

cnn_ddos.pt – trained model weights

metrics.txt – evaluation summary


8. Execution

The full pipeline can be executed using:

python src/run_pipeline.py

or step-by-step for debugging and validation.


9. Assignment 4 Conclusion

Assignment 4 successfully delivered a complete, modular, and reproducible software solution that converts large-scale network traffic data into images and trains a CNN to detect DDoS attacks. The system is scalable, interpretable, and suitable for real-world cybersecurity experimentation.
