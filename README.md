AI fault detetction in Electrical Signals using an autoencoder

##  Overview
This project is an implementation of an unsupervised anomaly detection system for electrical signals using an autoencoder. The model learns to reconstruct normal operating conditions of electrical currents and voltages. Any significant deviation in reconstruction error is flagged as a potential electrical fault.

##  Workflow
1. **Data Loading**  
   Reads the dataset (`electrical_fault_detect_dataset.csv`) containing current (`Ia, Ib, Ic`) and voltage (`Va, Vb, Vc`) signals, along with fault labels (`Output (S)`).

2. **Preprocessing**  
   Missing values/ Nan are filled with zeros.  
   Features are standardized using `StandardScaler` (mean = 0, std = 1).

3. **Training Data Selection**  
   Only samples labeled as *normal* (`Output (S) = 0`) are used to train the autoencoder.  
   Data is split into training and validation sets.

4. **Model Architecture**  
   A multi-layer autoencoder built with TensorFlow/Keras.  
   Encoder compresses input into a lower-dimensional representation.  
   Decoder reconstructs the original input.  
   Loss function: Mean Squared Error (MSE).

5. **Training**  
   Model is trained for 100 epochs with batch size 32.  
   Validation data is used to monitor reconstruction performance.  
   Trained model is saved as `transformer_electrical_autoencoder.h5`.

6. **Fault Detection**  
   Reconstruction error (MSE) is computed for test data.  
   Threshold = mean + 3 × standard deviation of validation errors.  
   Samples with error above threshold are flagged as anomalies.  
   Final output reports the number of detected potential faults.

##  Example Output
376/376 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step
41/41 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
Detected 5463 potential electrical faults

## 🛠️ Requirements
- Python 3.8–3.11  
- Libraries:  
  - pandas  
  - numpy  
  - scikit-learn  
  - scipy  
  - tensorflow  

Install dependencies:
```bash
pip install pandas numpy scikit-learn scipy tensorflow

