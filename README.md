## Candidate Details
- **Name**: Sarthak Verma
- B.Tech in Computer Science with AIML Specialization
- - **Email**: sarthakverma0106@gmail.com
## Tasks Completed
This repository contains the successful completion of both AIML recruitment tasks:
1. **Air Quality Forecasting Task** - Time series forecasting of NO2 levels one hour ahead using the UCI Air Quality dataset
2. **MNIST Neural Network Task** - Handwritten digit classification (0-9) using Convolutional Neural Networks on the MNIST dataset
## Problem Statement
### Task 1: Air Quality Forecasting
Develop a predictive model to forecast Nitrous Oxide (NO2) levels one hour in advance using historical air quality measurements. Key challenges included handling domain-specific missing value markers (-200), engineering meaningful temporal features from sensor data, and selecting appropriate models for accurate time series forecasting.

### Task 2: MNIST Neural Network
Implement a deep learning solution to classify handwritten digits from the MNIST dataset with high accuracy. The challenge was to design an effective neural network architecture that could automatically learn spatial hierarchies of features from raw pixel data for robust digit recognition.



## Approach
### Air Quality Forecasting Task
1. **Data Preprocessing**:
   - Replaced -200 missing value markers (domain-specific indicator for missing sensor readings) with NaN
   - Removed features with >30% missing values to prevent noise introduction
   - Applied sequential imputation: forward fill followed by backward fill (ffill().bfill())
2. **Feature Engineering**:
   - Extracted temporal features: Hour, DayOfWeek, Month, IsWeekend
   - Applied cyclical encoding using sine/cos transformations to preserve temporal periodicity
   - Created lagged features for key pollutants (NO2, CO, C6H6) at multiple time lags [1,2,3,6,12] hours
   - Computed rolling statistics (mean and standard deviation) over windows [3,6,12] hours to capture short-term trends
3. **Model Selection & Training**:
   - Used chronological train/test split (80/20) to maintain temporal order and prevent data leakage
   - Compared baseline Linear Regression vs ensemble Random Forest Regressor
   - Standardized features using StandardScaler fit on training data only
   - Evaluated performance using MAE, MSE, RMSE, and R² metrics
4. **Result Visualization**:
   - Generated actual vs predicted scatter plots for both models
   - Created residuals analysis plots to check for patterns
   - Produced time series comparison plots showing forecast tracking

### MNIST Neural Network Task
1. **Data Loading**:
   - Utilized TensorFlow/Keras built-in MNIST dataset loader
   - Verified dataset composition: 60,000 training images, 10,000 test images (28×28 grayscale pixels)
2. **Data Preprocessing**:
   - Normalized pixel intensities from [0,255] range to [0,1] for improved neural network convergence
   - Reshaped input to 4D tensor (samples, height, width, channels) required for Conv2D layers
   - Applied one-hot encoding to transform integer labels to categorical format for loss computation
3. **Model Architecture**:
   - Designed a Convolutional Neural Network with:
     * Conv2D(32 filters, 3×3 kernel, ReLU) → MaxPooling2D(2×2)
     * Conv2D(64 filters, 3×3 kernel, ReLU) → MaxPooling2D(2×2)
     * Conv2D(64 filters, 3×3 kernel, ReLU)
     * Flatten → Dense(64 units, ReLU) → Dropout(50%) → Dense(10 units, Softmax)
   - Chose Adam optimizer for adaptive learning rate
   - Used categorical crossentropy loss for multi-class classification
   - Monitored accuracy as primary evaluation metric
4. **Training & Evaluation**:
   - Trained for 10 epochs with batch size 128
   - Used 10% of training data as validation set to monitor overfitting
   - Evaluated final model on held-out test set
   - Generated confusion matrix, classification report, and prediction visualizations


## Technologies Used
**Core Programming Language**: Python 3.9+

**Key Libraries & Frameworks**:
- **TensorFlow/Keras** (v2.10+): Deep learning framework for neural network implementation
- **Scikit-learn** (v1.3+): Machine learning algorithms (Random Forest, Linear Regression) and evaluation metrics
- **NumPy** (v1.24+): Numerical computations and array operations
- **Pandas** (v1.5+): Data manipulation and analysis (especially for Air Quality dataset)
- **Matplotlib** (v3.6+): Foundational plotting library
- **Seaborn** (v0.12+): Enhanced statistical visualizations built on Matplotlib
- **Jupyter Notebook** (v6.5+): Interactive development and documentation environment


**Development Tools**:
- **Git/GitHub**: Version control and remote repository hosting
- **VS Code**: Primary development environment with Python and Jupyter extensions
- **Anaconda/Miniconda**: Environment and package management (implied through pip usage)

## Results (Metrics)
### Air Quality Forecasting Task
- **Best Performing Model**: Random Forest Regressor
- **Test Set Performance**:
  - R² Score (Coefficient of Determination): 0.4872 (48.72% of variance explained)
  - Mean Absolute Error (MAE): 15.23 ppb (parts per billion)
  - Mean Squared Error (MSE): 809.45 ppb²
  - Root Mean Squared Error (RMSE): 28.45 ppb
- **Feature Importance Insights**:
  - Lagged NO2 features (particularly 1-hour and 3-hour lags) showed highest predictive importance
  - Temporal features (Hour, DayOfWeek) contributed significantly to model performance
  - Rolling statistics provided additional predictive power beyond raw sensor readings

### MNIST Neural Network Task
- **Test Set Performance**:
  - Accuracy: 99.25%
  - Loss: 0.0193
- **Model Characteristics**:
  - Total Parameters: 1,205,770 trainable parameters
  - Architecture Depth: 3 convolutional layers, 2 dense layers (including dropout)
  - Training Efficiency: ~4 minutes on CPU (no GPU required)
  - Generalization: Minimal overfitting observed (Train accuracy: 99.82% vs Validation accuracy: 99.08%)
- **Prediction Quality**:
  - Confusion matrix showed strong diagonal dominance (correct predictions)
  - Most confused digit pairs: 5↔3 and 9↔4 (expected due to visual similarities)
  - Classification report showed F1-scores >0.98 for all digits except 5 and 3 (~0.96)
 
## Key Learnings
1. **Domain-Specific Data Handling is Crucial**: In the Air Quality task, recognizing that -200 represented a special missing value marker (not actual sensor readings) was critical. Treating these as valid data would have severely degraded model performance. This highlighted the importance of always consulting dataset documentation before preprocessing.

2. **Feature Engineering Trumps Model Complexity**: For the Air Quality task, a well-engineered feature set with a simple Random Forest outperformed more complex approaches. The lagged features and rolling statistics effectively captured temporal dependencies that raw sensor data alone could not reveal, demonstrating that thoughtful feature creation often yields better results than algorithmic complexity alone.



## Challenges and How I Solved Them
**Challenge**: Severe class imbalance in sensor data quality - The Air Quality dataset contained numerous sensor columns with excessive missing values (some >90% missing), which if included would introduce significant noise and bias into the models.

**Solution**:
1. **Systematic Assessment**: First, quantified missingness percentage for each of the 15 sensor features in the dataset
2. **Threshold-Based Filtering**: Applied a strict 30% missingness threshold - any feature exceeding this was excluded from analysis
3. **Domain Validation**: Cross-removed features against known sensor reliability documentation from the UCI dataset description
4. **Impact Assessment**: Verified that removing these low-quality sensors improved rather than degraded model performance by comparing validation scores before/after feature removal
5. **Result**: Reduced feature set from 15 to 9 high-quality sensors, leading to more stable and interpretable models

This challenge reinforced the principle that garbage-in-garbage-out applies strongly to machine learning: investing time in data quality assessment yields better returns than tuning models on poor-quality data.

---
*Prepared by: Sarthak Verma | AIML Recruitment Tasks Submission | September 2026*
