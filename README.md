## Candidate Details
Name: Sarthak Verma
B.Tech Computer Science, AIML Specialization
Email: sarthakverma0106@gmail.com

## Tasks Completed
Both tasks are done:
1. Air Quality Forecasting — predicting NO2 levels one hour ahead using the UCI Air Quality dataset
2. MNIST Neural Network — classifying handwritten digits (0-9) using a CNN

## Problem Statement

**Task 1:** The goal was to forecast NO2 levels one hour into the future using past sensor readings. The dataset itself was messy — a lot of missing values, and some columns barely usable — so a good chunk of the work was just getting the data into a state where a model could actually learn from it.

**Task 2:** Build a neural network that classifies MNIST digits. Fairly standard task, but the point was to actually understand what each layer is doing rather than just stacking things that work.

## Approach

### Task 1 — Air Quality
The dataset uses `-200` as a placeholder for missing sensor readings, which I only realized after checking the UCI documentation — treating those as real values would have wrecked the model. I converted them to NaN, dropped columns that were missing more than 30% of their data, and filled the rest with forward-fill then backward-fill.

For features, I pulled out hour, day of week, month, and whether it was a weekend, and used sine/cosine encoding on the cyclical ones (hour, day) so the model doesn't treat 23:00 and 00:00 as far apart. I also added lagged versions of NO2, CO, and C6H6 (1, 2, 3, 6, 12 hours back) and rolling mean/std over a few window sizes, since air quality tends to depend heavily on recent history.

Split the data 80/20 by time (not randomly, since it's a forecasting problem) and scaled features using a scaler fit only on the training set. Tried both Linear Regression and Random Forest, and compared them using MAE, MSE, RMSE, and R².

### Task 2 — MNIST
Loaded MNIST through Keras, normalized pixels to 0–1, reshaped to the 4D shape Conv2D layers expect, and one-hot encoded the labels.

Model: two Conv2D+MaxPooling blocks, one more Conv2D layer, then Flatten → Dense(64, ReLU) → Dropout(0.5) → Dense(10, Softmax). Trained with Adam, categorical crossentropy, 10 epochs, batch size 128, with 10% of training data held out for validation.

## Technologies Used
Python, TensorFlow/Keras, scikit-learn, NumPy, pandas, Matplotlib, Seaborn, Jupyter Notebook, Git/GitHub, VS Code.

## Results

**Air Quality (Random Forest, best model):**
- R²: 0.4872
- MAE: 15.23 ppb
- RMSE: 28.45 ppb

Lagged NO2 (1hr, 3hr) turned out to matter the most, followed by hour-of-day. Rolling stats helped a bit beyond that.

**MNIST:**
- Test accuracy: 99.25%, test loss: 0.0193
- Train vs. val accuracy (99.82% vs 99.08%) — close enough that overfitting isn't a big issue
- Most confusion was between 5/3 and 9/4, which makes sense given how similar those digits can look when handwritten

## Key Learnings
1. Missing-value markers aren't always obvious — `-200` looked like a real number until I checked the documentation. Lesson: always read the dataset description before touching the data.
2. Good features beat a fancier model. Random Forest with well-engineered lag/rolling features did better than I expected, and I didn't need to reach for anything more complex.
3. Time series splits have to respect chronological order — shuffling before splitting would have leaked future information into training, and I almost did this before catching it.

## Challenges
Biggest one was dealing with how much of the sensor data was missing — some columns were unusable (over 90% missing). I checked missingness per column, dropped anything over 30%, and cross-checked against the UCI docs to make sure I wasn't dropping something important. Went from 15 sensor columns down to 9, and the models were noticeably more stable after that.
