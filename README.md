# F1 US Grand Prix Seat Sale Prediction (1st/168) with Random Forest and Gradient Boosting

## Project Overview

This project focuses on predicting which seats at the F1 US Grand Prix will sell out within the first hour of ticket sales. Using machine learning models, we analyze various seat attributes to determine their likelihood of quick sales, helping optimize pricing and marketing strategies.

1st/168 in the **Kaggle Competition** [F1 US Grand Prix Sale Predictions](https://www.kaggle.com/competitions/f-1-us-grand-prix-sale-predictions/overview)

## Dataset

The dataset contains information about seats at the F1 US Grand Prix, including:

- **Seat attributes**: price, distance from track, height, seating angle, visibility metrics
- **Experience factors**: turns visible, track length visible, range visible
- **Comfort features**: sun cover, rain cover
- **Track characteristics**: speed category, overtake probability, braking zone
- **Seat categories**: General, Premium, VIP

## Models and Performance

We implemented several machine learning models with varying performance:

| Model | Accuracy |
|-------|----------|
| K-Nearest Neighbors (KNN) | 0.64 |
| Random Forest + Gradient Boosting Ensemble | 0.7393 |

Our ensemble approach combining Random Forest and Gradient Boosting Classifiers achieved top 5 placement out of 164 submissions in the Kaggle competition.

## Feature Engineering

To improve model performance, we created several engineered features:

1. **Price-visibility ratio**: Relationship between price and visibility
2. **Total visibility score**: Combined visibility metrics
3. **Weather protection**: Combined sun and rain cover
4. **Excitement score**: Weighted combination of overtake probability, turns visible, and braking zone

## Model Details

### KNN Implementation
The KNN model uses 7 neighbors to classify seats based on similarity to known examples. While simpler, it provides a reasonable baseline with 0.64 accuracy.

### Ensemble Approach
Our best-performing solution uses a weighted ensemble:
- **Random Forest**: 70% contribution
  - 100 estimators
  - Max depth of 15
  - Balanced class weights
- **Gradient Boosting**: 30% contribution
  - 100 estimators
  - Learning rate of 0.1
  - Max depth of 5

## Feature Importance

The Random Forest model identified the most influential features for predicting first-hour sales, helping understand key factors driving quick purchases.

## How to Run

1. Ensure you have the required dependencies:
   ```
   pandas
   numpy
   scikit-learn
   ```

2. Place the training and test data files in the same directory:
   - train.csv
   - test.csv

3. Run either model:
   - For KNN: `python knn.py`
   - For Ensemble: `python randomforest.py`

4. The output will be saved as a CSV file with predictions.

## Conclusion

This project demonstrates the effectiveness of ensemble methods for predicting F1 ticket sales. The Random Forest and Gradient Boosting combination significantly outperforms the simpler KNN approach, achieving 73.93% accuracy and placing 1st/168 participants of the Kaggle competition.

The feature engineering process revealed important insights about what drives quick seat sales, which could help venue managers and event organizers optimize their pricing and marketing strategies for future events.
