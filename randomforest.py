import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
submission_df = pd.DataFrame()

print("Training data:")
print(train_df.head())

id = "seatid"
target = "sold_first_hour"
categorical = ["seat_type", "speed_category"]

numerical = [c for c in train_df.columns if c not in categorical and c != id and c != target]

for c in categorical:
    le = LabelEncoder()
    combined = pd.concat([train_df[c], test_df[c]])
    le.fit(combined)
    train_df[c] = le.transform(train_df[c])
    test_df[c] = le.transform(test_df[c])

print("\nAfter encoding categorical features:")
print(train_df.head())

# Feature engineering
train_df['price_visibility_ratio'] = train_df['price'] / (train_df['visibility'] + 1)
test_df['price_visibility_ratio'] = test_df['price'] / (test_df['visibility'] + 1)
# Total visibility features combined
train_df['total_visibility_score'] = train_df['visibility'] + train_df['track_length_visible'] + train_df['range_visible']
test_df['total_visibility_score'] = test_df['visibility'] + test_df['track_length_visible'] + test_df['range_visible']

# weather protection = sun cover + rain cover
train_df['weather_protection'] = train_df['sun_cover'] + train_df['rain_cover']
test_df['weather_protection'] = test_df['sun_cover'] + test_df['rain_cover']

# Excitement potential = overtake probability * 10 + turns visible + braking zone * 2
train_df['excitement_score'] = train_df['overtake_probability'] * 10 + train_df['turns_visible'] + train_df['braking_zone'] * 2
test_df['excitement_score'] = test_df['overtake_probability'] * 10 + test_df['turns_visible'] + test_df['braking_zone'] * 2

scaler = StandardScaler()
train_df[numerical] = scaler.fit_transform(train_df[numerical])
test_df[numerical] = scaler.transform(test_df[numerical])

new_features = ['price_visibility_ratio', 'total_visibility_score', 'weather_protection', 'excitement_score']

train_df[new_features] = scaler.fit_transform(train_df[new_features])
test_df[new_features] = scaler.transform(test_df[new_features])

# print("\nAfter feature engineering and scaling:")
# print(train_df.head())

X_train = train_df.drop([id, target], axis=1)
y_train = train_df[target]
X_test = test_df.drop([id], axis=1)

print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# feature_importances = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': rf_model.feature_importances_
# }).sort_values('Importance', ascending=False)

# print("\nTop 10 most important features:")
# print(feature_importances.head(10))

print("\nTraining Gradient Boosting model...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=200,
    min_samples_leaf=50,
    subsample=0.8,
    random_state=42
)

gb_model.fit(X_train, y_train)

rf_preds = rf_model.predict_proba(X_test)[:, 1]
gb_preds = gb_model.predict_proba(X_test)[:, 1]

combined_preds = 0.7 * rf_preds + 0.3 * gb_preds

final_preds = (combined_preds >= 0.5).astype(int)

test_df[target] = final_preds
submission = test_df[[id, target]]
submission.to_csv("submission_randomforest_2.csv", index=False)

print("\nSubmission file created successfully!")

