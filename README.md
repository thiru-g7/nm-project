import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('accidents.csv')

# Handle missing values
df.dropna(inplace=True)

# Encode categorical data
le = LabelEncoder()
df['weather'] = le.fit_transform(df['weather'])
df['location'] = le.fit_transform(df['location'])

# Define features and target
X = df[['time', 'location', 'weather']]
y = df['severity']  # Assume 'severity' is the target to predict

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
