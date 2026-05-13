from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def train_model(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = X.select_dtypes(include=['int64', 'float64'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    joblib.dump(model, "models/model.pkl")

    return mse
