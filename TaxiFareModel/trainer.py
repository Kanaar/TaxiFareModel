from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = self.set_pipeline()
        self.X = X
        self.y = y

    def set_pipeline(self):
        '''returns a pipelined model'''
        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), StandardScaler())
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']
        preprocessor = ColumnTransformer([('time', pipe_time, time_cols),
                                          ('distance', pipe_distance, dist_cols)]
                                          )
        print('set pipeline')
        return Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])

    def run(self):
        """set and train the pipeline"""
        print('training...')
        return self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        '''prints and returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(f"rmse: {rmse}")
        return rmse

if __name__ == "__main__":
    data = get_data()
    data = clean_data(data)
    X = data.drop(columns=['fare_amount'])
    y = data.fare_amount
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    trainer = Trainer(X_train, y_train)
    trainer.run()
    trainer.evaluate(X_test, y_test)
