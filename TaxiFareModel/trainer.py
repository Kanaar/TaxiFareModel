# imports

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline():
        '''returns a pipelined model'''
        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), StandardScaler())
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']
        preprocessor = ColumnTransformer([('time', pipe_time, time_cols),
                                          ('distance', pipe_distance, dist_cols)]
                                          )
        return Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])

    def run(self):
        """set and train the pipeline"""
        return pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        '''prints and returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(f"rmse: {rmse}")
        return rmse

if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
