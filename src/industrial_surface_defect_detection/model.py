from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import numpy as np
from sklearn.pipeline import Pipeline



class Model :
    def __init__(self,
                 scaler = StandardScaler(),
                 algorithm : str = 'isolation_forest',
                 **kwargs):
        self.algorithm = algorithm
        self.available_models = {
            "isolation_forest" : IsolationForest,
        }
        if algorithm not in self.available_models :
            raise ValueError(f"unsupported algorithm {algorithm}")

        self.model = self.available_models[algorithm](**kwargs)

        self.scaler = scaler
        # construct the pipeline
        self.pipeline = Pipeline([
            ("scaler" , self.scaler) ,
            ("model" , self.model) ,
        ])
        
    # fit the pipeline
    def fit(self, X_train : np.ndarray)  :
        self.pipeline.fit(X_train)

        return  self

    # predict & test
    def predict(self, X_train : np.ndarray) -> np.ndarray :

        predictions = self.pipeline.predict(X_train)

        # -1 : anomaly , 1 : normal
        return np.where(predictions == -1 , 1 , 0)

    # 
    def anomaly_score(self, X_scaled : np.ndarray) -> np.ndarray :

        #Higher anomaly score = More likely to be anomalous.
        return -self.pipeline.score_samples(X_scaled)

    #save model
    def save(self, path : str):

        joblib.dump(self, path)

    @staticmethod
    def load(path : str):

        return joblib.load(path)







        
