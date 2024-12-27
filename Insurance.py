import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

class InsuranceModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def prepare_data(self, df):
        df = df.copy()
        df['sex'] = df['sex'].map({'female': 0, 'male': 1})
        df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
        
       
        region_map = {
            'southwest': [1,0,0,0],
            'southeast': [0,1,0,0],
            'northwest': [0,0,1,0],
            'northeast': [0,0,0,1]
        }
        region_values = df['region'].map(region_map).tolist()
        df[['region_sw', 'region_se', 'region_nw', 'region_ne']] = pd.DataFrame(region_values, index=df.index)
        df.drop('region', axis=1, inplace=True)
        
        return df
        
    def train(self, X, y):
        self.model = RandomForestRegressor(
            n_estimators=50,  
            max_depth=10,     
            min_samples_split=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained or loaded")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def save(self, path='insurance_model.joblib'):
        joblib.dump((self.model, self.scaler), path)
        
    def load(self, path='insurance_model.joblib'):
        self.model, self.scaler = joblib.load(path)

from Insurance import InsuranceModel
import pandas as pd

df = pd.read_csv('/home/kimath/Desktop/PROJECTS/Projects/INSURENCE/insurence/insur/static/insurance.csv')
model = InsuranceModel()

X = model.prepare_data(df.drop('expenses', axis=1))
y = df['expenses']

model.train(X, y)
model.save()