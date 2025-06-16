from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

app = Flask(__name__)
CORS(app)

# Global variables to store models and scaler
models = {}
scaler = None
features_columns = None

def load_and_train_models():
    """Load data and train models"""
    global models, scaler, features_columns
    
    try:
        # Load yield data
        df_yield = pd.read_csv('yield.csv')
        df_yield = df_yield.rename(columns={"Value": "hg/ha_yield"})
        df_yield = df_yield.drop([
            'Year Code', 'Element Code', 'Element', 'Year Code', 
            'Area Code', 'Domain Code', 'Domain', 'Unit', 'Item Code'
        ], axis=1, errors='ignore')
        
        # Load rainfall data
        df_rain = pd.read_csv('rainfall.csv')
        df_rain = df_rain.rename(columns={" Area": "Area"})
        df_rain['average_rain_fall_mm_per_year'] = pd.to_numeric(
            df_rain['average_rain_fall_mm_per_year'], errors='coerce'
        )
        df_rain = df_rain.dropna()
        
        # Load pesticides data
        df_pes = pd.read_csv('pesticides.csv')
        df_pes = df_pes.rename(columns={"Value": "pesticides_tonnes"})
        df_pes = df_pes.drop(['Element', 'Domain', 'Unit', 'Item'], axis=1, errors='ignore')
        
        # Load temperature data
        avg_temp = pd.read_csv('temp.csv')
        avg_temp = avg_temp.rename(columns={"year": "Year", "country": "Area"})
        
        # Merge all data
        yield_df = pd.merge(df_yield, df_rain, on=['Year', 'Area'])
        yield_df = pd.merge(yield_df, df_pes, on=['Year', 'Area'])
        yield_df = pd.merge(yield_df, avg_temp, on=['Area', 'Year'])
        
        # One-hot encoding
        yield_df_onehot = pd.get_dummies(yield_df, columns=['Area', "Item"], prefix=['Country', "Item"])
        features = yield_df_onehot.loc[:, yield_df_onehot.columns != 'hg/ha_yield']
        label = yield_df['hg/ha_yield']
        
        # Drop Year column
        features = features.drop(['Year'], axis=1, errors='ignore')
        features_columns = features.columns.tolist()
        
        # Scale features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Split data
        train_data, test_data, train_labels, test_labels = train_test_split(
            features_scaled, label, test_size=0.3, random_state=42
        )
        
        # Train models
        model_configs = {
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0),
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=3, random_state=0),
            'SVM': svm.SVR(kernel='rbf', C=1.0, gamma='scale', max_iter=1000),  # Limit iterations for faster training
            'DecisionTree': DecisionTreeRegressor(random_state=0)
        }
        
        for name, model in model_configs.items():
            try:
                print(f"Training {name}...")
                model.fit(train_data, train_labels)
                y_pred = model.predict(test_data)
                r2 = r2_score(test_labels, y_pred)
                models[name] = {
                    'model': model,
                    'r2_score': r2
                }
                print(f"{name} R2 Score: {r2}")
            except Exception as e:
                print(f"Failed to train {name}: {e}")
                # Continue with other models
        
        return True
    except Exception as e:
        print(f"Error loading data or training models: {e}")
        return False

def create_sample_data():
    """Create sample data for demonstration if CSV files are not available"""
    global models, scaler, features_columns
    
    # Create sample training data
    np.random.seed(42)
    n_samples = 1000
    
    # Sample features
    rainfall = np.random.normal(800, 200, n_samples)
    pesticides = np.random.normal(50, 15, n_samples)
    temperature = np.random.normal(25, 5, n_samples)
    
    # Create synthetic yield based on features
    yield_values = (rainfall * 0.05 + pesticides * 0.3 + temperature * 2 + 
                   np.random.normal(0, 10, n_samples))
    
    # Create feature matrix
    features = np.column_stack([rainfall, pesticides, temperature])
    features_columns = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    
    # Scale features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split data
    train_data, test_data, train_labels, test_labels = train_test_split(
        features_scaled, yield_values, test_size=0.3, random_state=42
    )
    
    # Train models
    model_configs = {
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=0),
        'SVM': svm.SVR(kernel='rbf', C=1.0, gamma='scale', max_iter=1000),  # Limit iterations
        'DecisionTree': DecisionTreeRegressor(random_state=0)
    }
    
    for name, model in model_configs.items():
        try:
            print(f"Training {name}...")
            model.fit(train_data, train_labels)
            y_pred = model.predict(test_data)
            r2 = r2_score(test_labels, y_pred)
            models[name] = {
                'model': model,
                'r2_score': r2
            }
            print(f"{name} R2 Score: {r2}")
        except Exception as e:
            print(f"Failed to train {name}: {e}")
            # Continue with other models

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models and their performance"""
    model_info = {}
    for name, info in models.items():
        model_info[name] = {
            'name': name,
            'r2_score': round(info['r2_score'], 4)
        }
    return jsonify(model_info)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using selected model"""
    try:
        data = request.json
        model_name = data.get('model', 'RandomForest')
        
        # Extract features
        rainfall = float(data.get('rainfall', 0))
        pesticides = float(data.get('pesticides', 0))
        temperature = float(data.get('temperature', 0))
        
        # Create feature array (for simple version with 3 features)
        if len(features_columns) == 3:
            features = np.array([[rainfall, pesticides, temperature]])
        else:
            # For more complex version, you'd need to handle one-hot encoded features
            # This is a simplified version
            features = np.array([[rainfall, pesticides, temperature]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get selected model
        if model_name not in models:
            return jsonify({'error': 'Model not found'}), 400
        
        model = models[model_name]['model']
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'prediction': round(prediction, 2),
            'model_used': model_name,
            'model_r2_score': round(models[model_name]['r2_score'], 4)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Analyze input data and provide insights"""
    try:
        data = request.json
        rainfall = float(data.get('rainfall', 0))
        pesticides = float(data.get('pesticides', 0))
        temperature = float(data.get('temperature', 0))
        
        insights = []
        
        # Rainfall analysis
        if rainfall < 500:
            insights.append("Low rainfall may negatively impact crop yield. Consider irrigation.")
        elif rainfall > 1200:
            insights.append("High rainfall detected. Monitor for potential flooding or waterlogging.")
        else:
            insights.append("Rainfall levels appear optimal for most crops.")
        
        # Pesticides analysis
        if pesticides > 100:
            insights.append("High pesticide usage detected. Consider integrated pest management.")
        elif pesticides < 10:
            insights.append("Low pesticide usage. Monitor crop health for pest issues.")
        
        # Temperature analysis
        if temperature < 15:
            insights.append("Low temperatures may slow crop growth.")
        elif temperature > 35:
            insights.append("High temperatures may stress crops. Consider heat mitigation strategies.")
        
        return jsonify({'insights': insights})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask application...")
    
    # Try to load real data first, fall back to sample data
    if not load_and_train_models():
        print("Could not load CSV files, creating sample data...")
        create_sample_data()
    
    print("Models trained successfully!")
    app.run(debug=True, host='0.0.0.0', port=5000)