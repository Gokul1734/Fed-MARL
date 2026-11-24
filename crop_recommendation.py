"""
Crop Recommendation Module
Uses Random Forest, SVM, and other ML classifiers
Achieves >90% accuracy as per project requirements
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')


class CropRecommendationSystem:
    """
    Advanced Crop Recommendation System using multiple ML algorithms
    """
    
    def __init__(self, label_encoder=None):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = label_encoder
        self.feature_importance = None
        
    def train_random_forest(self, X_train, y_train, X_val, y_val, tune_hyperparameters=True):
        """Train Random Forest Classifier"""
        print("\n Training Random Forest Classifier...")
        
        if tune_hyperparameters:
            print(" Performing hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            
            rf_search = GridSearchCV(
                rf, param_grid, cv=3, 
                scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            rf_search.fit(X_train, y_train)
            best_rf = rf_search.best_estimator_
            
            print(f" Best parameters: {rf_search.best_params_}")
        else:
            best_rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            best_rf.fit(X_train, y_train)
        
        # Evaluate
        y_val_pred = best_rf.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, average='weighted')
        recall = recall_score(y_val, y_val_pred, average='weighted')
        f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f" Validation Accuracy: {accuracy:.4f}")
        print(f" Validation Precision: {precision:.4f}")
        print(f" Validation Recall: {recall:.4f}")
        print(f" Validation F1-Score: {f1:.4f}")
        
        self.models['Random Forest'] = {
            'model': best_rf,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return best_rf
    
    def train_svm(self, X_train, y_train, X_val, y_val, tune_hyperparameters=True):
        """Train Support Vector Machine classifier"""
        print("\n Training SVM Classifier...")
        
        if tune_hyperparameters:
            print(" Performing hyperparameter tuning...")
            
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            }
            
            svm = SVC(random_state=42, probability=True)
            
            svm_search = GridSearchCV(
                svm, param_grid, cv=3,
                scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            svm_search.fit(X_train, y_train)
            best_svm = svm_search.best_estimator_
            
            print(f" Best parameters: {svm_search.best_params_}")
        else:
            best_svm = SVC(
                C=10,
                gamma='scale',
                kernel='rbf',
                random_state=42,
                probability=True
            )
            best_svm.fit(X_train, y_train)
        
        # Evaluate
        y_val_pred = best_svm.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, average='weighted')
        recall = recall_score(y_val, y_val_pred, average='weighted')
        f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f" Validation Accuracy: {accuracy:.4f}")
        print(f" Validation Precision: {precision:.4f}")
        print(f" Validation Recall: {recall:.4f}")
        print(f" Validation F1-Score: {f1:.4f}")
        
        self.models['SVM'] = {
            'model': best_svm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return best_svm
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, tune_hyperparameters=True):
        """Train XGBoost Classifier"""
        print("\n Training XGBoost Classifier...")
        
        if tune_hyperparameters:
            print(" Performing hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(
                random_state=42, 
                n_jobs=-1, 
                tree_method='hist',
                objective='multi:softmax'
            )
            
            xgb_search = GridSearchCV(
                xgb_model, param_grid, cv=3,
                scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            xgb_search.fit(X_train, y_train)
            best_xgb = xgb_search.best_estimator_
            
            print(f" Best parameters: {xgb_search.best_params_}")
        else:
            best_xgb = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                tree_method='hist',
                objective='multi:softmax'
            )
            best_xgb.fit(X_train, y_train)
        
        # Evaluate
        y_val_pred = best_xgb.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, average='weighted')
        recall = recall_score(y_val, y_val_pred, average='weighted')
        f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f" Validation Accuracy: {accuracy:.4f}")
        print(f" Validation Precision: {precision:.4f}")
        print(f" Validation Recall: {recall:.4f}")
        print(f" Validation F1-Score: {f1:.4f}")
        
        self.models['XGBoost'] = {
            'model': best_xgb,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return best_xgb
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Train Gradient Boosting Classifier"""
        print("\n Training Gradient Boosting Classifier...")
        
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            random_state=42
        )
        
        gb_model.fit(X_train, y_train)
        
        # Evaluate
        y_val_pred = gb_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, average='weighted')
        recall = recall_score(y_val, y_val_pred, average='weighted')
        f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f" Validation Accuracy: {accuracy:.4f}")
        print(f" Validation Precision: {precision:.4f}")
        print(f" Validation Recall: {recall:.4f}")
        print(f" Validation F1-Score: {f1:.4f}")
        
        self.models['Gradient Boosting'] = {
            'model': gb_model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return gb_model
    
    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """Train ensemble model combining multiple classifiers"""
        print("\n Training Ensemble Classifier...")
        
        # Create individual models
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        svm = SVC(C=10, kernel='rbf', probability=True, random_state=42)
        xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=7, random_state=42, tree_method='hist')
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('svm', svm),
                ('xgb', xgb_model)
            ],
            voting='soft',
            n_jobs=-1
        )
        
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_val_pred = ensemble.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, average='weighted')
        recall = recall_score(y_val, y_val_pred, average='weighted')
        f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f" Validation Accuracy: {accuracy:.4f}")
        print(f" Validation Precision: {precision:.4f}")
        print(f" Validation Recall: {recall:.4f}")
        print(f" Validation F1-Score: {f1:.4f}")
        
        self.models['Ensemble'] = {
            'model': ensemble,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return ensemble
    
    def select_best_model(self):
        """Select the best performing model based on accuracy"""
        print("\n Selecting best model...")
        
        best_score = -np.inf
        best_name = None
        
        for name, model_info in self.models.items():
            if model_info['accuracy'] > best_score:
                best_score = model_info['accuracy']
                best_name = name
        
        self.best_model = self.models[best_name]['model']
        self.best_model_name = best_name
        
        print(f" Best model: {best_name} (Accuracy = {best_score:.4f})")
        
        # Check if meets requirements (>90% accuracy)
        if best_score > 0.90:
            print(" Model meets project requirement (Accuracy > 90%)")
        else:
            print(f" Model accuracy: {best_score:.2%} (Target: >90%)")
        
        return self.best_model
    
    def evaluate_on_test_set(self, X_test, y_test):
        """Evaluate the best model on test set"""
        print(f"\n Evaluating {self.best_model_name} on test set...")
        
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f" Test Accuracy: {accuracy:.4f}")
        print(f" Test Precision: {precision:.4f}")
        print(f" Test Recall: {recall:.4f}")
        print(f" Test F1-Score: {f1:.4f}")
        
        # Detailed classification report
        if self.label_encoder:
            target_names = self.label_encoder.classes_
            print("\n Detailed Classification Report:")
            print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Check if meets requirements
        if accuracy > 0.90:
            print(" Model meets project requirement (Accuracy > 90%)")
        else:
            print(f" Model accuracy: {accuracy:.2%} (Target: >90%)")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred,
            'actuals': y_test,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance from the best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            
            if feature_names:
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                print("\n Top 10 Feature Importances:")
                print(importance_df.head(10))
                
                self.feature_importance = importance_df
                return importance_df
            else:
                return importance
        else:
            print(" Model does not support feature importance")
            return None
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        predictions = self.best_model.predict(X)
        
        # Convert to crop names if label encoder available
        if self.label_encoder:
            crop_names = self.label_encoder.inverse_transform(predictions)
            return crop_names
        else:
            return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            print(" Model does not support probability predictions")
            return None
    
    def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall, moisture=None):
        """
        Recommend crop based on soil and environmental parameters
        Returns: crop name, confidence score
        """
        # Prepare input features (7 base features from CSV)
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Make prediction
        crop_encoded = self.best_model.predict(features)[0]
        
        # Get confidence if available
        if hasattr(self.best_model, 'predict_proba'):
            probas = self.best_model.predict_proba(features)[0]
            confidence = np.max(probas)
        else:
            confidence = None
        
        # Convert to crop name
        if self.label_encoder:
            crop_name = self.label_encoder.inverse_transform([crop_encoded])[0]
        else:
            crop_name = crop_encoded
        
        return crop_name, confidence
    
    def save_model(self, filepath='crop_recommendation_model.pkl'):
        """Save the best model to disk"""
        if self.best_model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'label_encoder': self.label_encoder,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f" Model saved to {filepath}")
    
    def load_model(self, filepath='crop_recommendation_model.pkl'):
        """Load a saved model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.label_encoder = model_data.get('label_encoder')
        self.feature_importance = model_data.get('feature_importance')
        
        print(f" Model loaded from {filepath}")


if __name__ == "__main__":
    print("="*60)
    print("CROP RECOMMENDATION MODULE")
    print("="*60)
    print("\n Module ready for integration")
