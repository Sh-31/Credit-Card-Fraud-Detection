import os
import joblib
import argparse
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import EnsembleVoteClassifier 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV , StratifiedKFold , RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from cradit_fraud_utils_helper import load_config ,  save_model_comparison
from credit_fraud_utils_data import load_data, scale_data
from credit_fraud_utils_eval import *


def train_random_forest(X_train, y_train, X_val, y_val, random_seed, model_comparison, trainer):

    if trainer['trainer']['Random_forest']['Randomized_Search'] == True:
        param_distributions = {
            'n_estimators': [200, 400, 600 ,800],
            'min_samples_leaf': [2, 5, 10, 15],
            'min_samples_split': [5, 10, 20],
            'class_weight': [{0: 0.20, 1: 0.80}, 'balanced_subsample', {0: 0.15, 1: 0.85}],
        }

        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
        scorer = make_scorer(f1_score, pos_label=1)

        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(n_jobs=-1, bootstrap=True, random_state=random_seed),
            param_distributions=param_distributions,
            scoring=scorer,
            cv=stratified_kfold,
            n_iter=20,  
            n_jobs=-1,
            verbose=2,
            random_state=random_seed 
        )

        random_search.fit(X_train, y_train)

        parameters = random_search.best_params_
        print("Best Hyperparameters for Random Forest:", parameters)
    else:
        parameters = trainer['trainer']['Random_forest']['parameters']


    rf = RandomForestClassifier(
        **parameters,         
        random_state=random_seed  
    )

    rf.fit(X_train, y_train)    

    model_comparison , optimal_threshold = evaluate_model(rf, model_comparison, path, 'Random Forest', X_train, y_train, X_val, y_val, trainer['evaluation'])

    return {"model": rf ,  "parameters": parameters, "threshold": optimal_threshold} 


def train_logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val, random_seed, model_comparison,  trainer):

    best_params = {}

    if trainer['trainer']['Logistic_Regression']['grid_search'] == True:
             param_grid = {
                            'C':            [0.1, 1.0, 10.0],
                            'penalty':      ['l2'],
                            'class_weight': ['balanced', None, {0: 0.35, 1: 0.65}, {0: 0.25, 1: 0.75}, {0: 0.15, 1: 0.85}],
                            'solver':       ['sag', 'lbfgs', 'saga', ' newton-cg'],  
                            'max_iter':     [400, 500, 600, 800],
                        }
                        
             lr = LogisticRegression()
             scorer = make_scorer(f1_score, pos_label=1)

             stratified_kfold = StratifiedKFold(n_splits=5, 
                                            shuffle=True,
                                            random_state=42)

             grid_search = GridSearchCV(lr, 
                                    param_grid,cv=stratified_kfold, 
                                    scoring=scorer, 
                                    n_jobs=-1)

             grid_search.fit(X_train_scaled, y_train)

             best_params = grid_search.best_params_
             print("Best Hyperparameters:", best_params)

  
    else:
        best_params = trainer['trainer']['Logistic_Regression']['parameters']
       

    lr = LogisticRegression(**best_params, random_state=random_seed)
    lr.fit(X_train_scaled, y_train)

    model_comparison , optimal_threshold = evaluate_model(lr, model_comparison, path, 'Logistic Regression', X_train_scaled, y_train, X_val_scaled, y_val, trainer['evaluation'])

    return {"model": lr , "parameters": best_params, "threshold": optimal_threshold}
    


def train_neural_network(X_train_scaled, y_train, X_val_scaled, y_val, random_seed, model_comparison,  trainer):

    best_params = {}

    if trainer['trainer']['Neural_Network']['Randomized_Search'] == True:
        param_dist = {
        'activation': ['relu'],
        'hidden_layer_sizes': [
            (30, 20), 
            (30, 20, 10), 
            (40, 30, 20), 
            (64, 32, 16),
            (64, 32, 32, 16)
        ],
        'solver': ['adam', 'sgd'],
        'batch_size': [64, 128, 512],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'alpha': [0.001, 0.01, 0.025],
        'max_iter': [500, 800, 1000, 2000],
        'random_state': [random_seed]
        }

        MLP_CV = MLPClassifier()
        scorer = make_scorer(f1_score, pos_label=1)  
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
        random_search = RandomizedSearchCV(MLP_CV, param_distributions=param_dist, n_iter=30, cv=cv, scoring=scorer, n_jobs=-1, random_state=random_seed)
        random_search.fit(X_train_scaled, y_train)

        MLP = MLPClassifier(**best_params)

    else:
        # load parameters from config file
        parameters = trainer['trainer']['Neural_Network']['parameters']
    
        MLP = MLPClassifier(
            hidden_layer_sizes=eval(parameters['hidden_layer_sizes']), # eval to convert string to tuple
            activation=parameters['activation'],
            solver=parameters['solver'],
            alpha=parameters['alpha'],
            batch_size=parameters['batch_size'],
            learning_rate_init=parameters['learning_rate_init'],
            max_iter=parameters['max_iter'],
            random_state=random_seed
        )

    MLP.fit(X_train_scaled, y_train)

    model_comparison, optimal_threshold = evaluate_model(MLP, model_comparison, path, 'Neural Network', X_train_scaled, y_train, X_val_scaled, y_val, trainer['evaluation'])

    return {"model": MLP ,  "parameters": parameters, "threshold": optimal_threshold}

def train_voting_classifier(X_train, y_train, x_val, y_val, models, random_seed, model_comparison, trainer):
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # make scaler learn statistics from training data
    param = trainer['trainer']['Voting_Classifier']['parameters']

    voting_classifier = EnsembleVoteClassifier(
        clfs=[
            make_pipeline(scaler, models['Logistic_Regression']['model']),
            make_pipeline(scaler, models['Neural_Network']['model']),
            models['Random_forest']['model'],
        ],
        weights=param['weights'],
        fit_base_estimators=param['fit_base_estimators'],
        use_clones=param['use_clones'],
        voting=param['voting'],
    )

    voting_classifier.fit(X_train, y_train) #  no refiting required here

    model_comparison, optimal_threshold = evaluate_model(voting_classifier, model_comparison, path, 'Voting Classifier', X_train, y_train, x_val, y_val, trainer['evaluation'])

    return {"model": voting_classifier , "parameters": param, "threshold": optimal_threshold}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  help="path to the dataset and preprocessing config file", default="config.yml")
    parser.add_argument("--trainer", help="path to trainer and evaluation config file", default="trainer_config.yml")
    args = parser.parse_args()
   
    config_folder_path = "config/" 
    config =  load_config(config_folder_path + args.config)
    trainer = load_config(config_folder_path + args.trainer)

    RANDOM_SEED = config['random_seed']
    np.random.seed(RANDOM_SEED)
    
    X_train, y_train, X_val, y_val = load_data(config)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val, config['preprocessing']['scaler_type'])

    model_comparison = {} # model comparison stats dictionary
    models = {} # trained models dictionary

    # results path for trained models and evaluation figures
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    path = 'models/{}/'.format(now)
    if not os.path.exists(path):
        os.makedirs(path)

    if trainer['trainer']['Random_forest']['train']:
       models["Random_forest"] = train_random_forest(X_train, y_train, X_val, y_val, RANDOM_SEED , model_comparison, trainer) # Random forest does not need scaled data

    if trainer['trainer']['Logistic_Regression']['train']:
        models['Logistic_Regression'] = train_logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val, RANDOM_SEED, model_comparison, trainer)

    if trainer['trainer']['Neural_Network']['train']:
        models['Neural_Network'] = train_neural_network(X_train_scaled, y_train, X_val_scaled, y_val, RANDOM_SEED, model_comparison, trainer)    

    if trainer['trainer']['Voting_Classifier']['train']:
        models['Voting_Classifier'] = train_voting_classifier(X_train, y_train, X_val, y_val, models, RANDOM_SEED, model_comparison, trainer)

    # Save the models
    model_path = path + "trained_models.pkl"
    joblib.dump(models, model_path)
    print('Model saved at: {}'.format(model_path))
    print('Evaluation plots saved at: {}evaluation/plot'.format(path))

    # Save the model comparison
    model_comparison_path = path + "model_comparison-(validation dataset).png"
    save_model_comparison(model_comparison, model_comparison_path)

    print('\nModels comparison:\n')
    print(pd.DataFrame(model_comparison).T.to_markdown())