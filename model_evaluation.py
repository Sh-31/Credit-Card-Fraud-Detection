import torch
from credit_fraud_utils_eval import *
from cradit_fraud_utils_helper import load_config , load_model, load_checkpoint
from credit_fraud_utils_data import load_test,load_data, scale_data
from sklearn.preprocessing import RobustScaler
from focal_loss import FraudDetectionNN

def evaluate_FraudDetectionNN_focal_loss(config):
    checkpoint_id = 400
    checkpoint_path = f'models/focal_loss_checkpoints/checkpoint_epoch_{checkpoint_id}.pth' # Neural Network with Focal-Loss
    FraudDetectionNN_model = FraudDetectionNN()

    load_checkpoint(FraudDetectionNN_model, checkpoint_path)

    X_train, _ ,  _ , _ = load_data(config)
    X_test, y_test = load_test(config)
  
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_test_tensor = torch.tensor(X_test ,dtype=torch.float32)

    FraudDetectionNN_model.eval()
    X_test_pred = FraudDetectionNN_model(X_test_tensor)
    X_test_pred_prob = X_test_pred.sigmoid().detach().numpy()
    X_test_pred = (X_test_pred_prob > 0.5).astype(int)
 
    eval_classification_report_confusion_matrix(y_true=y_test, y_pred=X_test_pred, title='FraudDetectionNN  test') 
    print(f"Precision Recall - AUC: {eval_auc_precision_recall_curve(y_pred_prob=X_test_pred_prob, y_true=y_test)}")


def evaluate_VotingClassifier(config, model_path):
    model_path = 'models/2024_07_18_07_29/trained_models.pkl' # path to voting classifier model
    model = load_model(model_path) # load model

    X_test, y_test = load_test(config)
   
    x_test_pred = model['Voting_Classifier']['model'].predict(X_test)
    x_test_pred_prob = model['Voting_Classifier']['model'].predict_proba(X_test)[:, 1]

    eval_classification_report_confusion_matrix(y_pred=x_test_pred, y_true=y_test, title="Voting Classifier test")
    print(f"Precision Recall - AUC: {eval_auc_precision_recall_curve(y_pred_prob=x_test_pred_prob, y_true=y_test)}")


if __name__ == "__main__":
    config_path = 'config/config.yml' # path to config
    model_path = 'models/2024_07_18_07_29/trained_models.pkl' # path to voting classifier model
    config = load_config(config_path)

    evaluate_VotingClassifier(config=config, model_path=model_path) # evaluate voting classifier
    evaluate_FraudDetectionNN_focal_loss(config) # evaluate neural network with focal loss