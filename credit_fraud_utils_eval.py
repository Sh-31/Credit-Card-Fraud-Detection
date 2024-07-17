import os
import seaborn as sns
from numpy import argmax
from sklearn.metrics import classification_report , precision_recall_curve ,confusion_matrix , auc
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 100

def eval_predict_with_threshold(model , x , threshold=0.5):
    """
    Predict with specified threshold

    parmeters:
    model: trained model
    x: features
    threshold: threshold for prediction

    return: predicted values
    """
    y_pred_proba = model.predict_proba(x)
    y_pred = (y_pred_proba[:,1] >= threshold).astype('int') # probability of positive class
    return y_pred


def eval_confusion_matrix(y_pred,y_true, title="" , save_png=False,  path=""): 
    labels = ['True Negative', 'False Positive' , 'False Negative', 'True Positive']
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm_flat = cm.flatten()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    for i, txt in enumerate(cm_flat):
        plt.text(i % 2 + 0.5, i // 2 + 0.5, f"{labels[i]}\n{txt}", ha='center', va='center', color='black')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    if save_png:
        plt.savefig(f'{path}/{title} Confusion Matrix.png') 
    else:
        plt.show()
        
     


def eval_auc_precision_recall_curve(y_pred_prob, y_true):
     """
     Get Area under curve of precision recal of precision recall curve

     Uasge:
     Auc of precision recall curve give good indicator of over all model peformance.
     """
     precision, recall, _ = precision_recall_curve(y_score=y_pred_prob,y_true=y_true)
     
     return float(auc(x=recall, y=precision))


def eval_precision_recall_for_different_threshold(y_pred_prob, y_true, title="" ,save_png=False, path=""):

    precision, recall, thresholds = precision_recall_curve(y_score=y_pred_prob,y_true=y_true)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], label='Precision', marker='.')
    plt.plot(thresholds, recall[:-1], label='Recall', marker='.')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall for different Thresholds')
    plt.legend()

    if save_png:
        plt.savefig(f'{path}/{title} precision recall for different threshold.png')
    else:    
        plt.show()

def eval_classification_report_confusion_matrix(y_pred, y_true, title="" ,save_png=False, path="", digits=5 ):

    print(f'{title} Classification Report')
    print(classification_report(y_pred=y_pred, y_true=y_true, digits=digits))   
    report_stats = classification_report(y_pred=y_pred, y_true=y_true, digits=digits, output_dict=True)

    labels = ['True Negative', 'False Positive' , 'False Negative', 'True Positive'] # order of confusion matrix labels
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm_flat = cm.flatten()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    for i, txt in enumerate(cm_flat):
        plt.text(i % 2 + 0.5, i // 2 + 0.5, f"{labels[i]}\n{txt}", ha='center', va='center', color='black')
    plt.title(f'Confusion Matrix of {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    if save_png: 
        plt.savefig(f'{path}/{title} Confusion Matrix.png')
    else: 
        plt.show()

    return report_stats


def eval_precision_recall_curve(y_pred_prob,y_true, title="", save_png=False, path=""):
    '''
    Plot precision recall curve
    '''

    precision, recall, thresholds = precision_recall_curve(y_score=y_pred_prob,y_true=y_true)
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title} Precision-Recall Curve')
    plt.legend()
    if save_png:
        plt.savefig(f'{path}/{title} precision recall area under curve.png') 
    else:    
        plt.show()   

def eval_best_threshold(y_pred,y_true , with_repect_to="f1_score"): 
    """
    Get best threshold from precision recall curve with respect to f1_score, precision or recall

    parameters:
    y_pred: predicted values
    y_true: true values
    with_repect_to: "f1_score" , "precision" or "recall"

    returns:
    optimal threshold and f1 scores
    """
    precision, recall, thresholds = precision_recall_curve(y_score=y_pred,y_true=y_true)
    f1_scores = ((2 * precision * recall) / (precision + recall))

    if with_repect_to == "f1_score":
        optimal_threshold_index = argmax(f1_scores)
    elif with_repect_to == "precision":
        optimal_threshold_index = argmax(precision)
    elif with_repect_to == "recall":
        optimal_threshold_index = argmax(recall)
    else:
        raise ValueError("Invalid value for with_repect_to. Please choose 'f1_score', 'precision' or 'recall'.")        

    optimal_threshold = thresholds[optimal_threshold_index]
    print("Optimal Threshold:", optimal_threshold , "F1 Score:", f1_scores[optimal_threshold_index])
    return optimal_threshold , f1_scores


def eval_update_model_stats(model_comparison , model_name, report_val, metric_config={}):
    """
    Update model_comparison dictionary with evaluation metrics of validation set
    parameters:
    model_comparison: model_comparison dictionary
    model_name: model name
    report_val: evaluation metrics of validation set
    """
    if len(metric_config.keys()) == 0:
        model_comparison[model_name] = {
            "F1 Score Positive class": report_val['1']['f1-score'],
            "F1 Score Negative class": report_val['0']['f1-score'],
            "Precision Positive class": report_val['1']['precision'],
            "Recall Positive class": report_val['1']['recall'],   
            "F1 Score Average": report_val['macro avg']['f1-score'],
            }
    else:
       model_comparison[model_name] = {}
       for key, value in metric_config['pos'].items():
            if value:
                 model_comparison[model_name][f"{key} positive class"] = report_val['1'][key]

       for key, value in metric_config['neg'].items():
            if value:
                  model_comparison[model_name][f"{key} negative class"] = report_val['0'][key]

       if metric_config['macro_avg']: 
           model_comparison[model_name]['F1 macro avg'] = report_val['macro avg']['f1-score']
                   
     
    return model_comparison


def evaluate_model(model, model_comparison, path, title, X_train, y_train, x_val, y_val, evaluation_config):

    optimal_threshold = 0.5 # default threshold

    eval_plots_path = path + evaluation_config['plot_path']
    if not os.path.exists(eval_plots_path):
        os.makedirs(eval_plots_path)

    save_cm_plots = evaluation_config['confusion_matrix']
    save_pr_rc_th_plots = evaluation_config['precision_recall_threshold']
    save_roc_curve = evaluation_config['roc_curve']
    if evaluation_config['train'] == True:

        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:,1]

        eval_classification_report_confusion_matrix(y_train, y_train_pred, title + ' train', save_png=save_cm_plots, path=eval_plots_path)
        eval_precision_recall_for_different_threshold(y_pred_prob=y_train_pred_proba, y_true=y_train,title=title+' train',save_png=save_pr_rc_th_plots, path=eval_plots_path)
        eval_precision_recall_curve(y_pred_prob=y_train_pred_proba, y_true=y_train, title=title + ' train', save_png=save_roc_curve, path=eval_plots_path)

    if evaluation_config['validation'] == True:

        y_val_pred = model.predict(x_val)
        y_val_pred_proba = model.predict_proba(x_val)[:,1]
        report_val = eval_classification_report_confusion_matrix(y_val, y_val_pred, title + ' validation', save_png=save_cm_plots, path=eval_plots_path)
        eval_precision_recall_curve(y_pred_prob=y_val_pred_proba, y_true=y_val, title=title + ' validation', save_png=save_roc_curve, path=eval_plots_path)
        model_comparison = eval_update_model_stats(model_comparison, title,  report_val, evaluation_config['metric'])


        if evaluation_config['metric']['PR_AUC'] == True:
            y_val_pred_proba = model.predict_proba(x_val)[:,1]
            model_comparison[title]['PR AUC'] = eval_auc_precision_recall_curve(y_pred_prob=y_val_pred_proba, y_true=y_val)

        if evaluation_config['optimal_threshold'] == True: # we use only training data to find optimal threshold

            y_train_pred_proba = model.predict_proba(X_train)[:,1] 
            optimal_threshold , f1_scores = eval_best_threshold(y_pred=y_train_pred_proba,y_true=y_train)

            y_val_pred = eval_predict_with_threshold(model=model, x=x_val, threshold=optimal_threshold)
            report_val = eval_classification_report_confusion_matrix(y_pred=y_val_pred,y_true=y_val, title= title + ' val with optimal threshold', save_png=save_cm_plots, path=eval_plots_path)
            model_comparison = eval_update_model_stats(model_comparison, title + ' optimal threshold',  report_val ,  evaluation_config['metric'])
            
            if evaluation_config['metric']['PR_AUC'] == True: # It will same as validation (0.5 threshold)
                y_val_pred_proba = model.predict_proba(x_val)[:,1]
                model_comparison[title + ' optimal threshold']['PR AUC'] = eval_auc_precision_recall_curve(y_pred_prob=y_val_pred_proba, y_true=y_val)

    return model_comparison , optimal_threshold