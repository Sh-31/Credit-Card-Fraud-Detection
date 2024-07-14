import torch
import torch.nn as nn
import torch.nn.functional as F
from credit_fraud_utils_data import load_data, scale_data , balance_data_transformation
from cradit_fraud_utils_helper import *
from credit_fraud_utils_eval import *
from torch.utils.tensorboard import SummaryWriter  

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        # focal loss : https://leimao.github.io/blog/Focal-Loss-Explained/
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_logits, target):
        BCELoss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        prob = pred_logits.sigmoid()
        alpha_t = torch.where(target == 1, self.alpha, (1 - self.alpha))
        pt =  torch.where(target == 1, prob, 1 - prob)
        loss = alpha_t * ((1 - pt) ** self.gamma) * BCELoss
        return loss.sum()

class FraudDetectionNN(nn.Module):
    def __init__(self):
        super(FraudDetectionNN, self).__init__()
        self.hidden1 = nn.Linear(30, 32, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.hidden2 = nn.Linear(32, 32, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.hidden3 = nn.Linear(32, 16, bias=False)
        self.bn3 = nn.BatchNorm1d(16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.relu(self.bn1(self.hidden1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.hidden2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.hidden3(x)))
        x = self.dropout(x)
        x = self.output(x)
        return x

if __name__ == '__main__':

    config = load_config("config/config.yml")
    torch.manual_seed(config['random_seed']) 

    X_train, y_train, X_val, y_val = load_data(config)
    X_train, X_val = scale_data(X_train, X_val, scaler_type='robust')

    # print("Nig: ", len(y_train[y_train == 0])," Pos: ",len(y_train[y_train == 1]))
    # # balance data
    # X_train, y_train = balance_data_transformation(X_train, y_train, balance_type='over',sampling_strategy={0: len(y_train[y_train == 0]), 
    #                                                                                                         1:  3500},
                                                                                                           
    #                                                                                                                 random_state=config['random_seed'])
    # print("Nig: ",len(y_train[y_train == 0])," Pos: ",len(y_train[y_train == 1]))

   
    model = FraudDetectionNN()
    alpha = 0.95 # (rate to make balance class)                # 0.75 - 0.80 - 0.85 - 0.90 - 0.95 - 0.99
    gamma = 2.0  # (focusing on hard samples "minority class") # 0 - 1 - 1.5 - 2 - 4 - 6 - 8
    lr = 0.0004

    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    # optimizer =  torch.optim.Adam(model.parameters(), lr=lr) # try stochastic gradient descent after 100 ecpochs 
    optimizer =  torch.optim.SGD(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train ,dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)


    batch_size = 1024 # 1024 2048 4096
    num_epochs = 1000
    start_epoch = 0


    run_name = f"gamma_{gamma}_alpha_{alpha}_batch_size_{batch_size}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}_Adam_optimizer")


    # Uncomment the line below to load from a checkpoint  
    # checkpoint_id = 500
    # path = f'models/focal_loss_checkpoints/checkpoint_epoch_{checkpoint_id}.pth'
    # start_epoch = load_checkpoint(model, path) + 1
    

    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        
        # shuffle training data
        permutation = torch.randperm(X_train_tensor.size()[0])
        X_train_tensor_shuffled = X_train_tensor[permutation].clone()
        y_train_tensor_shuffled = y_train_tensor[permutation].clone()

        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor_shuffled[i:i+batch_size]
            y_batch = y_train_tensor_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # log epoch statistics
        epoch_loss /= len(X_train_tensor) / batch_size
        writer.add_scalar('Loss/train', epoch_loss, epoch)  

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'{name}.grad', param.grad, epoch)

        print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, num_epochs, epoch_loss))


        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, y_val_tensor).item()
            writer.add_scalar('Loss/validation', val_loss, epoch)


        # Checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, epoch + 1)

            model.eval()
            with torch.no_grad():
                val_output = model(X_val_tensor)
                val_loss = criterion(val_output, y_val_tensor).item()
           
                y_val_prob = val_output.sigmoid().numpy()
                y_val_pred = (y_val_prob > 0.5).astype(int)
                
                report_val = classification_report(y_true=y_val, y_pred=y_val_pred, output_dict=True)
                # auc_pr = eval_auc_precision_recall_curve(y_pred=y_val_pred, y_true=y_val)
                writer.add_scalar('Validation/Precision',report_val["1"]["precision"], epoch + 1)
                writer.add_scalar('Validation/Recall', report_val["1"]["recall"], epoch + 1)
                writer.add_scalar('Validation/F1', report_val["1"]["f1-score"], epoch + 1) 
                # writer.add_scalar('Validation/AUC', auc_pr, epoch + 1)  

    model.eval()
    with torch.no_grad():
        val_output = model(X_train_tensor)
        y_train_prob = val_output.sigmoid().numpy()
        y_train_pred = (y_train_prob > 0.5).astype(int)
        _ = eval_classification_report_confusion_matrix(y_true=y_train, y_pred=y_train_pred, title='Neural Network-Focal train') 
        # eval_precision_recall_for_different_threshold(y_pred=y_train_prob, y_true=y_train)

        val_output = model(X_val_tensor)
        y_val_prob = val_output.sigmoid().numpy()
        y_val_pred = (y_val_prob > 0.50).astype(int)
        report_val = eval_classification_report_confusion_matrix(y_true=y_val, y_pred=y_val_pred, title='Neural Network-Focal valdtion')

        
        optimal_threshold, f1_scores = eval_best_threshold(y_pred=y_train_prob, y_true=y_train, with_repect_to="f1_score")  
        y_val_pred = (y_val_prob > optimal_threshold).astype(int)
        report_val = eval_classification_report_confusion_matrix(y_pred=y_val_pred, y_true=y_val, title='Neural Network-Focal optimal threshold')

        writer.close()  # tensorboard --logdir=runs

######################################################################################################################################
# Some learning lessons & Notes:
# 1. Alpth and gamma sometimes unstables train using batchnorm make this effect less occur and switching from Adam to SGD also.     
# 2. High gamma (5~7) gives very noisey loss Curve 
# 3. Alpha is very crucial to balance the two classes (need hyperparameter tuning).
# 4. I tried Neural network with three layer and four layer but the results saturated in (0.80 recall, 0.80 precision for positive class)
#   So tring more comple Neural network with more hidden layer may helps.
#####################################################################################################################################
