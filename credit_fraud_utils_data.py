import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.preprocessing import MinMaxScaler, StandardScaler , RobustScaler

def scale_data(train_data, val_data=None, scaler_type='robust'):
    """
    Scale the input training and validation data separately using either MinMaxScaler or StandardScaler.

    Parameters:
    - train_data (numpy array or pandas DataFrame): The training data to be scaled.
    - val_data (numpy array or pandas DataFrame, optional): The validation data to be scaled. Default is None.
    - scaler_type (str, optional): The type of scaler to be used. 
      Options: 'minmax' for MinMaxScaler, 'standard' for StandardScaler.
      Default is 'minmax'.

    Returns:
    - scaled_train_data (numpy array or pandas DataFrame): The scaled training data.
    - scaled_val_data (numpy array or pandas DataFrame): The scaled validation data, if provided. None otherwise.
    """

    # Check if the data is a pandas DataFrame, convert it to a numpy array
    if isinstance(train_data, pd.DataFrame):
        train_data = train_data.values

    if val_data is not None:
        if isinstance(val_data, pd.DataFrame):
            val_data = val_data.values

    # Select the scaler based on the specified type
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaler type. Please choose 'minmax' or 'standard' or 'robust'.")

    # Fit and transform the training data using the chosen scaler
    scaled_train_data = scaler.fit_transform(train_data)

    # Transform the validation data if provided
    scaled_val_data = None
    if val_data is not None:
        scaled_val_data = scaler.transform(val_data)

    return scaled_train_data, scaled_val_data

def load_data(config):
    train_data = pd.read_csv(config['dataset']['train']['path'])
    val_data = pd.read_csv(config['dataset']['val']['path'])

    X_train = train_data.drop(config['dataset']['target'], axis=1)
    y_train = train_data[config['dataset']['target']]

    X_val = val_data.drop(config['dataset']['target'], axis=1)
    y_val = val_data[config['dataset']['target']]

    return X_train, y_train, X_val, y_val


def load_test(config):
    test_data = pd.read_csv(config['dataset']['test']['path'])
    X_test = test_data.drop(config['dataset']['target'], axis=1)
    y_test = test_data[config['dataset']['target']]
    return X_test, y_test


def balance_data_transformation(X_train, y_train, balance_type='smote',sampling_strategy='auto',k=5,random_state=None):
    """
    Balance the input training data using the specified balancing strategy.

    Parameters:
    - X_train (numpy array or pandas DataFrame): The training data features.
    - y_train (numpy array or pandas DataFrame): The training data labels.
    - balance_type (str, optional): The type of balancing strategy to be used.
      Options:
      - 'under_sampling': for random under-sampling
      - 'over_sampling': for random over-sampling
      - 'smote': for SMOTE
      - 'SMOTEENN': for SMOTEENN combination
      - 'SMOTETomek': for SMOTETomek combination
      Default is 'smote'.
    - random_state (int, optional): The random state for reproducibility. Default is None.

    Returns:
    - X_resampled (numpy array or pandas DataFrame): The balanced training data features.
    - y_resampled (numpy array or pandas DataFrame): The balanced training data labels.
    """
    
 
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.values.ravel() 
    
    print("Dataset before balancing:")
    print(f"Number of Non-fraud transactions: {len(y_train[y_train == 0])}")
    print(f"Number of fraud transactions:     {len(y_train[y_train == 1])}")

    if balance_type == 'under':
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    elif balance_type == 'over':
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    elif balance_type == 'smote':
        sampler = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=k)
    elif balance_type == 'SMOTEENN':
        sampler = SMOTEENN(
                          random_state=random_state, 
                          sampling_strategy=sampling_strategy,
                          smote=SMOTE(random_state=random_state,
                                      sampling_strategy=sampling_strategy,
                                      k_neighbors=k),  
                         )
    elif balance_type == 'SMOTETomek':
        sampler = SMOTETomek(
                             random_state=random_state, 
                             sampling_strategy=sampling_strategy,
                             smote=SMOTE(random_state=random_state,
                                         sampling_strategy=sampling_strategy,
                                         k_neighbors=k),
                             n_jobs=-1
                        )
    else:
        raise ValueError("Invalid balance type. Please choose 'under_sampling', 'over_sampling', 'smote', 'SMOTEENN', or 'SMOTETomek'.")
    
    # Fit and resample the training data using the chosen sampler
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    print("\n Dataset after balancing:")
    print(f"Number of Non-fraud transactions: {len(y_resampled[y_resampled == 0])}")
    print(f"Number of fraud transactions:     {len(y_resampled[y_resampled == 1])}")

    return X_resampled, y_resampled
