import itertools
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import KFold

def encode_tags(df, min_frequency=10):

    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low 
    counts of tags to keep cardinality to a minimum.
       
    Args:
        pandas.DataFrame

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
    tags = df["tags"].tolist()
    # create a unique list of tags and then create a new column for each tag
        
    return df

def custom_cross_validation(training_data, target_column, n_splits =5):
    '''creates n_splits sets of training and validation folds

    Args:
      training_data: the dataframe of features and target to be divided into folds
      n_splits: the number of sets of folds to be created

    Returns:
      A tuple of lists, where the first index is a list of the training folds, 
      and the second the corresponding validation fold

    Example:
        >>> output = custom_cross_validation(train_df, n_splits = 10)
        >>> output[0][0] # The first training fold
        >>> output[1][0] # The first validation fold
        >>> output[0][1] # The second training fold
        >>> output[1][1] # The second validation fold... etc.
    '''
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    training_folds = []
    validation_folds = []

    for train_index, val_index in kfold.split(training_data):
        train_fold = training_data.iloc[train_index].copy()
        val_fold = training_data.iloc[val_index].copy()

        training_folds.append(train_fold)
        validation_folds.append(val_fold)
        
    return training_folds, validation_folds


def hyperparameter_search(train_folds, val_folds, pipeline, param_grid, target_column):
    '''outputs the best combination of hyperparameter settings in the param grid, 
    given the training and validation folds

    Args:
      training_folds: the list of training fold dataframes
      validation_folds: the list of validation fold dataframes
      param_grid: the dictionary of possible hyperparameter values for the chosen model

    Returns:
      A list of the best hyperparameter settings based on the chosen metric

    Example:
        >>> param_grid = {
          'max_depth': [None, 10, 20, 30],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4],
          'max_features': ['sqrt', 'log2']} # for random forest
        >>> hyperparameter_search(output[0], output[1], param_grid = param_grid) 
        # assuming 'ouput' is the output of custom_cross_validation()
        [20, 5, 2, 'log2'] # hyperparams in order
    '''
    hyperparams = list(itertools.product(*param_grid.values()))
    best_score = float('inf')
    best_params = None

    for params in hyperparams:
        param_dict = dict(zip(param_grid.keys(), params))
        pipeline.set_params(**param_dict)
        
        scores = []

        for train_fold, val_fold in zip(train_folds, val_folds):
            X_train_fold = train_fold.drop(columns=[target_column])
            y_train_fold = train_fold[target_column]
            X_val_fold = val_fold.drop(columns=[target_column])
            y_val_fold = val_fold[target_column]

            #model.fit(X_train_fold, y_train_fold)
            pipeline.fit(X_train_fold, y_train_fold)
            predictions = pipeline.predict(X_val_fold)
           
            score = np.sqrt(mean_squared_error(y_val_fold, predictions))  # RMSE
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        if avg_score < best_score:
            best_score = avg_score
            best_params = param_dict

    return best_params