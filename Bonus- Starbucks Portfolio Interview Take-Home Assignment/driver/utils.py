# testing baseline model, where we send promotions to everyone

import numpy as np
import pandas as pd

def promotion_strategy(df, tpe, model=None):
        '''
        INPUT 
        df - a dataframe with *only* the columns V1 - V7 (same as train_data)
        tpe - a string showing the promotion strategy type
            ex:
                - all_purchase
                - logistic_regression
                - uplift
        model - sklearn model 
            ex of model types: 
                - sklearn.linear_model._logistic.LogisticRegression
                - xgboost.sklearn.XGBClassifier
        
        OUTPUT
        promotion_df - np.array with the values
                       'Yes' or 'No' related to whether or not an 
                       individual should recieve a promotion 
                       should be the length of df.shape[0]            
        Ex:
        INPUT: df

        V1	V2	  V3	V4	V5	V6	V7
        2	30	-1.1	1	1	3	2
        3	32	-0.6	2	3	2	2
        2	30	0.13	1	1	4	2
        

        OUTPUT: promotion

        array(['Yes', 'Yes', 'No'])
        indicating the first two users would recieve the promotion and 
        the last should not.
        '''
        
        #testing baseline model, where we send promotions to everyone
        if tpe == 'all_purchase':
            test = df
            promotion = []
            num_test_points = test.shape[0]

            for i in range(num_test_points):
                promotion.append('Yes')

            promotion = np.array(promotion)
            return promotion
        
        #testing logistic regression model
        elif tpe == 'logistic_regression':
            # transform categorical variables
            df_train = pd.get_dummies(data=df, columns=['V1', 'V4', 'V5','V6','V7'])

            df_train
            # predict
            preds = model.predict(df_train) # log_mod

            # transform 0/1 array to No/Yes 
            my_map = {0: "No", 1: "Yes"}
            promotion = np.vectorize(my_map.get)(preds)

            return promotion
        
        # testing the uplift model
        elif tpe == 'uplift':
            test = df
            
            # predict
            preds = model.predict(test, ntree_limit=model.best_ntree_limit)

            # transform 0/1 array to No/Yes
            my_map = {0: "No", 1: "Yes"}
            promotion = np.vectorize(my_map.get)(preds)
#             promotion = []
#             for pred in preds:
#                 if pred == 1:
#                     promotion.append('Yes')
#                 else:
#                     promotion.append('No')
#             promotion = np.array(promotion)
            return promotion
        else:
            return "promotion strategy type was not chosen"
        