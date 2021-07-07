import pandas as pd
import numpy as np
import os


def score(df, promo_pred_col = 'Promotion'):
    '''
    INPUT
    df - a dataframe with *only* the columns V1 - V7 including the promotion
    
    OUTPUT: (irr, nir)
    irr - float - incremental response rate
    nir - float - net incremental revenue
    '''
    n_treat       = df.loc[df[promo_pred_col] == 'Yes',:].shape[0]
    n_control     = df.loc[df[promo_pred_col] == 'No',:].shape[0]
    n_treat_purch = df.loc[df[promo_pred_col] == 'Yes', 'purchase'].sum()
    n_ctrl_purch  = df.loc[df[promo_pred_col] == 'No', 'purchase'].sum()
    irr = n_treat_purch / n_treat - n_ctrl_purch / n_control
    nir = 10 * n_treat_purch - 0.15 * n_treat - 10 * n_ctrl_purch
    return (irr, nir)


def test_results(promotion_strategy, tpe, model=None, test_path='./data/Test.csv'):
    '''
    INPUT
    promotion_strategy - function 
    tpe - string - showing the promotion strategy type
        ex:
            - all_purchase
            - logistic_regression
            - uplift
    model - sklearn model if available
    
    OUTPUT: 
    irr - float - incremental response rate
    nir - float - net incremental revenue
    '''    
    test_data = pd.read_csv(test_path) #'./data/Test.csv')

    df = test_data[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']]
    promos = promotion_strategy(df, tpe, model)
    score_df = test_data.iloc[np.where(promos == 'Yes')]    
    
    irr, nir = score(score_df)
    print("Nice job!  See how well your strategy worked on our test data below!")
    print()
    print('Your irr with this strategy is {:0.4f}.'.format(irr))
    print()
    print('Your nir with this strategy is {:0.2f}.'.format(nir))
    
    print("We came up with a model with an irr of {} and an nir of {} on the test set.\n\n How did you do?".format(0.0188, 189.45))
    return irr, nir
