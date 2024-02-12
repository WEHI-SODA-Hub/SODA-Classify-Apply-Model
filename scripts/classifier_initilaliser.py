import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from skopt.space import Real, Integer
import numpy as np 
from typing import Dict
from preprocess import data_balancer

class ClassifierInitialiser:

    def __init__(self) -> None:
        self.optmiser = None
        self.valid_scheme = {"Xgboost" : XgboostApplier}
        self.bayes_cv_tuner = None

    def tune_hyper_parameter(self, X : pd.DataFrame, y : pd.DataFrame, 
                        classifier_scheme : str, options : Dict) -> None:
        """
        Fits training data to the bayes cv hyper paramter tuning
        """
        if classifier_scheme not in self.valid_scheme:
            #TODO: raise a warning or error
            return
        try:
            model_options = options["CLASSIFIER_OPTIONS"]
            balance_scheme = options["BALANCE_SCHEME"]
            bayescv_options = options["BAYESCV_OPTIONS"]

            split_random_state = options["SPLIT_RANDOM_STATE"]
            test_size = options["TEST_SIZE"]

            n_jobs = bayescv_options["N_JOBS"]
            n_iter = bayescv_options["ITERATIONS"]
            scoring = bayescv_options["SCORING"]
        except:
            # TODO: raise error
            pass

        self.X = X
        self.y = y

        self.X_train_orig, self.X_test, self.y_train_orig, self.y_test = train_test_split(
                                                    X, y, 
                                                    test_size=test_size, 
                                                    random_state=split_random_state, 
                                                    shuffle=True, 
                                                    stratify=y)

        balancer = data_balancer.DataBalancer()
        self.X_train, self.y_train = balancer.balance_data(self.X_train_orig,
                                                        self.y_train_orig, 
                                                        balance_scheme)
        
        model_applier = self.valid_scheme[classifier_scheme](model_options)
        self.bayes_cv_tuner = BayesSearchCV(
            estimator = model_applier.model,
            search_spaces = model_applier.search_space,
            scoring = scoring,
            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=np.random.RandomState(0)
            ),
            n_jobs = n_jobs,
            n_iter = n_iter,   
            verbose = 2,
            refit = True,
            random_state=np.random.RandomState(42),
            fit_params={
                'sample_weights' : compute_sample_weight(
                                            class_weight='balanced', 
                                            y=self.y_train)
                }
        )

        self.bayes_cv_tuner.fit(self.X_train, self.y_train)

    def get_prediction_df(self) -> pd.DataFrame:
        """
        Returns a dafaframe with the first column being the actual predictions
        and the following columns the probability for being each of the classes
        """
        if self.bayes_cv_tuner == None:
            # TODO: Error message
            return

        y_pred_df = pd.DataFrame(self.bayes_cv_tuner.predict(self.X_test))
        y_proba_df = pd.DataFrame(self.bayes_cv_tuner.predict_proba(self.X_test))

        return pd.concat([y_pred_df, y_proba_df], axis=1)
    
    def get_final_classifier(self):
        """
        Fits the entire data to the found hyper-paramters
        """
        if self.bayes_cv_tuner == None:
            # TODO: Error message
            return
        
        return self.bayes_cv_tuner.best_estimator_.fit(
                                        self.X, self.y, 
                                        sample_weight=compute_sample_weight(
                                            class_weight='balanced', 
                                            y=self.y
                                        ))

class XgboostApplier:
    
    def __init__(self, args : Dict) -> None:
        
        self.search_space = {
                        'eta' : Real(1e-8, 1, 'log-uniform'),
                        'reg_alpha' : Real(1e-8, 1.0, 'log-uniform'),
                        'reg_lambda': Real(1e-8, 1000, 'log-uniform'),
                        'max_depth': Integer(0, 50, 'uniform'),
                        'n_estimators': Integer(10, 300, 'uniform'), 
                        'learning_rate': Real(1e-8, 1.0, 'log-uniform'),
                        'min_child_weight': Integer(0, 10, 'uniform'),
                        'max_delta_step': Integer(1, 100, 'uniform'),
                        'subsample': Real(1e-8, 1.0, 'uniform'),
                        'colsample_bytree': Real(1e-8, 1.0, 'uniform'),
                        'colsample_bylevel': (1e-8, 1.0, 'uniform'),
                        'gamma': Real(1e-8, 1.0, 'log-uniform'),
                        'min_child_weight': Integer(0, 5, 'uniform')
                    }
        try:
            self.model = xgb.XGBClassifier(
                    n_jobs = args["N_JOBS_MODEL"],
                    objective = args["OBJECTIVE_FUNC"],
                    tree_method='hist'
                )
        except:
            # TODO: Raise an error
            pass