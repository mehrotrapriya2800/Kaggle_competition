# Kaggle_competition


### Description of the task:

This competition was about modelling the performance of computer programs. 
The dataset provided describes a few examples of running SGDClassifier in Python. 
The features of the dataset describe the SGDClassifier as well as the features used to generate the synthetic training data. 
The data to be analysed or predicted is the training time of the SGDClassifier.

- I have used Python programming language for implementation.
- I have done the following imports. (packages used)
  
        import pandas as pd

        import tensorflow as tf

- My source code is a .py file named kaggle_code_github.py which can run directly.
- The code uses train.csv and test.csv as input files.
- I have used DNNLinearCombinedRegressor() model for prediction with following hidden layers. (Link: https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedRegressor)

        model = tf.estimator.DNNLinearCombinedRegressor(\

        linear_feature_columns =wide_columns,\

        dnn_feature_columns=Feature_columns,\

        dnn_hidden_units=[1000, 500, 250, 125, 75, 25, 14],\

        )

- It produces the prediction file named output_github_submit.csv
- I have applied a lot of feature engineering by modifying some of the columns and making new columns from existing ones by studying the parameters of SGDClassifier() (link: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) ).
- Predictions are obtained as follows:

        model.train(input_fn=input_train)
        predict_input_fn = tf.estimator.inputs.pandas_input_fn(\

                x=test,\

               batch_size=1,\

                num_epochs=1,\

                shuffle=False)
        predictions = model.predict(input_fn=predict_input_fn)
