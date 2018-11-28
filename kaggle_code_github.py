import pandas as pd
import tensorflow as tf
                        
data_train_final = pd.read_csv("train.csv")                                
test_data = pd.read_csv("test.csv")

#applying feature engineering to both training and test dataset

test_data['new_col1'] = (test_data['n_classes'] * test_data['n_clusters_per_class'])/test_data['n_jobs']
test_data['new_col3'] = test_data['new_col1']/test_data['n_informative']
test_data.loc[test_data['n_jobs'] == -1, 'n_jobs'] = 16
test_data['new_col2'] = (test_data['max_iter'] * test_data['n_samples'])/test_data['n_jobs']
data_train_final['new_col1'] = (data_train_final['n_classes'] * data_train_final['n_clusters_per_class'])/data_train_final['n_jobs']
data_train_final['new_col3'] = data_train_final['new_col1']/data_train_final['n_informative']
data_train_final.loc[data_train_final['n_jobs'] == -1, 'n_jobs'] = 16
data_train_final['new_col2'] = (data_train_final['max_iter'] * data_train_final['n_samples'])/data_train_final['n_jobs']

lab = data_train_final["time"]


data_train_final = data_train_final.drop(['time'], axis=1)
pen = data_train_final["penalty"]
pen1 = test_data["penalty"]


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
new_df_train = data_train_final.select_dtypes(include=numerics)
new_df_test = test_data.select_dtypes(include=numerics)

train = (new_df_train - new_df_test.mean())/new_df_test.std(ddof=0)
train = train.join(pen) #joining penalty column
test = (new_df_test - new_df_test.mean())/new_df_test.std(ddof=0)
test = test.join(pen1) #joining penalty column

#dropping some columns

train = train.drop(columns=['l1_ratio','scale','random_state','alpha','flip_y'])
test = test.drop(columns=['l1_ratio','scale','random_state','alpha','flip_y'])

#deciding the BATCH_SIZE and num_epochs

BATCH_SIZE = 140
num_epochs = 1000


#converting train and test data to tensors
input_train = tf.estimator.inputs.pandas_input_fn(x=train,y=lab,batch_size=BATCH_SIZE,num_epochs=num_epochs,shuffle=True)


#converting our columns to tensorflow columns
max_iter = tf.feature_column.numeric_column("max_iter")
n_jobs = tf.feature_column.numeric_column("n_jobs")
n_samples = tf.feature_column.numeric_column("n_samples")
n_features = tf.feature_column.numeric_column("n_features")
n_classes = tf.feature_column.numeric_column("n_classes")
n_clusters_per_class = tf.feature_column.numeric_column("n_clusters_per_class")
n_informative = tf.feature_column.numeric_column("n_informative")
new_col1 = tf.feature_column.numeric_column("new_col1")
new_col2 = tf.feature_column.numeric_column("new_col2")
new_col3 = tf.feature_column.numeric_column("new_col3")
penalty = tf.feature_column.categorical_column_with_vocabulary_list(key="penalty", vocabulary_list=["l2", "l1", "none", "elasticnet"])

Feature_columns = [
    new_col1,
    new_col2,
    new_col3,
    max_iter,
    n_jobs, 
    n_samples, 
    n_features, 
    n_classes,
    n_clusters_per_class, 
    n_informative,
    tf.feature_column.indicator_column(penalty),
]

wide_columns= [new_col1, new_col2, new_col3, max_iter, n_jobs, n_samples, n_features, n_classes, n_clusters_per_class, n_informative,]

model = tf.estimator.DNNLinearCombinedRegressor(\
    linear_feature_columns =wide_columns,\
    dnn_feature_columns=Feature_columns,\
    dnn_hidden_units=[1000, 500, 250, 125, 75, 25, 14],\
    )

model.train(input_fn=input_train)

predict_input_fn = tf.estimator.inputs.pandas_input_fn(\
        x=test,\
        batch_size=1,\
        num_epochs=1,\
        shuffle=False)

predictions = model.predict(input_fn=predict_input_fn)

prediction = []
for pred in predictions:
    prediction.append(pred["predictions"][0])

prediction = pd.DataFrame(prediction, columns=['time']).abs()
prediction.to_csv('output_github_submit.csv', index=True, index_label='id') #creating output file with predictions
