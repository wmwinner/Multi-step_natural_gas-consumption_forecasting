import os
from module.regression.Model_train import train_factory as regression_factory
from utils.timeUtils import *

df = pd.read_csv('data/finance.csv')
params = {}
params['id'] = 1
params['seq_len'] = 24
params['pred_len'] = 1
params['label_len'] = 4
params['date_column'] = 'date'
params['label'] = ['Open']
params['columns'] = ['Open', "High", "Low", "Close", "Volume", "Adj Close"]
params['input_dim'] = 6
params['output_dim'] = 1
params['seed'] = 1234
params['features'] = 'MS'
params['inverse'] = True
params['output_test'] = True
params['batch_size'] = 256
params['loss'] = 'MAPE'
params['lr'] = 1e-4
params['dropout'] = 0.2
params['grid_search'] = False

for model_name in ["HATL_Seq2Seq"]:
    metrics_data, y_true, y_pred, test_date = regression_factory(model_name, df, params)





