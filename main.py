from etc_data_handler import etc_handler
from util import *
from etc_rnn import *


# base params
sequence_length = 20
number_features = 5
root_path = 'I:\ETC\TBrain_Round2_DataSet_20180331'
code_number = 1101
# network params
layer_2_output = 100
FC_output = 1
bs = 200
epoch = 200
vali = 0.1



etc_data = etc_handler(root_path)
price_data = etc_data.get_price(code_number)
X_train, Y_train, X_test, Y_test = preprocess_data(price_data, sequence_length,number_features)
model = build_model( [X_train.shape[2], sequence_length, layer_2_output, FC_output] )
model.fit(X_train, Y_train, batch_size = bs, nb_epoch=epoch, validation_split=vali, verbose=0)
train_result = score(X_train, Y_train, model)
test_result = score(X_test, Y_test, model)
predict_result = predict(X_test,model)
print (predict_result)
