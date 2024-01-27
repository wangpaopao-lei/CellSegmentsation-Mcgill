import numpy as np
import os
import gc
from transformer1 import create_transformer_classifier
import tensorflow as tf
    
def load_batch(file_path, start_idx, batch_size):
    data = np.load(file_path, allow_pickle=True)
    x_test = data['x_test'][start_idx:start_idx+batch_size].astype(np.float32)
    x_test_pos = data['x_test_pos'][start_idx:start_idx+batch_size].astype(int)
    x_test_labels = data['x_test_labels'][start_idx:start_idx+batch_size].astype(np.float32)
    return x_test, x_test_pos, x_test_labels    

def pred(model,batch_size=10):
    
    
    checkpoint_filepath = os.path.join('./ckpt', 'model_' , 'ckpt')
    print('Inference on all the spots...')
    model.load_weights(checkpoint_filepath)
    
    pred_binary_test_all = []
    total_size = 1440000  # 这里设置您的数据集大小
    for i in range(int(total_size / 10000) + 1):
        tf.keras.backend.clear_session()
        x_batch, x_test_pos_batch, x_test_labels_batch = load_batch('dataset/x_test0:0:0:0.npz', i * 10000, min(10000, total_size - i * 10000))
        
        pred_binary_test_ = model.predict(
            x=[x_batch, x_test_pos_batch, x_test_labels_batch], batch_size=batch_size)
        pred_binary_test_all.append(pred_binary_test_)
        gc.collect()

    pred_binary_test = np.vstack(pred_binary_test_all)

    
    # for i in range(len(y_train_)):
    #    print(y_train_[i], pred_binary_train[i], pred_centers_train[i])
    x_test_pos_ = np.load('dataset/x_test_pos0:0:0:0.npz')
    x_test_pos_ = x_test_pos_['x_test_pos']
    # print(y_train_.shape, x_train_pos__.shape)

    print('Write prediction results...')
    with open('results/spot_prediction_0:0:0:0.txt',
              'w') as fw:
        for i in range(len(x_test_pos_)):
            fw.write(str(x_test_pos_[i][0][0]) + '\t' + str(x_test_pos_[i][0][1]) + '\t' + str(
                pred_binary_test[i][0]) + '\n')
    return 




transformer_classifier = create_transformer_classifier(class_num=16, input_shape=[50,2000], input_position_shape=[50,2],input_watershed_shape=[50,1], num_patches=50,
                                                           projection_dim=64, num_heads=1, transformer_units=[64*2,64],
                                                           transformer_layers=2, mlp_head_units=[1024,256])
pred(transformer_classifier)