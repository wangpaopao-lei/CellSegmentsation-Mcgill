#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 05:52
# @Author  : WangLei
# @File    : transformer1.py
# @Description :
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import gc
from tensorflow.keras.layers.experimental import preprocessing
import os
import time
import anndata as ad

def load_batch(file_path, start_idx, batch_size):
    data = np.load(file_path, allow_pickle=True)
    x_test = data['x_test'][start_idx:start_idx+batch_size].astype(np.float32)
    x_test_pos = data['x_test_pos'][start_idx:start_idx+batch_size].astype(int)
    x_test_labels = data['x_test_labels'][start_idx:start_idx+batch_size].astype(np.float32)
    return x_test, x_test_pos, x_test_labels



def dir_to_class(y_dir, class_num):
    y_dir_class = []
    for i in range(len(y_dir)):
        x, y = y_dir[i]
        if x == -9999:
            y_vec = np.zeros(class_num)
            y_dir_class.append(y_vec)
        else:
            if y == 0 and x > 0:
                deg = np.arctan(float('inf'))
            elif y == 0 and x < 0:
                deg = np.arctan(-float('inf'))
            elif y == 0 and x == 0:
                deg = np.arctan(0)
            else:
                deg = np.arctan((x / y))
            if (x > 0 and y < 0) or (x <= 0 and y < 0):
                deg += np.pi
            elif x < 0 and y >= 0:
                deg += 2 * np.pi
            cla = int(deg / (2 * np.pi / class_num))
            y_vec = np.zeros(class_num)
            y_vec[cla] = 1
            y_dir_class.append(y_vec)
    return np.array(y_dir_class)

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Dense(units=projection_dim)
        self.watershed_label=layers.Dense(units=projection_dim)

    def call(self, patch, position,watershed_label):
        encoded = self.projection(patch) + self.position_embedding(position)+self.watershed_label(watershed_label)
        return encoded


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x



def create_transformer_classifier(class_num, input_shape, input_position_shape,input_watershed_shape, num_patches, projection_dim, num_heads,
                                  transformer_units, transformer_layers, mlp_head_units):
    inputs = layers.Input(shape=input_shape)
    inputs_positions = layers.Input(shape=input_position_shape)
    inputs_watershed=layers.Input(shape=input_watershed_shape)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(inputs, inputs_positions,inputs_watershed)

    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Add MLP.
    features = mlp(representation[:, 0, :], hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    binary = layers.Dense(1, activation='sigmoid', name='cat_out')(features)

    model = keras.Model(inputs=[inputs, inputs_positions, inputs_watershed], outputs=binary)


    return model

        
class FirstEpochValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, batch_size):
        super().__init__()
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.first_epoch_done = False

        # 检查 validation_data 是否有效
        if len(self.validation_data[0]) == 0:
            raise ValueError("Validation data should contain at least one sample.")

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 and not self.first_epoch_done:
            # 随机选择一个 batch
            batch_indices = np.random.choice(len(self.validation_data[0]), self.batch_size)
            val_batch_data = [data[batch_indices] for data in self.validation_data[:-1]]

            # 计算这个 batch 的预测
            predictions = self.model.predict(val_batch_data)

            # 打印 labels 和 predictions
            actual_labels = self.validation_data[-1][batch_indices]  # 假设最后一个元素是标签
            print(f'Epoch {epoch + 1}: Actual Labels: {actual_labels}')
            print(f'Epoch {epoch + 1}: Predictions:\n {predictions}')

            self.first_epoch_done = True




def run_experiment(learning_rate, weight_decay,model, 
                        x_train_, x_train_pos_,x_train_labels_,x_train, x_train_pos,x_train_labels,
                        y_train_ ,y_train,
                        x_validation, x_validation_pos,x_validation_labels, 
                        y_bin_validation,  
                        batch_size, num_epochs):
    
    optimizer=tfa.optimizers.AdamW(
        learning_rate=learning_rate,weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss={
            'cat_out':keras.losses.BinaryCrossentropy(from_logits=False),
        },
        metrics={
            'cat_out':keras.metrics.BinaryAccuracy(name="accuracy"),
        },
    )

    checkpoint_filepath = os.path.join('./ckpt', 'model_' , 'ckpt')
    if not os.path.exists(checkpoint_filepath):
        os.makedirs(checkpoint_filepath)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    
    # custom_callback=CustomCallback()
     # 检查 x_validation 是否包含数据
    if len(x_validation) == 0:
        raise ValueError("x_validation should contain at least one sample.")

    validation_data = [x_validation, x_validation_pos, x_validation_labels, y_bin_validation]
    first_epoch_val_callback = FirstEpochValidationCallback(validation_data, batch_size)
    
    model.fit(
        x=[x_train, x_train_pos, x_train_labels], # 现在包含了 watershed 输入
        y=[y_train],
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.2,
        callbacks=[checkpoint_callback,first_epoch_val_callback],
        verbose=2
        )
    
    
    print('Inference on all the spots...')
    model.load_weights(checkpoint_filepath)
    pred_binary_test_all = []
    total_size = 160000  # 这里设置您的数据集大小pred_batch_size=10000
    pred_batch_size=10000
    print(f"{total_size/pred_batch_size} in total")
    for i in range((total_size + pred_batch_size - 1) // pred_batch_size):
        start_time=time.time()
        print(f"Batch {i} processing...", flush=True)
        x_batch, x_test_pos_batch, x_test_labels_batch = load_batch('dataset/x_test0:0:0:0.npz', i * pred_batch_size, min(pred_batch_size, total_size - i * pred_batch_size))
        pred_binary_test_ = model.predict(x=[x_batch, x_test_pos_batch, x_test_labels_batch], batch_size=batch_size)
        pred_binary_test_all.append(pred_binary_test_)
        tf.keras.backend.clear_session()
        gc.collect()
        
        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time  # 计算耗时
        # 将耗时转换为时:分:秒格式
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"Batch {i} completed. Time taken: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}", flush=True)
        

    pred_binary_test = np.vstack(pred_binary_test_all)

    
    x_test_pos_ = np.load('dataset/x_test_pos0:0:0:0.npz')
    x_test_pos_ = x_test_pos_['x_test_pos']
    # print(y_train_.shape, x_train_pos__.shape)

    print('Write prediction results...')
    with open('dataset/task1_result.txt',
              'w') as fw:
        for i in range(len(x_test_pos_)):
            fw.write(str(x_test_pos_[i][0][0]) + '\t' + str(x_test_pos_[i][0][1]) + '\t' + str(
                pred_binary_test[i][0]) + '\n')
    return 


def train(startx=0,starty=0,patchsize=0,epochs=20,val_ratio=0.2):
    startx=str(startx)
    starty=str(starty)
    patchsize=str(patchsize)
    try:
        os.mkdir('results/')
    except FileExistsError:
        print('results folder exists.')
    x_optimize_train_ = np.load('dataset/x_optimize_train' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz',allow_pickle=True)
    x_optimize_train_ = x_optimize_train_['x_optimize_train'].astype(np.float32)
    x_optimize_pos_ = np.load('dataset/x_optimize_pos' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz',allow_pickle=True)
    x_optimize_pos_ = x_optimize_pos_['x_optimize_pos'].astype(np.int32)
    x_optimize_labels_=np.load('dataset/x_optimize_labels' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz',allow_pickle=True)
    x_optimize_labels_=x_optimize_labels_['x_optimize_labels'].astype(np.float32)
    # y_optimize_train_ = np.load('dataset/y_optimize_train' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz',allow_pickle=True)
    # y_optimize_train_ = y_optimize_train_['y_optimize_train']
    
    y_bin_=np.load('dataset/y_bin' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz',allow_pickle=True)
    y_bin_=y_bin_['y_bin']

    class_num = 16

    # 获取数据集的总大小
    total_size = len(x_optimize_pos_)

    # 计算验证集的大小
    val_size = int(total_size * val_ratio)

    # 生成随机排列的索引
    indices = np.random.permutation(total_size)

    # 分割索引为训练集和验证集
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # 使用分割的索引来创建训练集和验证集
    x_train = x_optimize_train_[train_indices]
    x_train_pos = x_optimize_pos_[train_indices]
    x_train_labels = x_optimize_labels_[train_indices]
    # y_train = y_optimize_train_[train_indices]
    y_bin = y_bin_[train_indices]

    x_validation = x_optimize_train_[val_indices]
    x_validation_pos = x_optimize_pos_[val_indices]
    x_validation_labels = x_optimize_labels_[val_indices]
    # y_validation = y_optimize_train_[val_indices]
    y_bin_validation = y_bin_[val_indices]

    # val_threshold_0 = int(adata.layers['watershed_labels'].shape[0] * (1 - np.sqrt(val_ratio)))
    # val_threshold_1 = int(adata.layers['watershed_labels'].shape[1] * (1 - np.sqrt(val_ratio)))
    # print("Validation thresholds:", val_threshold_0, val_threshold_1)

    # for i in range(len(x_optimize_pos_)):
    #     if x_optimize_pos_[i][0][0] > val_threshold_0 and x_optimize_pos_[i][0][1] > val_threshold_1:
    #         x_validation_select.append(i)
    #     else:
    #         x_train_select.append(i)

    # print("Length of x_train_select:", len(x_train_select))
    # print("Length of x_validation_select:", len(x_validation_select))
    # # 可以打印出一些 x_optimize_pos_ 的样本来检查
    # print("Some samples from x_optimize_pos_:", x_optimize_pos_[:5])


    
    input_shape = (x_train.shape[1], x_train.shape[2])
    input_position_shape = (x_train_pos.shape[1], x_train_pos.shape[2])
    input_watershed_shape=(x_train_labels.shape[1],1)
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 32
    num_epochs = epochs
    num_patches = x_train.shape[1]
    projection_dim = 64
    num_heads = 1
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transfoœ∑rmer layers
    transformer_layers = 3
    mlp_head_units = [1024, 256]  # Size of the dense layers of the final classifier

    transformer_classifier = create_transformer_classifier(class_num, input_shape, input_position_shape,input_watershed_shape, num_patches,
                                                           projection_dim, num_heads, transformer_units,
                                                           transformer_layers, mlp_head_units)
    
    
    # run_experiment(learning_rate, weight_decay,transformer_classifier,
    #                x_optimize_train_, x_optimize_pos_,x_optimize_labels_, 
    #                x_train, x_train_pos,x_train_labels, 
    #                y_bin_,y_bin,
    #                x_test,x_test_pos,x_test_labels,
    #                 x_validation, x_validation_pos,x_validation_labels,
    #                 y_validation,y_bin_validation,  
    #                 batch_size, num_epochs)
    run_experiment(learning_rate, weight_decay,transformer_classifier,
                   x_optimize_train_, x_optimize_pos_,x_optimize_labels_, 
                   x_train, x_train_pos,x_train_labels, 
                   y_bin_,y_bin,
                    x_validation, x_validation_pos,x_validation_labels,
                    y_bin_validation,  
                    batch_size, num_epochs)


train()