# Text Classification with CNN and RNN

使用循环神经网络进行中文文本分类

## 环境

- Python 3
- TensorFlow 1.3以上
- numpy
- scikit-learn
- scipy

## 数据集

使用THUCNews的一个子集进行训练与测试。

本次训练使用了其中的10个分类，每个分类6500条数据。

类别如下：

```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```
三个文件
- cnews.train.txt: 训练集(50000条)
- cnews.val.txt: 验证集(5000条)
- cnews.test.txt: 测试集(10000条)

## 预处理

- `read_file()`: 读取文件数据;
- `build_vocab()`: 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理;
- `read_vocab()`: 读取上一步存储的词汇表，转换为`{词：id}`表示;
- `read_category()`: 将分类目录固定，转换为`{类别: id}`表示;
- `to_words()`: 将一条由id表示的数据重新转换为文字;
- `process_file()`: 将数据集从文字转换为固定长度的id序列表示;
- `batch_iter()`: 为神经网络的训练准备经过shuffle的批次的数据。

经过数据预处理，数据的格式如下：

| Data | Shape | Data | Shape |
| :---------- | :---------- | :---------- | :---------- |
| x_train | [50000, 600] | y_train | [50000, 10] |
| x_val | [5000, 600] | y_val | [5000, 10] |
| x_test | [10000, 600] | y_test | [10000, 10] |


## RNN循环神经网络

### 配置项

RNN可配置的参数如下所示，在`model.py`中。

```python
class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    vocab_size = 5000       # 词汇表达小

    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 10          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard
```

### RNN模型

具体参看`model.py`的实现。


### 训练与验证

运行`python rnn.py train`训练：若超过1000轮未提升则结束

```
Configuring RNN model...
Configuring TensorBoard and Saver...
Loading training and validation data...
Time usage: 0:00:14
Training and evaluating...
Epoch: 1
Iter:      0, Train Loss:    2.3, Train Acc:   8.59%, Val Loss:    2.3, Val Acc:   9.48%, Time: 0:00:11
Iter:    100, Train Loss:   0.87, Train Acc:  63.28%, Val Loss:    1.2, Val Acc:  55.88%, Time: 0:01:38 
Iter:    200, Train Loss:    0.4, Train Acc:  87.50%, Val Loss:   0.87, Val Acc:  70.58%, Time: 0:03:05 
Iter:    300, Train Loss:   0.26, Train Acc:  91.41%, Val Loss:   0.57, Val Acc:  83.22%, Time: 0:04:31 
Epoch: 2
Iter:    400, Train Loss:   0.34, Train Acc:  89.06%, Val Loss:   0.59, Val Acc:  84.84%, Time: 0:05:57 
Iter:    500, Train Loss:   0.27, Train Acc:  87.50%, Val Loss:   0.65, Val Acc:  82.82%, Time: 0:07:23 
Iter:    600, Train Loss:   0.17, Train Acc:  93.75%, Val Loss:   0.49, Val Acc:  87.64%, Time: 0:08:50 
Iter:    700, Train Loss:   0.22, Train Acc:  93.75%, Val Loss:   0.48, Val Acc:  87.50%, Time: 0:10:16 
Epoch: 3
Iter:    800, Train Loss:   0.13, Train Acc:  96.09%, Val Loss:   0.47, Val Acc:  88.52%, Time: 0:11:42 
Iter:    900, Train Loss:   0.16, Train Acc:  95.31%, Val Loss:   0.44, Val Acc:  89.24%, Time: 0:13:09 
Iter:   1000, Train Loss:   0.16, Train Acc:  96.09%, Val Loss:   0.42, Val Acc:  89.20%, Time: 0:14:35 
Iter:   1100, Train Loss:  0.078, Train Acc:  97.66%, Val Loss:   0.34, Val Acc:  91.68%, Time: 0:16:01 
Epoch: 4
Iter:   1200, Train Loss:  0.074, Train Acc:  97.66%, Val Loss:   0.36, Val Acc:  90.76%, Time: 0:17:27 
Iter:   1300, Train Loss:    0.2, Train Acc:  95.31%, Val Loss:   0.36, Val Acc:  91.70%, Time: 0:18:53 
Iter:   1400, Train Loss:   0.27, Train Acc:  91.41%, Val Loss:   0.73, Val Acc:  82.02%, Time: 0:20:19 
Iter:   1500, Train Loss:  0.077, Train Acc:  98.44%, Val Loss:   0.35, Val Acc:  90.50%, Time: 0:21:46 
Epoch: 5
Iter:   1600, Train Loss:  0.024, Train Acc:  99.22%, Val Loss:   0.33, Val Acc:  91.58%, Time: 0:23:12 
Iter:   1700, Train Loss:   0.14, Train Acc:  96.09%, Val Loss:   0.35, Val Acc:  91.38%, Time: 0:24:38 
Iter:   1800, Train Loss:  0.073, Train Acc:  99.22%, Val Loss:   0.35, Val Acc:  91.32%, Time: 0:26:04 
Iter:   1900, Train Loss:  0.074, Train Acc:  96.09%, Val Loss:   0.31, Val Acc:  91.62%, Time: 0:27:30 
Epoch: 6
Iter:   2000, Train Loss:  0.038, Train Acc:  98.44%, Val Loss:   0.32, Val Acc:  92.04%, Time: 0:28:56 
Iter:   2100, Train Loss:  0.031, Train Acc:  99.22%, Val Loss:   0.31, Val Acc:  92.32%, Time: 0:30:22 
Iter:   2200, Train Loss:   0.16, Train Acc:  95.31%, Val Loss:   0.36, Val Acc:  91.74%, Time: 0:31:48 
Iter:   2300, Train Loss:  0.058, Train Acc:  99.22%, Val Loss:   0.31, Val Acc:  91.76%, Time: 0:33:13 
Epoch: 7
Iter:   2400, Train Loss:  0.086, Train Acc:  96.88%, Val Loss:   0.41, Val Acc:  89.92%, Time: 0:34:39 
Iter:   2500, Train Loss:   0.11, Train Acc:  98.44%, Val Loss:   0.36, Val Acc:  91.30%, Time: 0:36:05 
Iter:   2600, Train Loss:  0.077, Train Acc:  95.31%, Val Loss:    0.4, Val Acc:  90.10%, Time: 0:37:31 
Iter:   2700, Train Loss:   0.12, Train Acc:  96.88%, Val Loss:   0.33, Val Acc:  91.58%, Time: 0:38:57 
Epoch: 8
Iter:   2800, Train Loss:  0.041, Train Acc:  98.44%, Val Loss:   0.36, Val Acc:  91.40%, Time: 0:40:23 
Iter:   2900, Train Loss:   0.13, Train Acc:  95.31%, Val Loss:   0.39, Val Acc:  90.54%, Time: 0:41:48 
Iter:   3000, Train Loss:  0.077, Train Acc:  97.66%, Val Loss:   0.37, Val Acc:  91.16%, Time: 0:43:14 
Iter:   3100, Train Loss:   0.04, Train Acc:  99.22%, Val Loss:   0.37, Val Acc:  90.38%, Time: 0:44:40 
No optimization for a long time, auto-stopping...
```

在验证集上的最佳效果为92.32%，经过了8轮迭代停止。

准确率和误差如图所示

![image-20230615194345118](C:\Users\31925\AppData\Roaming\Typora\typora-user-images\image-20230615194345118.png)

![image-20230615194418479](C:\Users\31925\Desktop\image-20230615194418479.png)


### 测试

运行 `python rnn.py test` 在测试集上进行测试

```
Testing...
Test Loss:   0.18, Test Acc:  95.23%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

          体育       0.99      0.97      0.98      1000
          财经       0.95      0.98      0.97      1000
          房产       1.00      1.00      1.00      1000
          家居       0.97      0.88      0.92      1000
          教育       0.88      0.94      0.91      1000
          科技       0.91      0.98      0.94      1000
          时尚       0.96      0.96      0.96      1000
          时政       0.93      0.92      0.93      1000
          游戏       0.98      0.93      0.95      1000
          娱乐       0.96      0.96      0.96      1000

    accuracy                           0.95     10000
   macro avg       0.95      0.95      0.95     10000
weighted avg       0.95      0.95      0.95     10000

Confusion Matrix...
[[975   0   0   0  13   3   0   4   0   5]
 [  0 982   0   0   2   2   0  14   0   0]
 [  0   0 996   3   0   0   1   0   0   0]
 [  0  17   1 878  26  34  15  20   4   5]
 [  2   3   0   7 938  19   5  21   3   2]
 [  0   1   1  10   2 978   2   2   4   0]
 [  1   1   0   6  14   1 963   0   4  10]
 [  2  14   0   3  35  21   0 923   1   1]
 [  0  12   0   1  18  12  13   1 929  14]
 [  2   2   0   1  12   4   7   5   6 961]]
Time usage: 0:00:42
```

在测试集上的准确率达到了95.23%。

从混淆矩阵可以看出分类效果非常优秀。


## 预测

为方便预测，`predict.py` 提供了 RNN 模型的预测方法。
