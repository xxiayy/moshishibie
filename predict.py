# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import keras as kr

from model import TRNNConfig, TextRNN
from data.cnews_loader import read_category, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textrnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class RnnModel:
    def __init__(self):
        self.config = TRNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextRNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    rnn_model = RnnModel()
    test_demo = ['《云光仙境赤月战争》视频抢先看史诗般的RPG玄幻巨著，年度超五星强力推荐！高自由度的操作考验玩家的决策能力，多结局的选择由你掌控，有极高的重复可玩性。58种场景，255个人物，人、鬼、神、兽四大族类就此展开一场腥风血雨的赤月战争！游戏于本周在各大渠道正式上线，火爆视频也请抢先感受下吧。']
    for i in test_demo:
        print(rnn_model.predict(i))
