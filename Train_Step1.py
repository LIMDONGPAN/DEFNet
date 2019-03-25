import tensorflow as tf
import numpy as np
import Model
import random
import os
import UTILS


class Main:
    def __init__(self):
        self.channel_size = 4
        self.param_size =7    
        self.batch_size = 8
        self.patch_size = 64
        self.learning_rate = 10e-4
        self.checkpoint_dir  = 'checkpoint'
        
        self.model = Model.Model(self.learning_rate)
        
        
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.coord = tf.train.Coordinator()



    def train(self):
        print("Begin training...")
        init = tf.global_variables_initializer()
        with self.sess as sess:
            sess.run(init)

            if self.model.load(sess, self.saver, self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            file_list = UTILS.get_training_file_list_by_metafile()
            
            batch_count = 0
            train_input = []
            train_target = []
            train_params = []
            while True:
                for fi in file_list:
                    noisy, gt, params = UTILS.get_sample_from_file(fi, self.patch_size)
                    train_input.append(noisy)
                    train_target.append(gt)
                    train_params.append(params)
                    batch_count+=1
                    if batch_count >= self.batch_size:
                        train_feed = {
                                	self.model.input:train_input,
                                	self.model.target:train_target,
                                	self.model.parameters:train_params,
                                    self.model.is_training:True
                                }
                        loss,_ = sess.run([self.model.loss,self.model.train_op_MainDenoiser],train_feed)
                        print(loss)
                        train_input.clear()
                        train_target.clear()
                        train_params.clear()
                        batch_count = 0
                        
                    
                self.model.save(sess, self.saver, self.checkpoint_dir)
        
        


if __name__ == '__main__':
    obj = Main()
    obj.train()



