import tensorflow as tf
import tensorflow.contrib.slim as slim
import os


class Model:
    def __init__(self, learning_rate):
        self.model_name = 'DEFNet'
        normalize_value = 100
        self.input = tf.placeholder(tf.float32, [None, None, None, 4])
        self.target = tf.placeholder(tf.float32, [None, None, None, 4])
        self.parameters = tf.placeholder(tf.float32, [None, 7])
        self.is_training = tf.placeholder(tf.bool)

        #Parameter networks
        input = self.input * normalize_value
        target = self.target * normalize_value
        
        param_net = self.gen_param_net(self.parameters, self.is_training)
        param_net = tf.tile(param_net[:,tf.newaxis,tf.newaxis,:], [1, tf.shape(input)[1], tf.shape(input)[2], 1])

        x = []
        x0 = self.MainDenoiser(input, param_net)
        x.append(x0)
        x1 = self.Compensator(1, input, x, param_net)
        x.append(x1)
        x2 = self.Compensator(2, input, x, param_net)
        x.append(x2)
        x3 = self.Compensator(3, input, x, param_net)

        x3 = tf.clip_by_value(x3, 0, normalize_value)
        
        self.clean_img = x3 / normalize_value

        self.loss_MainDenoiser = tf.reduce_mean(tf.square(x0 - target))
        self.loss = tf.reduce_mean(tf.square(x3 - target)) + 10e-3*(tf.reduce_mean(tf.square(x2 - target)) + tf.reduce_mean(tf.square(x1 - target)) + tf.reduce_mean(tf.square(x0 - target)))
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)

        self.train_op_MainDenoiser = self.optimizer.minimize(self.loss_MainDenoiser, global_step=global_step)
        self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)

    def gen_param_net(input, parameters, is_training):
        param_net = tf.layers.dense(inputs=parameters,  units=7,  activation=None, name='param_net_dense1',  kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        param_net = tf.layers.dense(inputs=param_net,   units=7,  activation=None, name='param_net_dense2',  kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        param_net = tf.layers.dense(inputs=param_net,   units=7,  activation=None, name='param_net_dense3',  kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        param_net = tf.layers.dense(inputs=param_net,   units=7,  activation=None, name='param_net_dense4',  kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        param_net = tf.layers.dropout(param_net, 0.9, training=is_training)
        param_net = tf.layers.dense(inputs=param_net,   units=4,  activation=None, name='param_net_dense5',  kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        
        return param_net
        

    def MainDenoiser(self, input, P):
        r = h = None
        for i in range(10):
            r, h = self.SubDenoiser(i + 1, input, P, r, h, i != 9)
        return input - r

    def SubDenoiser(self, block_idx, input, P, r=None, h=None, h_gen=True):
        net = input
        if r != None:
            net = tf.concat([net, r], 3)
        if h != None:
            net = tf.concat([net, h], 3)
        net = tf.concat([net, P], 3)
        net = self.DenoiserDeconvLayer(block_idx, 1, net, 64, tf.nn.relu, 1)
        net = self.DenoiserDeconvLayer(block_idx, 2, tf.concat([net, P], 3), 64, tf.nn.relu, 2, 0.1)
        net = self.DenoiserDeconvLayer(block_idx, 3, tf.concat([net, P], 3), 64, tf.nn.relu, 3, 0.1)
        net = self.DenoiserDeconvLayer(block_idx, 4, tf.concat([net, P], 3), 64, tf.nn.relu, 3, 0.1)
        net = self.DenoiserDeconvLayer(block_idx, 5, tf.concat([net, P], 3), 64, tf.nn.relu, 2, 0.1)
        h_net = self.DenoiserDeconvLayer(block_idx, 6, net, 64, None, 1) if h_gen else None
        r_net = self.DenoiserDeconvLayer(block_idx, 7, net, 4, None, 1)
        r_net = r_net + r if r != None else r_net

        return r_net, h_net

    def DenoiserDeconvLayer(self, block_idx, layer_idx, input, filters, activate, rate, scale=1.0):
        net = slim.conv2d(input, filters, [3, 3], rate=rate, activation_fn=activate, scope='block_%d/conv_%d' % (block_idx, layer_idx), weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
        if scale != 1.0:
            net*=scale
        return net

    def Compensator(self, subnet_idx, input, x, P):
        r = h = None
        for i in range(10):
            r, h = self.SubCompensator(subnet_idx, i + 1, input, x, P, r, h, i != 9)
        return x[-1] - r
        
    def SubCompensator(self, subnet_idx, block_idx, input, x, P, r=None, h=None, h_gen=True):
        net = tf.concat([input]+x, 3)
        if r != None:
            net = tf.concat([net, r], 3)
        if h != None:
            net = tf.concat([net, h], 3)
        net = tf.concat([net, P], 3)
        net = self.CompensatorDeconvLayer(subnet_idx, block_idx, 1, net, 32, tf.nn.relu, 1)
        net = self.CompensatorDeconvLayer(subnet_idx, block_idx, 2, tf.concat([net, P], 3), 32, tf.nn.relu, 2, 0.1)
        net = self.CompensatorDeconvLayer(subnet_idx, block_idx, 3, tf.concat([net, P], 3), 32, tf.nn.relu, 3, 0.1)
        net = self.CompensatorDeconvLayer(subnet_idx, block_idx, 4, tf.concat([net, P], 3), 32, tf.nn.relu, 3, 0.1)
        net = self.CompensatorDeconvLayer(subnet_idx, block_idx, 5, tf.concat([net, P], 3), 32, tf.nn.relu, 2, 0.1)
        h_net = self.CompensatorDeconvLayer(subnet_idx, block_idx, 6, net, 32, None, 1) if h_gen else None
        r_net = self.CompensatorDeconvLayer(subnet_idx, block_idx, 7, net, 4, None, 1)
        r_net = r_net + r if r != None else r_net

        return r_net, h_net

    def CompensatorDeconvLayer(self, subnet_idx, block_idx, layer_idx, input, filters, activate, rate, scale=1.0):
        net = slim.conv2d(input, filters, [3, 3], rate=rate, activation_fn=activate, scope='subnet_%d/block_%d/conv_%d' % (subnet_idx, block_idx, layer_idx), weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
        if scale != 1.0:
            net*=scale
        return net



    def save(self, sess, saver, checkpoint_dir):
        print(" [*] Save checkpoints...")
        model_name = self.model_name + ".model"
        model_dir = self.model_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        saver.save(sess, os.path.join(checkpoint_dir, model_name))

    def load(self, sess, saver, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = self.model_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False