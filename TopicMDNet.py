"""
 * @author[author]
 * @email[example@mail.com]
 * @date 2017 - 03 - 02 11: 33: 40
 * @desc[My CVPR paper MDNet]
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf


class TopicMDNet():

    def __init__(self, opt, data_opt, conv_feat_from_CNN, 
                cls_logits_from_CNN, cls_weight_op_name, 
                feed_img_batch=None):
        print ('--> Configuring TopicMDNet8v2 ...')
        # dataset params
        self.vocab_size = data_opt.vocab_size
        self.PAD_TOKEN = data_opt.PAD_TOKEN
        self.vocab_size = self.vocab_size 
        self.cls_loss_weight = opt.cls_loss_weight
        self.num_feature = data_opt.num_feature or 6
        self.max_len = data_opt.max_subseq_len
        # LSTM params
        self.input_encoding_size = 128
        self.rnn_size = 256
        self.num_layers = 1
        self.sample_max = opt.sample_max
        self.tempature = 1
        self.attfeat_dim = 128 #TODO
        self.att_weight = opt.att_weight
        # CNN params
        self.conv_feat_dim = opt.conv_feat_dim 
        self.conv_feat_w = opt.conv_feat_wh 
        self.conv_feat_h = opt.conv_feat_wh 
        self.conv_feat_spat =  self.conv_feat_w * self.conv_feat_h
        self.weight_decay = 0.0001
        self.cls_weight_op_name = cls_weight_op_name
        self.cnn_scope = 'PRETRAINED_CNN'
        # optimizer
        self.max_grad_norm = 0.1
        self.grad_clip = 0.1
        self.lr_decay_rate = opt.lr_decay_rate
        self.embed_initializer = tf.random_uniform_initializer(
                            minval=-0.08,
                            maxval=0.08)
        self.initializer = tf.contrib.layers.xavier_initializer()

        ## things need to get from input
        self.feed_conv_batch = conv_feat_from_CNN
        self.cls_logits = cls_logits_from_CNN

        self.feed_txt_batch = tf.placeholder(
            'int32', [None, self.max_len-1], name='text_seq')
       
        self.feed_cls_labels = tf.placeholder('int32', name='disease_label')
        self.feed_text_labels = tf.placeholder('int32', [None, self.max_len], name='seq_label')
        self.is_train_mode = tf.placeholder('bool', name='is_train_mode')
        self.feed_keep_drop = tf.placeholder('float32', name='dropout_rate')
        self.feed_stop_indictor = tf.placeholder('int32', [None, self.num_feature], name='stop_label')
        
        self.feed_img_batch = feed_img_batch
        #Sets up the global step Tensor.
        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[
                tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES
            ])

    def minimize(self, logits, stop_logits, logits_cls, init_learning_rate, 
                iter_epoch, freeze_layers=[]):
        """Minimize funcs
            Input: 
                logits: preds in [[batch_size,self.netout_dim], ...] with nHops returned by self.build_model
                learning_rate: a place_holder
                self.label [batch_size, self.max_len]
        """

        def decay(decay_vars):
            """L2 weight decay loss."""
            costs = []
            for var in decay_vars:
                costs.append(tf.nn.l2_loss(var))
            return tf.multiply(self.weight_decay, tf.add_n(costs))
        def clip_by_value(grad, exclude=None):
                if grad is None or exclude in grad.name:
                    return grad
                return tf.clip_by_value(grad, -self.grad_clip, self.grad_clip)
        
        def apply_gradient(loss, vars, optier):
            gradients = tf.gradients(loss, vars)    
            clipped_gradients = [clip_by_value(grad, exclude=self.cnn_scope) for grad in gradients]
            train_op = optier.apply_gradients(zip(clipped_gradients, vars), global_step=self.global_step)
            return train_op

        tvars_raw = tf.trainable_variables()
        print ('\t --> {} out of {} cnn vars are freezed'.format(len(freeze_layers), len(tvars_raw)))
        tvars_all = list(set(tvars_raw) - set(freeze_layers))
        assert(len(tvars_all)==(len(tvars_raw)-len(freeze_layers)) )
      
        tvars_lstm = [var for var in tvars_all if self.cnn_scope not in var.name]
        tvars_cnn = [var for var in tvars_all if self.cnn_scope in var.name]
        
        print ('\t --> {} vars are in the cnn optimizer'.format(len(tvars_cnn)))
        with tf.name_scope('learning_rate'):
            learning_rate = tf.train.exponential_decay(init_learning_rate, self.global_step, iter_epoch, self.lr_decay_rate, staircase=True)
            print ('\t --> decay learning rate every {} iteration'.format(iter_epoch))
            tf.summary.scalar('learning_rate', learning_rate)

        with tf.name_scope('cnn_optimizer'):
            # loss of cnn
            loss_op_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.feed_cls_labels, logits=logits_cls) 
            loss_op_cls = tf.reduce_mean(loss_op_cls)
            # loss_op_cls += decay(tvars_cnn) # add l2 decay
            correct_prediction = tf.equal(self.feed_cls_labels, tf.cast(tf.argmax(logits_cls, 1), tf.int32))
            self.cls_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('lstm_optimizer'):
            targets = tf.reshape(self.feed_text_labels, [-1])
            weights = tf.cast(tf.not_equal(targets, self.PAD_TOKEN), tf.float32)   
            logits = tf.reshape(logits, [-1, self.vocab_size])
            # loss of sentence lstm
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
            loss_op_slstm = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                                            tf.reduce_sum(weights),
                                            name="slstm_loss")
            # losss for topic lstm
            targets = tf.reshape(self.feed_stop_indictor, [-1])
            weights = tf.cast(tf.not_equal(targets, self.PAD_TOKEN), tf.float32) 
            stop_logits = tf.reshape(stop_logits, [-1, 3]) 

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=stop_logits)
            loss_op_tlstm = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                                            tf.reduce_sum(weights),
                                            name="tlstm_loss")
            loss_op_lstm = loss_op_slstm + loss_op_tlstm * 0.5

            ## Uing this update_ops will change moving_mean and moving_variance. It will cause performance decrease when extract slide features for diagnosis
            
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)   
            
            with tf.name_scope('training_lstm_only'):
                train_op_lstm = apply_gradient(loss_op_lstm, tvars_lstm, optimizer)
                
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for updating batch_norm
            with tf.control_dependencies(update_ops):
                with tf.name_scope('training_end2end'):
                    loss_joint = loss_op_lstm + loss_op_cls * 0.1
                    train_op_lstm_e2e = apply_gradient(loss_joint, tvars_all, optimizer)

        tf.losses.add_loss(loss_op_cls)
        total_loss = tf.losses.get_total_loss()
        # write summary
        tf.summary.scalar('accuracy/cnn_train_acc', self.cls_acc)
        tf.summary.scalar("loss/lstm_loss", loss_op_lstm)
        tf.summary.scalar("loss/cnn_loss", total_loss)
        ## tf.summary.scalar("parameter_norm", g_norm)
        for var in tvars_lstm:
            if self.cnn_scope not in var.name:
                tf.summary.histogram("parameters/" + var.op.name, var)
        summary_merged = tf.summary.merge_all()

        return loss_op_lstm, loss_op_cls, train_op_lstm, train_op_lstm_e2e, summary_merged

    def image_embedding(self, conv_maps, reuse=True):
        """Embedding the global-average-pool feature map vector
        """
        # with tf.variable_scope('image_embedding', reuse=True) as scope:
        with tf.variable_scope('image_embedding', reuse=reuse) as scope:
            linear_feat = tf.contrib.layers.avg_pool2d(conv_maps, 
                kernel_size=[self.conv_feat_h, self.conv_feat_w], padding="VALID")
            linear_feat = tf.contrib.layers.flatten(linear_feat)
            # #TODO: do we need dropout and batch norm here?
            embed_linear_feat = tf.contrib.layers.fully_connected(
                                    inputs=linear_feat,
                                    num_outputs=self.rnn_size,
                                    activation_fn=tf.identity, # if None, the LSTM will make an error. Weird!
                                    weights_initializer=self.initializer,
                                    scope=scope)
            
        return embed_linear_feat

    def attention(self, conv_maps, cam_embed, prev_h, id='', reuse=True): # attention v2
        # The very baisc attention model
        with tf.variable_scope('attenton', reuse=reuse):
            tfeatproj = tf.contrib.layers.fully_connected(
                    prev_h,
                    num_outputs=self.attfeat_dim,
                    activation_fn=None,
                    weights_initializer=self.initializer,
                    biases_initializer=None)
            tfeatproj = tf.expand_dims(tfeatproj, axis=1)
            tfeatproj = tf.tile(tfeatproj, multiples=[1, self.conv_feat_spat+1, 1]) # +1 to consider cam_ifeat

            ifeatproj = tf.contrib.layers.conv2d(
                    inputs=conv_maps,
                    num_outputs=self.attfeat_dim,
                    kernel_size=1,
                    activation_fn=None,
                    weights_initializer=self.initializer,
                    biases_initializer=None)
            ifeatproj = tf.reshape(ifeatproj, shape=[-1, self.conv_feat_spat, self.attfeat_dim])

            # cam ifeat
            ## cam_ifeat = tf.reduce_sum(ifeatproj * tf.expand_dims(tf.nn.sigmoid(cam_embed), axis=2), 1, name='cam_context_vector')
            cam_ifeat = tf.reduce_sum(ifeatproj * tf.expand_dims(tf.nn.softmax(cam_embed), axis=2), 1, name='cam_context_vector')
            cam_ifeat = tf.expand_dims(cam_ifeat, axis=1)
            # sum over 
            tot_feat = tf.concat([ifeatproj, cam_ifeat],axis=1) # [B, conv_feat_spat+1, attfeat_dim]
            addfeat = tf.nn.tanh(tot_feat + tfeatproj)

            addfeat = tf.expand_dims(addfeat, axis=2)
            attscore = tf.contrib.layers.conv2d(
                inputs=addfeat,
                num_outputs=1,
                kernel_size=1,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None)
            attscore = tf.squeeze(attscore, axis=[2, 3])

            att_prob = tf.nn.softmax(attscore) # pay attention to the weight
            
            att_prob = tf.slice(att_prob, begin=[0, 0], size=[-1, self.conv_feat_spat]) # slce the used one
            tf.summary.histogram('att/attscore'+id,attscore)

            ifeat = tf.reshape(conv_maps, shape=[-1, self.conv_feat_spat, self.conv_feat_dim])
            context_h = tf.reduce_sum(ifeat * tf.expand_dims(
                att_prob, axis=2), 1, name='context_vector')
            context_vector = tf.contrib.layers.fully_connected(
                inputs=context_h,
                num_outputs=self.rnn_size,
                activation_fn=tf.nn.relu, # relu here is useful
                weights_initializer=self.initializer)
                
            return context_vector, att_prob
        
    def build_lstm_cell(self, batch_size, rnn_size): 
        """Create attention LSTM cell, which will called by self.classifier"""

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=rnn_size, state_is_tuple=True)
        lstm_init_state = lstm_cell.zero_state(
            batch_size=batch_size, dtype='float32')

        return lstm_cell, lstm_init_state
        
    def lstm(self, name, reuse, join_input, prev_state):

        with tf.variable_scope(name, reuse=reuse) as scope:
            hidden, state = getattr(self, name)(join_input, prev_state)

        return hidden, state

    def word_embeddings(self, feed_seq_batch, reuse=True):
        """Builds the input sequence embeddings.
        Inputs:
            feed_seq_batch [batch_size, max_len]
        Outputs:
            word_embeddings [batch_size, input_encoding_size]
        """
        if not hasattr(self, 'word_scope'): 
            scope = "word_embedding"
        else: 
            scope = self.word_scope
        with tf.variable_scope(scope, reuse=reuse) as word_embeddings_scope:
            seq_embedding = tf.get_variable(
                name="word_embedding",
                shape=[self.vocab_size, self.rnn_size],
                initializer=self.embed_initializer)
            word_embeddings = tf.nn.embedding_lookup(seq_embedding,
                                                        feed_seq_batch)
        self.word_scope = word_embeddings_scope
        return word_embeddings

    def decoder(self, feat, reuse=True):

        with tf.variable_scope('decoder', reuse=reuse):
            feat = tf.nn.dropout(feat, keep_prob=self.feed_keep_drop)
            pred = tf.contrib.layers.fully_connected(
                            inputs=feat,
                            num_outputs=self.vocab_size,
                            activation_fn=None,
                            weights_initializer=self.initializer)
            # pred = tf.cond(self.is_train_mode, lambda : pred, lambda : tf.nn.softmax(pred))
            pred = tf.cond(self.is_train_mode, lambda : pred, lambda : tf.nn.softmax(pred))
            
        return pred

    def continue_indictor(self, feat, reuse=True):
        # predict whether continue or stop
        with tf.variable_scope('continue_indictor', reuse=reuse):
                feat = tf.nn.dropout(feat, keep_prob=self.feed_keep_drop)
                pred = tf.contrib.layers.fully_connected(
                                inputs=feat,
                                num_outputs=3, 
                                activation_fn=None,
                                weights_initializer=self.initializer)
                # pred = tf.cond(self.is_train_mode, lambda : pred, lambda : tf.nn.softmax(pred))
                pred = tf.cond(self.is_train_mode, lambda : pred, lambda : tf.nn.softmax(pred))
                
        return pred

    def return_feed_placeholder(self):
        
        return self.feed_txt_batch, self.feed_stop_indictor, self.feed_text_labels, self.feed_cls_labels, self.global_step, self.cls_acc

    def unsample_att(self, att_prob):

        with tf.name_scope('unsample_att'):
            imgs = tf.tile(self.feed_img_batch, [self.num_feature, 1,1,1])
            h, w = imgs.shape[1:3]
            # normalization
            norm_att_prob = att_prob - tf.expand_dims(tf.reduce_min(att_prob,1), axis=1)
            norm_att_prob = norm_att_prob / tf.expand_dims(tf.reduce_max(norm_att_prob, axis=1), axis=1)
            # overlay
            attention_maps = tf.reshape(norm_att_prob, shape=[-1, self.conv_feat_h, self.conv_feat_w, 1])
            attention_maps = imgs * 0.3 + 0.7 * tf.image.resize_images(attention_maps, size=[int(h), int(w)])

        return attention_maps


    def build_train_model(self):
        """ Build the overall model
        """
        # tile the conv_feat for num_feature
        with tf.name_scope('CNN_inputs'):
            # feed_conv_batch = self._batch_norm(self.feed_conv_batch, name='conv_features', reuse=False)
            feed_conv_batch = self.feed_conv_batch
            # feed_conv_batch has two branches, Brach 2 can accept gradients and backpro to CNN
            # Branch 1
            with tf.name_scope('No_backprop'):
                conv_feat = tf.stop_gradient(feed_conv_batch, name='stop_conv_gradient') # do not back gradients
                cls_logits = tf.stop_gradient(self.cls_logits, name='stop_cls_logit_gradient')  # do not back gradients

                cam_embed = self.get_att_embedding(conv_feat, cls_logits)
                # cam_embed = tf.tile(cam_embed, multiples=[self.num_feature,1], name='cam_feat')
                # cam_images = self.unsample_att(cam_embed)
                # tf.summary.image('cam_attention', cam_images)
                # conv_feat = tf.tile(conv_feat, multiples=[self.num_feature,1,1,1], name='conv_feat')
            # Branch 2
            # Note that, when not training end2end, we block the gradient also
            i_embed = self.image_embedding(feed_conv_batch, reuse=False)
           
        flow_batch_size = tf.shape(self.feed_txt_batch)[0]  # get batch size 
        t_embed = self.word_embeddings(self.feed_txt_batch, reuse=False)
        
        concept_states = []
        stop_preds = []
        contexts = []
        with tf.variable_scope('CLSTM'):
            # concept lstm with multiple topics
            self.clstm_cell, lstm_state = self.build_lstm_cell(flow_batch_size//self.num_feature, rnn_size=self.rnn_size)
            for i in range(self.num_feature): # 
                # the first is image embedding
                prev_h, lstm_state = self.lstm('clstm_cell', i!=0, i_embed, lstm_state)
                context, att_prob = self.attention(feed_conv_batch, cam_embed, prev_h, id=str(i+1), reuse=i!=0) # do not block conv_feat in attention computation
                pred = self.continue_indictor(prev_h, reuse=i!=0)

                contexts.append(tf.expand_dims(context,axis=0))
                concept_states.append(tf.expand_dims(prev_h, axis=0)) # put batch at the second dimension
                stop_preds.append(tf.expand_dims(pred, axis=1))

        stop_preds = tf.concat(stop_preds, axis=1)
        concept_states = tf.concat(concept_states, axis=0)
        concept_states = tf.reshape(concept_states, shape=[-1, concept_states.shape[2]]) #[bum_feature*batch_size, rnn_size]
        contexts = tf.concat(contexts, axis=0)
        contexts = tf.reshape(contexts, shape=[-1, contexts.shape[2]]) #[bum_feature*batch_size, rnn_size]

        with tf.variable_scope('SLSTM'):
            preds = []
            # init state
            self.slstm_cell, lstm_state = self.build_lstm_cell(flow_batch_size, rnn_size=self.rnn_size//2)
            # max_len is the number of iter [actual seq + <end>]
            prev_h, lstm_state = self.lstm('slstm_cell', False, contexts, lstm_state)
            for i in range(0, self.max_len): 
                if i == 0:
                    x = concept_states
                else:
                    x = t_embed[:,i-1,:] 
                prev_h, lstm_state = self.lstm('slstm_cell', True, x, lstm_state)
                pred = self.decoder(prev_h, reuse=i!=0)
                ## summary attention maps
                # attention_maps = self.unsample_att(att_prob)
                # tf.summary.image('attention/att_train_t'+str(i), attention_maps)
                preds += [tf.expand_dims(pred, axis=1)]

        preds = tf.concat(preds,axis=1)

        return (preds, stop_preds)

    def build_test_model(self):
        """ Build the overall model
            Input:
                conv_feat [batch_size, self.conv_feat_h, self.conv_feat_w, self.conv_feat_dim]
                text [batch_size, self.num_feature, self.max_len]
        """
        with tf.name_scope('CNN_inputs'):
            # conv_feat = self._batch_norm(self.feed_conv_batch, name='conv_features')
            conv_feat = self.feed_conv_batch
            # tile the conv_feat for num_feature
            cam_embed = self.get_att_embedding(conv_feat, self.cls_logits)
            self.cam_att = cam_embed
            # cam_embed = tf.tile(cam_embed, multiples=[self.num_feature,1])
            # conv_feat = tf.tile(conv_feat, multiples=[self.num_feature,1,1,1])

            # compute image embedding
            i_embed = self.image_embedding(self.feed_conv_batch)
        
        # compute text embedding
        flow_batch_size = tf.shape(self.feed_txt_batch)[0]  # get batch size 
        
        concept_states = []
        stop_preds = []
        att_probs = []
        contexts = []
        with tf.variable_scope('CLSTM'):
            # concept lstm
            lstm_state = self.clstm_cell.zero_state(
                                batch_size=flow_batch_size//self.num_feature, dtype='float32')
            for i in range(self.num_feature):
                # the first is image embedding
                prev_h, lstm_state = self.lstm('clstm_cell', True, i_embed, lstm_state)
                context, att_prob = self.attention(conv_feat, cam_embed, prev_h, id=str(i+1), reuse=True)
                pred = self.continue_indictor(prev_h, reuse=True)
                
                contexts.append(tf.expand_dims(context, axis=0))
                concept_states.append(tf.expand_dims(prev_h, axis=0)) # put batch at the second dimension
                stop_preds.append(tf.expand_dims(pred, axis=1))
                att_probs += [tf.expand_dims(att_prob, axis=1)]

        stop_preds = tf.concat(stop_preds, axis=1)
        concept_states = tf.concat(concept_states, axis=0)
        contexts = tf.concat(contexts, axis=0)
        self.semantic_knowledge = tf.concat([concept_states, contexts], axis=2)

        concept_states = tf.reshape(concept_states, shape=[-1, concept_states.shape[2]]) #[bum_feature*batch_size, rnn_size]
        contexts = tf.reshape(contexts, shape=[-1, contexts.shape[2]]) #[bum_feature*batch_size, rnn_size]

        with tf.variable_scope('SLSTM'):
            preds = []
            # init state
            lstm_state = self.slstm_cell.zero_state(
                                batch_size=flow_batch_size, dtype='float32')
            # max_len is the number of iter [actual seq + <end>]
            prev_h, lstm_state = self.lstm('slstm_cell', True, contexts, lstm_state)
            for i in range(0, self.max_len): # 0 is topic model so it is actually starts from 1
                if i == 0:
                    x = concept_states
                else:
                    x = self.word_embeddings(last_pred)
                prev_h, lstm_state = self.lstm('slstm_cell', True, x, lstm_state)
                # get argmax
                pred_logit = self.decoder(prev_h)
                if self.sample_max:
                    last_pred = tf.argmax(pred_logit, axis=1)
                else:
                    temp_pred_logit = tf.exp(tf.div(pred_logit, self.tempature))
                    last_pred = tf.squeeze(tf.multinomial(temp_pred_logit, 1), axis=1)

                # attention_maps = self.unsample_att(att_prob)
                preds += [tf.expand_dims(last_pred, axis=1)]
                
        
        preds = tf.to_float(tf.concat(preds, axis=1))
        self.att_probs = tf.concat(att_probs, axis=1)
        if not self.sample_max:
            print ('\t --> sampling using multinomial distribution with tempature {}'.format(self.tempature))

        stops = tf.cast(tf.argmax(stop_preds, 2), tf.int32) # [batch_size, num_feat]

        return (preds, stops)  

    def build_model(self):

        train_preds = self.build_train_model()
        # tf.get_variable_scope().reuse_variables()
        test_preds = self.build_test_model()
        # preds = tf.cond(self.is_train_mode, self.build_train_model, self.build_test_model)
        return train_preds, test_preds
    
    def get_att_embedding(self, conv_maps, cls_logist):
        """Get class attention map 
            Output:
                att_embed: [batch_size, conv_feat_spat]

        """
        with tf.name_scope('att_embedding'):
            # select weight based on largest class reponse
            cls_preds = tf.cast(tf.argmax(cls_logist, 1), tf.int32)
            cls_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.cls_weight_op_name)[0]
            sample_w = tf.gather(tf.transpose(cls_weights), indices=cls_preds)
            flat_conv = tf.reshape(conv_maps, shape=[-1, self.conv_feat_spat, self.conv_feat_dim])
            cam_embed = tf.reduce_sum(flat_conv * tf.expand_dims(sample_w, axis=1), axis=2, name='cam_embed')
     
        return cam_embed