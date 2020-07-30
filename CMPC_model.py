import numpy as np
import tensorflow as tf
import sys
from deeplab_resnet import model as deeplab101
from util.cell import ConvLSTMCell

from util import data_reader
from util.processing_tools import *
from util import im_processing, text_processing, eval_tools
from util import loss


class LSTM_model(object):

    def __init__(self, batch_size=1,
                 num_steps=20,
                 vf_h=40,
                 vf_w=40,
                 H=320,
                 W=320,
                 vf_dim=2048,
                 vocab_size=12112,
                 w_emb_dim=1000,
                 v_emb_dim=1000,
                 mlp_dim=500,
                 start_lr=0.00025,
                 lr_decay_step=800000,
                 lr_decay_rate=1.0,
                 rnn_size=1000,
                 keep_prob_rnn=1.0,
                 keep_prob_emb=1.0,
                 keep_prob_mlp=1.0,
                 num_rnn_layers=1,
                 optimizer='adam',
                 weight_decay=0.0005,
                 mode='eval',
                 conv5=False,
                 glove_dim=300,
                 emb_name='Gref'):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.vf_h = vf_h
        self.vf_w = vf_w
        self.H = H
        self.W = W
        self.vf_dim = vf_dim
        self.start_lr = start_lr
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.vocab_size = vocab_size
        self.w_emb_dim = w_emb_dim
        self.v_emb_dim = v_emb_dim
        self.glove_dim = glove_dim
        self.emb_name = emb_name
        self.mlp_dim = mlp_dim
        self.rnn_size = rnn_size
        self.keep_prob_rnn = keep_prob_rnn
        self.keep_prob_emb = keep_prob_emb
        self.keep_prob_mlp = keep_prob_mlp
        self.num_rnn_layers = num_rnn_layers
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.mode = mode
        self.conv5 = conv5

        self.words = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.im = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 3])
        self.target_fine = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 1])
        self.valid_idx = tf.placeholder(tf.int32, [self.batch_size, 1])

        resmodel = deeplab101.DeepLabResNetModel({'data': self.im}, is_training=False)
        self.visual_feat_c5 = resmodel.layers['res5c_relu']
        self.visual_feat_c4 = resmodel.layers['res4b22_relu']
        self.visual_feat_c3 = resmodel.layers['res3b3_relu']

        # GloVe Embedding
        glove_np = np.load('data/{}_emb.npy'.format(self.emb_name))
        print("Loaded embedding npy at data/{}_emb.npy".format(self.emb_name))
        self.glove = tf.convert_to_tensor(glove_np, tf.float32)  # [vocab_size, 400]

        with tf.variable_scope("text_objseg"):
            self.build_graph()
            if self.mode == 'eval':
                return
            self.train_op()

    def build_graph(self):
        print("\n")
        print("#" * 30)
        print("Mutan_RAGR_p345_glove_gvec_validlang_2stage_4loss, \n"
              "spatial graph = vis_la_sp, then gcn, \n"
              "adj matrix in gcn is obtained by [HW, T] x [T, HW]. \n"
              "words_parse: [entity, attribute, relation, unnecessary]. \n"
              "Multi-modal feature is obtained by mutan fusion without dropout. \n"
              "The valid language feature is obtained by [E, A]. \n"
              "adj_mat * relation. \n"
              "Fuse p345 with gvec_validlang as filters and validlang obtained by [E, A, R]\n"
              "Exchange features for two times. \n"
              "4 losses are used to optimize. \n"
              "Glove Embedding is used to initilize embedding layer.")
        print("#" * 30)
        print("\n")

        embedding_mat = tf.Variable(self.glove)
        embedded_seq = tf.nn.embedding_lookup(embedding_mat, tf.transpose(self.words))  # [num_step, batch_size, glove_emb]
        print("Build Glove Embedding.")

        rnn_cell_basic = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=False)
        if self.mode == 'train' and self.keep_prob_rnn < 1:
            rnn_cell_basic = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_basic, output_keep_prob=self.keep_prob_rnn)
        cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell_basic] * self.num_rnn_layers, state_is_tuple=False)

        state = cell.zero_state(self.batch_size, tf.float32)
        state_shape = state.get_shape().as_list()
        state_shape[0] = self.batch_size
        state.set_shape(state_shape)

        words_feat_list = []

        def f1():
            # return tf.constant(0.), state
            return tf.zeros([self.batch_size, self.rnn_size]), state

        def f2():
            # Word input to embedding layer
            w_emb = embedded_seq[n, :, :]
            if self.mode == 'train' and self.keep_prob_emb < 1:
                w_emb = tf.nn.dropout(w_emb, self.keep_prob_emb)
            return cell(w_emb, state)

        with tf.variable_scope("RNN"):
            for n in range(self.num_steps):
                if n > 0:
                    tf.get_variable_scope().reuse_variables()

                # rnn_output, state = cell(w_emb, state)
                rnn_output, state = tf.cond(tf.equal(self.words[0, n], tf.constant(0)), f1, f2)
                word_feat = tf.reshape(rnn_output, [self.batch_size, 1, self.rnn_size])
                words_feat_list.append(word_feat)

        lang_feat = tf.reshape(rnn_output, [self.batch_size, 1, 1, self.rnn_size])
        lang_feat = tf.nn.l2_normalize(lang_feat, 3)

        # words_feat: [B, num_steps, rnn_size]
        words_feat = tf.concat(words_feat_list, 1)
        words_feat = tf.slice(words_feat, [0, self.valid_idx[0, 0], 0],
                              [-1, self.num_steps - self.valid_idx[0, 0], -1])
        words_feat = tf.nn.l2_normalize(words_feat, 2)
        # words_feat: [B, 1, num_words, rnn_size]
        words_feat = tf.expand_dims(words_feat, 1)

        visual_feat_c5 = self._conv("c5_lateral", self.visual_feat_c5, 1, self.vf_dim, self.v_emb_dim, [1, 1, 1, 1])
        visual_feat_c5 = tf.nn.l2_normalize(visual_feat_c5, 3)
        visual_feat_c4 = self._conv("c4_lateral", self.visual_feat_c4, 1, 1024, self.v_emb_dim, [1, 1, 1, 1])
        visual_feat_c4 = tf.nn.l2_normalize(visual_feat_c4, 3)
        visual_feat_c3 = self._conv("c3_lateral", self.visual_feat_c3, 1, 512, self.v_emb_dim, [1, 1, 1, 1])
        visual_feat_c3 = tf.nn.l2_normalize(visual_feat_c3, 3)

        # Generate spatial grid
        spatial = tf.convert_to_tensor(generate_spatial_batch(self.batch_size, self.vf_h, self.vf_w))

        words_parse = self.build_lang_parser(words_feat)

        fusion_c5 = self.build_lang2vis(visual_feat_c5, words_feat, lang_feat,
                                        words_parse, spatial, level="c5")
        fusion_c4 = self.build_lang2vis(visual_feat_c4, words_feat, lang_feat,
                                        words_parse, spatial, level="c4")
        fusion_c3 = self.build_lang2vis(visual_feat_c3, words_feat, lang_feat,
                                        words_parse, spatial, level="c3")

        # For multi-level losses
        score_c5 = self._conv("score_c5", fusion_c5, 3, self.mlp_dim, 1, [1, 1, 1, 1])
        self.up_c5 = tf.image.resize_bilinear(score_c5, [self.H, self.W])
        score_c4 = self._conv("score_c4", fusion_c4, 3, self.mlp_dim, 1, [1, 1, 1, 1])
        self.up_c4 = tf.image.resize_bilinear(score_c4, [self.H, self.W])
        score_c3 = self._conv("score_c3", fusion_c3, 3, self.mlp_dim, 1, [1, 1, 1, 1])
        self.up_c3 = tf.image.resize_bilinear(score_c3, [self.H, self.W])

        valid_lang = self.nec_lang(words_parse, words_feat)
        fused_feats = self.gated_exchange_fusion_lstm_2times(fusion_c3,
                                                             fusion_c4, fusion_c5, valid_lang)
        score = self._conv("score", fused_feats, 3, self.mlp_dim, 1, [1, 1, 1, 1])

        self.pred = score
        self.up = tf.image.resize_bilinear(self.pred, [self.H, self.W])
        self.sigm = tf.sigmoid(self.up)

    def valid_lang(self, words_parse, words_feat):
        # words_parse: [B, 1, T, 4]
        words_parse_sum = tf.reduce_sum(words_parse, 3)
        words_parse_valid = words_parse[:, :, :, 0] + words_parse[:, :, :, 1]
        # words_parse_valid: [B, 1, T]
        words_feat_reshaped = tf.reshape(words_feat, [self.batch_size, self.num_steps - self.valid_idx[0, 0], self.rnn_size])
        # words_feat_reshaped: [B, T, C]
        valid_lang_feat = tf.matmul(words_parse_valid, words_feat_reshaped)
        # valid_lang_feat: [B, 1, C]
        valid_lang_feat = tf.nn.l2_normalize(valid_lang_feat, 2)
        valid_lang_feat = tf.reshape(valid_lang_feat, [self.batch_size, 1, 1, self.rnn_size])
        # [B, 1, 1, rnn_size]
        return valid_lang_feat

    def nec_lang(self, words_parse, words_feat):
        # words_parse: [B, 1, T, 4]
        words_parse_sum = tf.reduce_sum(words_parse, 3)
        words_parse_valid = words_parse_sum - words_parse[:, :, :, 3]
        # words_parse_valid: [B, 1, T]
        words_feat_reshaped = tf.reshape(words_feat, [self.batch_size, self.num_steps - self.valid_idx[0, 0], self.rnn_size])
        # words_feat_reshaped: [B, T, C]
        valid_lang_feat = tf.matmul(words_parse_valid, words_feat_reshaped)
        # valid_lang_feat: [B, 1, C]
        valid_lang_feat = tf.nn.l2_normalize(valid_lang_feat, 2)
        valid_lang_feat = tf.reshape(valid_lang_feat, [self.batch_size, 1, 1, self.rnn_size])
        # [B, 1, 1, rnn_size]
        return valid_lang_feat

    def lang_se(self, feat, lang_feat, level=""):
        '''
        Using lang feat as channel filter to select correlated features of feat.
        Just like Squeeze-and-Excite.
        :param feat: [B, 1, 1, C]
        :param lang_feat: [B, H, W, C]
        :return: feat': [B, H, W, C]
        '''
        lang_feat_trans = self._conv("lang_feat_{}".format(level),
                                     lang_feat, 1, self.mlp_dim, self.mlp_dim, [1, 1, 1, 1])  # [B, 1, 1, C]
        lang_feat_trans = tf.sigmoid(lang_feat_trans)
        feat_trans = self._conv("trans_feat_{}".format(level),
                                feat, 1, self.mlp_dim, self.mlp_dim, [1, 1, 1, 1])  # [B, H, W, C]
        feat_trans = tf.nn.relu(feat_trans)
        # use lang feat as a channel filter
        feat_trans = feat_trans * lang_feat_trans  # [B, H, W, C]
        return feat_trans

    def global_vec(self, feat, lang_feat, level=""):
        '''
        Get the global vector by adaptive avg pooling for feat.
        Pooling matrix is obtained by attention mechanism with lang feat.
        :param feat: [B, H, W, mlp_dim]
        :param lang_feat: [B, H, W, rnn_size]
        :param level
        :return: gv_lang: [B, 1, 1, mlp_dim]
        '''
        feat_key = self._conv("spa_graph_key_{}".format(level), feat, 1, self.mlp_dim, self.mlp_dim, [1, 1, 1, 1])
        feat_key = tf.reshape(feat_key, [self.batch_size, self.vf_h * self.vf_w, self.mlp_dim])  # [B, HW, C]
        lang_query = self._conv("lang_query_{}".format(level), lang_feat, 1, self.rnn_size, self.mlp_dim, [1, 1, 1, 1])
        lang_query = tf.reshape(lang_query, [self.batch_size, 1, self.mlp_dim])  # [B, 1, C]

        attn_map = tf.matmul(feat_key, lang_query, transpose_b=True)  # [B, HW, 1]
        # Normalization for affinity matrix
        attn_map = tf.divide(attn_map, self.mlp_dim ** 0.5)
        attn_map = tf.nn.softmax(attn_map, axis=1)
        # attn_map: [B, HW, 1]

        feat_reshaped = tf.reshape(feat, [self.batch_size, self.vf_h * self.vf_w, self.mlp_dim])
        # feat_reshaped: [B, HW, C]
        # Adaptive global average pooling
        gv_pooled = tf.matmul(attn_map, feat_reshaped, transpose_a=True)  # [B, 1, C]
        gv_pooled = tf.reshape(gv_pooled, [self.batch_size, 1, 1, self.mlp_dim])  # [B, 1, 1, C]

        gv_lang = tf.concat([gv_pooled, lang_feat], 3)  # [B, 1, 1, 3C]
        gv_lang = self._conv("gv_lang_{}".format(level), gv_lang, 1, self.mlp_dim + self.rnn_size, self.mlp_dim,
                             [1, 1, 1, 1])  # [B, 1, 1, C]
        gv_lang = tf.nn.l2_normalize(gv_lang)
        print("Build Global Lang Vec")
        return gv_lang

    def gated_exchange_module(self, feat, feat1, feat2, lang_feat, level=""):
        '''
        Exchange information of feat1 and feat2 with feat, using sentence feature
        as guidance.
        :param feat: [B, H, W, C]
        :param feat1: [B, H, W, C]
        :param feat2: [B, H, W, C]
        :param lang_feat: [B, 1, 1, C]
        :return: feat', [B, H, W, C]
        '''
        gv_lang = self.global_vec(feat, lang_feat, level + 'gv_f1')  # [B, 1, 1, C]
        feat1 = self.lang_se(feat1, gv_lang, level + '_f1')
        feat2 = self.lang_se(feat2, gv_lang, level + '_f2')
        feat_exg = feat + feat1 + feat2
        return feat_exg

    def gated_exchange_fusion_lstm_2times(self, feat3, feat4, feat5, lang_feat):
        '''
        Fuse exchanged features of level3, level4, level5
        LSTM is used to fuse the exchanged features
        :param feat3: [B, H, W, C]
        :param feat4: [B, H, W, C]
        :param feat5: [B, H, W, C]
        :param lang_feat: [B, 1, 1, C]
        :return: fused feat3, feat4, feat5
        '''
        feat_exg3 = self.gated_exchange_module(feat3, feat4, feat5, lang_feat, 'c3')
        feat_exg3 = tf.nn.l2_normalize(feat_exg3, 3)
        feat_exg4 = self.gated_exchange_module(feat4, feat3, feat5, lang_feat, 'c4')
        feat_exg4 = tf.nn.l2_normalize(feat_exg4, 3)
        feat_exg5 = self.gated_exchange_module(feat5, feat3, feat4, lang_feat, 'c5')
        feat_exg5 = tf.nn.l2_normalize(feat_exg5, 3)

        # Second time
        feat_exg3_2 = self.gated_exchange_module(feat_exg3, feat_exg4, feat_exg5, lang_feat, 'c3_2')
        feat_exg3_2 = tf.nn.l2_normalize(feat_exg3_2, 3)
        feat_exg4_2 = self.gated_exchange_module(feat_exg4, feat_exg3, feat_exg5, lang_feat, 'c4_2')
        feat_exg4_2 = tf.nn.l2_normalize(feat_exg4_2, 3)
        feat_exg5_2 = self.gated_exchange_module(feat_exg5, feat_exg3, feat_exg4, lang_feat, 'c5_2')
        feat_exg5_2 = tf.nn.l2_normalize(feat_exg5_2, 3)

        # Convolutional LSTM Fuse
        convlstm_cell = ConvLSTMCell([self.vf_h, self.vf_w], self.mlp_dim, [1, 1])
        convlstm_outputs, states = tf.nn.dynamic_rnn(convlstm_cell, tf.convert_to_tensor(
            [[feat_exg3_2[0], feat_exg4_2[0], feat_exg5_2[0]]]), dtype=tf.float32)
        fused_feat = convlstm_outputs[:, -1]
        print("Build Gated Fusion with ConvLSTM two times.")

        return fused_feat

    def mutan_head(self, lang_feat, spatial_feat, visual_feat, level=''):
        # visual feature transform
        vis_trans = tf.concat([visual_feat, spatial_feat], 3)   # [B, H, W, C+8]
        vis_trans = self._conv("vis_trans_{}".format(level), vis_trans, 1,
                               self.v_emb_dim+8, self.v_emb_dim, [1, 1, 1, 1])
        vis_trans = tf.nn.tanh(vis_trans)  # [B, H, W, C]

        # lang feature transform
        lang_trans = self._conv("lang_trans_{}".format(level), lang_feat,
                                1, self.rnn_size, self.v_emb_dim, [1, 1, 1, 1])

        lang_trans = tf.nn.tanh(lang_trans)  # [B, 1, 1, C]

        mutan_feat = vis_trans * lang_trans  # [B, H, W, C]
        return mutan_feat

    def mutan_fusion(self, lang_feat, spatial_feat, visual_feat, level=''):
        # fuse language feature and visual feature
        # lang_feat: [B, 1, 1, C], visual_feat: [B, H, W, C], spatial_feat: [B, H, W, 8]
        # output: [B, H, W, C']
        head1 = self.mutan_head(lang_feat, spatial_feat, visual_feat, '{}_head1'.format(level))
        head2 = self.mutan_head(lang_feat, spatial_feat, visual_feat, '{}_head2'.format(level))
        head3 = self.mutan_head(lang_feat, spatial_feat, visual_feat, '{}_head3'.format(level))
        head4 = self.mutan_head(lang_feat, spatial_feat, visual_feat, '{}_head4'.format(level))
        head5 = self.mutan_head(lang_feat, spatial_feat, visual_feat, '{}_head5'.format(level))

        fused_feats = tf.stack([head1, head2, head3, head4, head5], axis=4)  # [B, H, W, C, 5]
        fused_feats = tf.reduce_sum(fused_feats, 4)  # [B, H, W, C]
        fused_feats = tf.nn.tanh(fused_feats)
        fused_feats = tf.nn.l2_normalize(fused_feats, 3)

        print("Build Mutan Fusion Module.")

        return fused_feats

    def build_lang2vis(self, visual_feat, words_feat, lang_feat, words_parse, spatial, level=""):
        valid_lang_feat = self.valid_lang(words_parse, words_feat)
        vis_la_sp = self.mutan_fusion(valid_lang_feat, spatial, visual_feat, level=level)
        print("Build MutanFusion Module to get multi-modal features.")
        spa_graph_feat = self.build_spa_graph(vis_la_sp, words_feat, spatial,
                                              words_parse, level=level)
        print("Build Lang2Vis Module.")

        lang_vis_feat = tf.tile(valid_lang_feat, [1, self.vf_h, self.vf_w, 1])  # [B, H, W, C]
        feat_all = tf.concat([vis_la_sp, spa_graph_feat, lang_vis_feat, spatial], 3)
        # Feature fusion
        fusion = self._conv("fusion_{}".format(level), feat_all, 1,
                            self.v_emb_dim * 2 + self.rnn_size + 8,
                            self.mlp_dim, [1, 1, 1, 1])
        fusion = tf.nn.relu(fusion)
        return fusion

    def build_lang_parser(self, words_feat):
        # Language Attention
        words_parse = self._conv("words_parse_1", words_feat, 1, self.rnn_size, 500, [1, 1, 1, 1])
        words_parse = tf.nn.relu(words_parse)
        words_parse = self._conv("words_parse_2", words_parse, 1, 500, 4, [1, 1, 1, 1])
        words_parse = tf.nn.softmax(words_parse, axis=3)
        # words_parse: [B, 1, T, 4]
        # Four weights: Entity, Attribute, Relation, Unnecessary
        return words_parse

    def graph_conv(self, graph_feat, nodes_num, nodes_dim, adj_mat, graph_name="", level=""):
        # Node message passing
        graph_feat_reshaped = tf.reshape(graph_feat, [self.batch_size, nodes_num, nodes_dim])
        gconv_feat = tf.matmul(adj_mat, graph_feat_reshaped)  # [B, nodes_num, nodes_dim]
        gconv_feat = tf.reshape(gconv_feat, [self.batch_size, 1, nodes_num, nodes_dim])
        gconv_feat = tf.contrib.layers.layer_norm(gconv_feat,
                                                  scope="gconv_feat_ln_{}_{}".format(graph_name, level))
        gconv_feat = graph_feat + gconv_feat
        gconv_feat = tf.nn.relu(gconv_feat)  # [B, 1, nodes_num, nodes_dim]
        gconv_update = self._conv("gconv_update_{}_{}".format(graph_name, level),
                                       gconv_feat, 1, nodes_dim, nodes_dim, [1, 1, 1, 1])
        gconv_update = tf.contrib.layers.layer_norm(gconv_update,
                                                         scope="gconv_update_ln_{}_{}".format(graph_name, level))
        gconv_update = tf.nn.relu(gconv_update)

        return gconv_update

    def build_spa_graph(self, spa_graph, words_feat, spatial, words_parse, level=""):
        # Fuse visual_feat, lang_attn_feat and spatial for SGR
        words_trans = self._conv("words_trans_{}".format(level), words_feat, 1, self.rnn_size, self.rnn_size,
                                 [1, 1, 1, 1])
        words_trans = tf.reshape(words_trans, [self.batch_size, self.num_steps - self.valid_idx[0, 0], self.rnn_size])
        spa_graph_trans2 = self._conv("spa_graph_trans2_{}".format(level), spa_graph, 1, self.v_emb_dim, self.v_emb_dim,
                                     [1, 1, 1, 1])
        spa_graph_trans2 = tf.reshape(spa_graph_trans2, [self.batch_size, self.vf_h * self.vf_w, self.v_emb_dim])
        graph_words_affi = tf.matmul(spa_graph_trans2, words_trans, transpose_b=True)
        # Normalization for affinity matrix
        graph_words_affi = tf.divide(graph_words_affi, self.v_emb_dim ** 0.5)
        # graph_words_affi: [B, HW, T]
        graph_words_affi = words_parse[:, :, :, 2] * graph_words_affi
        gw_affi_w = tf.nn.softmax(graph_words_affi, axis=2)
        gw_affi_v = tf.nn.softmax(graph_words_affi, axis=1)
        adj_mat = tf.matmul(gw_affi_w, gw_affi_v, transpose_b=True)
        # adj_mat: [B, HW, HW], sum == 1 on axis 2

        spa_graph_nodes_num = self.vf_h * self.vf_w
        spa_graph = tf.reshape(spa_graph, [self.batch_size, 1, spa_graph_nodes_num, self.v_emb_dim])
        spa_graph = self.graph_conv(spa_graph, spa_graph_nodes_num, self.v_emb_dim, adj_mat,
                                    graph_name="spa_graph", level=level)
        spa_graph = tf.reshape(spa_graph, [self.batch_size, self.vf_h, self.vf_w, self.v_emb_dim])
        spa_graph = tf.nn.l2_normalize(spa_graph, 3)

        return spa_graph

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.conv2d(x, w, strides, padding='SAME') + b

    def _atrous_conv(self, name, x, filter_size, in_filters, out_filters, rate):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
                                initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.atrous_conv2d(x, w, rate=rate, padding='SAME') + b

    def train_op(self):
        if self.conv5:
            tvars = [var for var in tf.trainable_variables() if var.op.name.startswith('text_objseg')
                     or var.name.startswith('res5') or var.name.startswith('res4')
                     or var.name.startswith('res3')]
        else:
            tvars = [var for var in tf.trainable_variables() if var.op.name.startswith('text_objseg')]
        reg_var_list = [var for var in tvars if var.op.name.find(r'DW') > 0 or var.name[-9:-2] == 'weights']
        print('Collecting variables for regularization:')
        for var in reg_var_list: print('\t%s' % var.name)
        print('Done.')

        # define loss
        self.target = tf.image.resize_bilinear(self.target_fine, [self.vf_h, self.vf_w])
        self.cls_loss_c5 = loss.weighed_logistic_loss(self.up_c5, self.target_fine, 1, 1)
        self.cls_loss_c4 = loss.weighed_logistic_loss(self.up_c4, self.target_fine, 1, 1)
        self.cls_loss_c3 = loss.weighed_logistic_loss(self.up_c3, self.target_fine, 1, 1)
        self.cls_loss = loss.weighed_logistic_loss(self.up, self.target_fine, 1, 1)
        self.cls_loss_all = 0.7 * self.cls_loss + 0.1 * self.cls_loss_c5 \
                            + 0.1 * self.cls_loss_c4 + 0.1 * self.cls_loss_c3
        self.reg_loss = loss.l2_regularization_loss(reg_var_list, self.weight_decay)
        self.cost = self.cls_loss_all + self.reg_loss

        # learning rate
        lr = tf.Variable(0.0, trainable=False)
        self.learning_rate = tf.train.polynomial_decay(self.start_lr, lr, self.lr_decay_step, end_learning_rate=0.00001,
                                                       power=0.9)

        # optimizer
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError("Unknown optimizer type %s!" % self.optimizer)

        # learning rate multiplier
        grads_and_vars = optimizer.compute_gradients(self.cost, var_list=tvars)
        var_lr_mult = {}
        for var in tvars:
            if var.op.name.find(r'biases') > 0:
                var_lr_mult[var] = 2.0
            elif var.name.startswith('res5') or var.name.startswith('res4') or var.name.startswith('res3'):
                var_lr_mult[var] = 1.0
            else:
                var_lr_mult[var] = 1.0
        print('Variable learning rate multiplication:')
        for var in tvars:
            print('\t%s: %f' % (var.name, var_lr_mult[var]))
        print('Done.')
        grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v) for g, v in
                          grads_and_vars]

        # training step
        self.train_step = optimizer.apply_gradients(grads_and_vars, global_step=lr)
