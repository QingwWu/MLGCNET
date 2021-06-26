import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from model import GCNModel
from opt import Optimizer



def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    ey=np.eye(adj.shape[0])
    adj=adj+ey
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_nomalized = adj_.dot(
        degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_nomalized = adj_nomalized.tocoo()
    return sparse_to_tuple(adj_nomalized)

def constructNet(lnc_dis_matrix):
    lnc_matrix = np.matrix(
        np.zeros((lnc_dis_matrix.shape[0], lnc_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((lnc_dis_matrix.shape[1], lnc_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((lnc_matrix, lnc_dis_matrix))
    mat2 = np.hstack((lnc_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj

def constructHNet(lnc_dis_matrix, lnc_matrix, dis_matrix):
    mat1 = np.hstack((lnc_matrix, lnc_dis_matrix))
    mat2 = np.hstack((lnc_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))

def calemb(train_lnc_dis_matrix, lnc_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_lnc_dis_matrix, lnc_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_lnc_dis_matrix.sum()
    X = constructNet(train_lnc_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_lnc_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_lnc_dis_matrix.shape[0], name='GCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_lnc_dis_matrix.shape[0], num_v=train_lnc_dis_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    embed=sess.run(model.embeddings, feed_dict=feed_dict)
    sess.close()
    return embed