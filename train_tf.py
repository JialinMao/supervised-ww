import tensorflow as tf

def build_dqn(screen_width, screen_height, action_size):
    w = {}
    t_w = {}

    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    # training network
    with tf.variable_scope('prediction'):

        s_t = tf.placeholder('float32',
                [None, 1, screen_width, screen_height], name='s_t')
        
        l1, w['l1_w'], w['l1_b'] = conv2d(s_t, 
                4, [2, 2], [1, 1], initializer, activation_fn, 'NCHW', name='l1' )
        l2, w['l2_w'], w['l2_b'] = conv2d(l1, 
                4, [2, 2], [1, 1], initializer, activation_fn, 'NCHW', name='l2' )
        l3, w['l3_w'], w['l3_b'] = conv2d(l2, 
                4, [1, 1], [1, 1], initializer, activation_fn, 'NCHW', name='l3' )

        shape = l3.get_shape().as_list()
        l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
        l4, w['l4_w'], w['l4_b'] = linear(l3_flat, 8, activation_fn=activation_fn, name='l4')
        q, w['q_w'], w['q_b'] = linear(l4, action_size, name='q')
        q_softmax = tf.nn.softmax(q)
        q_action = tf.argmax(q_softmax, dimension=1)
        

    # optimizer
    with tf.variable_scope('optimizer'):
        target_q_t = tf.placeholder('float32', [None, env.action_size], name='target_q_t')

        loss = tf.reduce_mean(-tf.reduce_sum(target_q_t * tf.log(q_softmax), reduction_indices=[1]), name='loss')

        learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
        learning_rate_op = tf.maximum(learning_rate_minimum,
                tf.train.exponential_decay(
                        learning_rate,
                        learning_rate_step,
                        learning_rate_decay_step,
                        learning_rate_decay,
                        staircase=True))
        optim = tf.train.RMSPropOptimizer(
                learning_rate_op, momentum=0.95, epsilon=0.01).minimize(loss)

    with tf.variable_scope('summary'):
        scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
                'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

        summary_placeholders = {}
        summary_ops = {}

        for tag in scalar_summary_tags:
            summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
            summary_ops[tag]  = tf.scalar_summary("%s-%s/%s" % (env_name, env_type, tag), summary_placeholders[tag])

        histogram_summary_tags = ['episode.rewards', 'episode.actions']

        for tag in histogram_summary_tags:
            summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
            summary_ops[tag]  = tf.histogram_summary(tag, summary_placeholders[tag])

        writer = tf.train.SummaryWriter('./logs/%s' % model_dir, sess.graph)

    tf.initialize_all_variables().run()

    _saver = tf.train.Saver(w.values() + [step_op], max_to_keep=30)

    load_model()
    update_target_q_network()


