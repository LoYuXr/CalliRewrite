import tensorflow as tf

#############################################
# Common parameters
#############################################

FLAGS = tf.compat.v1.app.flags.FLAGS

tf.compat.v1.app.flags.DEFINE_string(
    'dataset_dir',
    'datasets',
    'The directory of sketch data of the dataset.')
tf.compat.v1.app.flags.DEFINE_string(
    'log_root',
    'outputs/log',
    'Directory to store tensorboard.')
tf.compat.v1.app.flags.DEFINE_string(
    'log_img_root',
    'outputs/log_img',
    'Directory to store intermediate output images.')
tf.compat.v1.app.flags.DEFINE_string(
    'snapshot_root',
    'outputs/snapshot',
    'Directory to store model checkpoints.')
tf.compat.v1.app.flags.DEFINE_string(
    'neural_renderer_path',
    'outputs/snapshot/pretrain_neural_renderer/renderer_300000.tfmodel',
    'Path to the neural renderer model.')
tf.compat.v1.app.flags.DEFINE_string(
    'perceptual_model_root',
    'outputs/snapshot/pretrain_perceptual_model',
    'Directory to store perceptual model.')
tf.compat.v1.app.flags.DEFINE_string(
    'data',
    '',
    'The dataset type.')


def get_default_hparams_phase_1():
    """Return default HParams for sketch-rnn."""
    hparams = dict(
        program_name='new_train_phase_1',
        data_set='clean_line_drawings',  # Our dataset.

        input_channel=1,

        num_steps=90040,  # Total number of steps of training.
        save_every=15000,
        eval_every=5000,

        max_seq_len=48,
        batch_size=12,
        gpus=[0],
        loop_per_gpu=1,

        sn_loss_type='increasing',  # ['decreasing', 'fixed', 'increasing']
        stroke_num_loss_weight=0.5,
        stroke_num_loss_weight_end=0.0,
        increase_start_steps=0,
        decrease_stop_steps=40000,

        perc_loss_layers=['ReLU1_2', 'ReLU2_2', 'ReLU3_3', 'ReLU5_1'],
        perc_loss_fuse_type='add',  # ['max', 'add', 'raw_add', 'weighted_sum']

        init_cursor_on_undrawn_pixel=False,

        early_pen_loss_type='move',  # ['head', 'tail', 'move']
        early_pen_loss_weight=0.1,
        early_pen_length=7,

        min_width=0.01,
        min_window_size=32,
        max_scaling=2.0,

        encode_cursor_type='value',

        image_size_small=128,
        image_size_large=278,

        cropping_type='v3',  # ['v2', 'v3']
        pasting_type='v3',  # ['v2', 'v3']
        pasting_diff=True,

        concat_win_size=True,

        encoder_type='conv13_c3',
        # ['conv10', 'conv10_deep', 'conv13', 'conv10_c3', 'conv10_deep_c3', 'conv13_c3']
        # ['conv13_c3_attn']
        # ['combine33', 'combine43', 'combine53', 'combineFC']
        vary_thickness=False,

        outside_loss_weight=10.0,
        win_size_outside_loss_weight=10.0,

        resize_method='AREA',  # ['BILINEAR', 'NEAREST_NEIGHBOR', 'BICUBIC', 'AREA']

        concat_cursor=True,

        use_softargmax=True,
        soft_beta=10,  # value for the soft argmax

        raster_loss_weight=1.0,

        dec_rnn_size=256,  # Size of decoder.
        dec_model='hyper',  # Decoder: lstm, layer_norm or hyper.
        # z_size=128,  # Size of latent vector z. Recommend 32, 64 or 128.
        bin_gt=True,

        stop_accu_grad=True,

        random_cursor=True,
        cursor_type='next',

        raster_size=128,

        pix_drop_kp=1.0,  # Dropout keep rate
        add_coordconv=True,
        position_format='abs',
        raster_loss_base_type='perceptual',  # [l1, mse, perceptual]

        grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.

        learning_rate=0.0001,  # Learning rate.
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        decay_power=0.9,
        min_learning_rate=0.000001,  # Minimum learning rate.

        use_recurrent_dropout=True,  # Dropout with memory loss. Recommended
        recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
        use_input_dropout=False,  # Input dropout. Recommend leaving False.
        input_dropout_prob=0.90,  # Probability of input dropout keep.
        use_output_dropout=False,  # Output dropout. Recommend leaving False.
        output_dropout_prob=0.90,  # Probability of output dropout keep.

        smoothness_loss_weight=0.0,
        angle_loss_weight=0.0,

        model_mode='train'  # ['train', 'eval', 'sample']
    )
    return hparams


def get_default_hparams_phase_2():
    """Return default HParams for sketch-rnn."""
    hparams = dict(
        program_name='new_train_phase_2',
        data_set='gb',  # Our dataset.

        input_channel=1,

        num_steps=30020,  # Total number of steps of training.
        save_every=5000,
        eval_every=5000,

        max_seq_len=48,
        batch_size=12,
        gpus=[0],
        loop_per_gpu=1,

        sn_loss_type='fixed',  # ['decreasing', 'fixed', 'increasing']
        stroke_num_loss_weight=0.5,
        stroke_num_loss_weight_end=0.0,
        increase_start_steps=0,
        decrease_stop_steps=40000,

        perc_loss_layers=['ReLU1_2', 'ReLU2_2', 'ReLU3_3', 'ReLU5_1'],
        perc_loss_fuse_type='add',  # ['max', 'add', 'raw_add', 'weighted_sum']

        init_cursor_on_undrawn_pixel=False,

        early_pen_loss_type='move',  # ['head', 'tail', 'move']
        early_pen_loss_weight=0.1,
        early_pen_length=7,

        min_width=0.01,
        min_window_size=32,
        max_scaling=2.0,

        encode_cursor_type='value',

        image_size_small=128,
        image_size_large=256,

        cropping_type='v3',  # ['v2', 'v3']
        pasting_type='v3',  # ['v2', 'v3']
        pasting_diff=True,

        concat_win_size=True,

        encoder_type='conv13_c3',
        # ['conv10', 'conv10_deep', 'conv13', 'conv10_c3', 'conv10_deep_c3', 'conv13_c3']
        # ['conv13_c3_attn']
        # ['combine33', 'combine43', 'combine53', 'combineFC']
        vary_thickness=False,

        outside_loss_weight=10.0,
        win_size_outside_loss_weight=10.0,

        resize_method='AREA',  # ['BILINEAR', 'NEAREST_NEIGHBOR', 'BICUBIC', 'AREA']

        concat_cursor=True,

        use_softargmax=True,
        soft_beta=10,  # value for the soft argmax

        raster_loss_weight=1.0,

        dec_rnn_size=256,  # Size of decoder.
        dec_model='hyper',  # Decoder: lstm, layer_norm or hyper.
        # z_size=128,  # Size of latent vector z. Recommend 32, 64 or 128.
        bin_gt=True,

        stop_accu_grad=True,

        random_cursor=True,
        cursor_type='next',

        raster_size=128,

        pix_drop_kp=1.0,  # Dropout keep rate
        add_coordconv=True,
        position_format='abs',
        raster_loss_base_type='perceptual',  # [l1, mse, perceptual]

        grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.

        learning_rate=0.0001,  # Learning rate.
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        decay_power=0.9,
        min_learning_rate=0.000001,  # Minimum learning rate.

        use_recurrent_dropout=True,  # Dropout with memory loss. Recommended
        recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
        use_input_dropout=False,  # Input dropout. Recommend leaving False.
        input_dropout_prob=0.90,  # Probability of input dropout keep.
        use_output_dropout=False,  # Output dropout. Recommend leaving False.
        output_dropout_prob=0.90,  # Probability of output dropout keep.

        smoothness_loss_weight=0.5,
        angle_loss_weight=1.0,

        model_mode='train'  # ['train', 'eval', 'sample']
    )
    return hparams
