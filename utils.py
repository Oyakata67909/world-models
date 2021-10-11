import configargparse

PARSER = configargparse.ArgParser(default_config_files=['configs/doom.config'])
PARSER.add('-c', '--config_path', required=False,
        is_config_file=True, help='config file path')
PARSER.add('--exp_name', required=True, help='name of experiment')
PARSER.add('--env_name', required=True, help='name of environment')
PARSER.add('--dream_env', required=True, type=int,
        help='0 to train controller in real env 1 to train controller in dream')
PARSER.add('--max_frames', required=True, type=int,
        help='max number of frames in episode')
PARSER.add('--min_frames', required=True, type=int,
        help='min number of frames in episode')
PARSER.add('--max_trials', required=True,
        type=int, help='max number of trials')
PARSER.add_argument('--full_episode', dest='full_episode',
                action='store_true', help='ignore dones')
PARSER.add_argument('--no_full_episode', dest='full_episode',
                action='store_false', help='ignore dones')
PARSER.add_argument('--render_mode', dest='render_mode', action='store_true')
PARSER.add_argument('--no_render_mode', dest='render_mode',
                action='store_false')
PARSER.add('--exp_mode', required=True, help='defines controller architecture')
PARSER.add('--a_width', required=True, type=int, help='width of action vector')
PARSER.add('--z_size', required=True, type=int, help='z size')
PARSER.add('--state_space', required=True, type=int,
        help='1 to only include hidden state. 2 to include both h and c')

PARSER.add('--vae_batch_size', required=True,
        type=int, help='batch size for vae train')
PARSER.add('--vae_learning_rate', required=True,
        type=float, help='vae learning rate')
PARSER.add('--vae_kl_tolerance', required=True, type=float,
        help='vae kl tolerance for clipping')
PARSER.add('--vae_num_epoch', required=True, type=int,
        help='vae num epoch for training')

PARSER.add('--rnn_num_steps', required=True, type=int,
        help='number of rnn training steps')
PARSER.add('--rnn_max_seq_len', required=True, type=int,
        help='sequence lenght to train rnn on')
PARSER.add('--rnn_r_pred', required=True, type=int,
        help='predict reward if 1 dont if 0')
PARSER.add('--rnn_d_pred', required=True, type=int,
        help='predict done if 1 dont if 0')
PARSER.add('--rnn_input_seq_width', required=True,
        type=int, help='size of rnn input')
PARSER.add('--rnn_size', required=True, type=int,
        help='size of hidden and cell state')
PARSER.add('--rnn_batch_size', required=True,
        type=int, help='batch size rnn uses')
PARSER.add('--rnn_grad_clip', required=True, type=float,
        help='clip rnn gradients by value to this')
PARSER.add('--rnn_num_mixture', required=True, type=int,
        help='number of mixtures in MDNRNN')
PARSER.add('--rnn_learning_rate', required=True, type=float,
        help='initial learning rate used by the rnn')
PARSER.add('--rnn_decay_rate', required=True, type=float,
        help='decay rate for rnn learning rate')
PARSER.add('--rnn_min_learning_rate', required=True, type=float,
        help='prevent decaying rnn learning rate paste this threshold')
PARSER.add('--rnn_d_true_weight', required=True,
        type=float, help='weight on done loss when True')
PARSER.add('--rnn_temperature', required=True, type=float,
        help='temperature used when sampling the MDNRNN')

PARSER.add('--controller_optimizer', type=str,
        help='ses, pepg, openes, ga, cma.', default='cma')
PARSER.add('--controller_num_episode', type=int,
        default=16, help='num episodes per trial')
PARSER.add('--controller_num_test_episode', type=int, default=100,
        help='number of random episodes to evaluate agent on')
PARSER.add('--controller_eval_steps', type=int, default=25,
        help='evaluate every eval_steps step')
PARSER.add('--controller_num_worker', type=int, default=64)
PARSER.add('--controller_num_worker_trial', type=int,
        help='trials per worker', default=1)
PARSER.add('--controller_antithetic', type=int, default=1,
        help='set to 0 to disable antithetic sampling')
PARSER.add('--controller_cap_time', type=int, default=0,
        help='set to 0 to disable capping timesteps to 2x of average.')
PARSER.add('--controller_retrain', type=int, default=0,
        help='set to 0 to disable retraining every eval_steps if results suck.\n only works w/ ses, openes, pepg.')
PARSER.add('--controller_seed_start', type=int, default=0, help='initial seed')
PARSER.add('--controller_sigma_init', type=float,
        default=0.1, help='sigma_init')
PARSER.add('--controller_sigma_decay', type=float,
        default=0.999, help='sigma_decay')
PARSER.add('--controller_batch_mode', type=str, default='mean', 
        help='optimize for either min or mean across episodes')
