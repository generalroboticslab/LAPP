import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run='-1', checkpoint='-1'):
    # PJ: I modify the int -1 to the string '-1' for comparison
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run=='-1':
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint=='-1':
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint
        if args.silence:
            # PJ: This args is to stop printing training logs to the terminal. Used for Hydra multi-process training
            cfg_train.runner.silence = args.silence

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "default": 101, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},

        # PJ: the following args are added by PJ
        {"name": "--log_root", "type": str, "default": None, "help": "The path to save the trained model."},
        {"name": "--record", "action": "store_true", "default": False, "help": "Record the video"},
        {"name": "--reward_module_name", "type": str, "default": None, "help": "The dot name to load the reward module."},
        {"name": "--silence", "action": "store_true", "default": False, "help": "Stop printing training logs to the terminal. Used for Hydra multi-process training"},
        {"name": "--test_direct", "type": str, "default": "forward", "help": "forward, backward, left, left_turn... locomotion directions while testing"},
        {"name": "--terrain", "type": str, "default": "plane", "help": "plain, pyramid_stairs, pyramid_sloped, discrete_obstacles, wave, stepping_stones ... different terrain while testing"},
        {"name": "--init_angle_low", "type": float, "help": "The lower bound of the initial angle range, which will be times with pi in the code. E.g., 0.25 will become 0.25*pi"},
        {"name": "--init_angle_high", "type": float, "help": "The higher bound of the initial angle range, which will be times with pi in the code. E.g., 1.50 will become 1.50*pi"},
        {"name": "--init_height_low", "type": float, "help": "The lower bound of the initial height range. E.g., 1.50 m"},
        {"name": "--init_height_high", "type": float, "help": "The higher bound of the initial height range. E.g., 3.50 m"},
        {"name": "--random_in_air", "type": int, "default": 0, "help": "Mode for random in air, if 1 is back flip."},
        {"name": "--num_steps_per_env", "type": int, "default": 24, "help": "Number of step per environment in each episode"},
        {"name": "--save_pairs", "action": "store_true", "default": False, "help": "Save the collected trajectory pairs"},
        {"name": "--pref_scale", "type": float, "default": 1.0, "help": "the scale of the preference rewards"},
        {"name": "--dense_reward_scale", "type": float, "default": 1.0, "help": "the scale of the dense rewards explicitly written out in a reward function"},
        {"name": "--state_feature_dim", "type": int, "default": 15, "help": "the feature dimension of the state that the preference predictors will take in"},
        {"name": "--pref_pred_start_after_eps", "type": int, "default": 499, "help": "start use the preference predictor to generate a preference reward after this episode, which means start at its next episode"},
        {"name": "--pref_pred_update_period_eps", "type": int, "default": 100, "help": "update the preference predictor every after this period of episodes number"},
        {"name": "--pref_pred_train_data_period_eps", "type": int, "default": 500, "help": "the data to train the preference predictor is sampled from this period of episodes number"},
        {"name": "--num_pref_pairs_total_train", "type": int, "default": 500, "help": "the number of trajectory pairs in a training set to train a reward predictor"},
        {"name": "--pref_pred_pool_models_num", "type": int, "default": 9, "help": "the number of preference models trained and stored in a pool"},
        {"name": "--pref_pred_select_models_num", "type": int, "default": 3, "help": "the number of preference models selected from the pool of trained models"},
        {"name": "--pref_pred_input_mode", "type": int, "default": 0, "help": "mode 0: state(15), 1: obs(48), 2: state(15)+action(12), 3: obs(48)+action(12), 4: state(15)+obs(48)+action(12)"},
        {"name": "--pref_pred_transformer_embed_dim", "type": int, "default": 64, "help": "the embedding dimension of the preference prediction transformer network"},
        {"name": "--pref_pred_seq_length", "type": int, "default": 1, "help": "the sequence length of the preference prediction transformer network"},
        {"name": "--pref_pred_batch_size", "type": int, "default": 256, "help": "the batch size to train a preference predictor"},
        {"name": "--pref_pred_epoch", "type": int, "default": 90, "help": "the number of epochs to train a preference predictor"},
        {"name": "--init_rollout_action_noise", "action": "store_true", "default": False, "help": "Add random noise to the action when doing the initial rollout to collect preference predictors training data"},
        {"name": "--prompt_init_task", "type": str, "default": "org_forward", "help": "Choose the initialization prompt for the GPT. org_forward stands for walking forward in speed range [0.0, 2.2]. bounding_forward stands for bounding (gate) forward in speed range [0.0, 2.2]."},
        {"name": "--static_pref_predictor_path", "type": str, "default": "pref_pred_models/Dec20_12-51-05_gpt-4o_4000data_pool_500traj_pairs_9pool_3models.pt", "help": "the path to load the one preference predictor in the entire training process"},
        {"name": "--save_right_camera_figures", "action": "store_true", "default": False, "help": "Save the figures from the camera on the right side following the robot"},
        {"name": "--video_figures_path", "type": str, "default": "normal_cadence", "help": "The path to save the figures from the camera on the right side following the robot"},
        {"name": "--init_difficulty_level", "type": int, "default": 0, "help": "initial difficulty level of the terrain curriculum"},
        # XW: add save_interval arg
        {"name": "--save_interval", "type": int, "default": 100, "help": "The interval of saving checkpoints"},
        {"name": "--num_pref_pairs_per_episode", "type": int, "default": 1, "help": "num_pref_pairs_per_episode"},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
