# LAPP: Large Language Model Feedback for Preference-Driven Reinforcement Learning

[Pingcheng Jian](https://pingcheng-jian.github.io/),
[Xiao Wei](https://www.linkedin.com/in/xiao-wei-36a10b325/),
[Yanbaihui Liu](https://www.linkedin.com/in/yanbaihui-liu-19077216b/),
[Samuel A. Moore](https://samavmoore.github.io/),
[Michael M. Zavlanos](https://mems.duke.edu/faculty/michael-zavlanos),
[Boyuan Chen](http://boyuanchen.com/)
<br>
Duke University
<br>

## Content

- [Installation](#installation)
- [Training](#training)
- [Testing](#testing)

## installation
- Install the packages below.
```
conda create -n lapp python=3.8

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

cd repo/isaacgym/python
python -m pip install -e .

cd repo/rsl_rl
python -m pip install -e .

cd repo/unitree_rl_gym
python -m pip install -e .

python -m pip install hydra-core

python -m pip install openai

python -m pip install opencv-python

python -m pip install numpy==1.21.0
```

- Add you OpenAI API key to the api_key/openai_api_key.txt file

## training
1. Example of training the go2 robot for flat ground locomotion
```
python train_flat_locomotion_with_preference.py --task gpt_go2 --log_root logs/go2_flat --rl_device cuda:0 \ 
--sim_device cuda:0 --max_iterations 3000 --reward_module_name flat_pref_prompt.reward.go2_forward_reward --headless --save_pairs --pref_scale 1.0
```

2. Example of training the go2 robot for jumping high
```
python train_backflip_light.py --task go2_backflip --log_root logs/go2_backflip_light --rl_device cuda:0 \ 
--sim_device cuda:0 --max_iterations 5000 --num_envs 4096 --reward_module_name go2backflip_prompt.reward.go2_jump_reward --headless --random_in_air 0
```

3. Example of training the go2 robot for backflip from the pre-trained jumping high behavior
```
python train_backflip_light_with_preference.py --task go2_backflip --log_root logs/go2_backflip_light \ 
--rl_device cuda:0 --sim_device cuda:0 --max_iterations 5000 --reward_module_name go2backflip_prompt.reward.go2_backflip_reward \ 
--headless --save_pairs --prompt_init_task backflip --pref_pred_pool_models_num 6 --pref_pred_select_models_num 3 \ 
--pref_pred_input_mode 0 --pref_pred_seq_length 8 --pref_pred_epoch 90 --load_run <jump_model_path> --checkpoint 5000 \ 
--resume --headless --init_angle_low 0.50 --init_angle_high 1.75 --init_height_low 1.50 --init_height_high 3.00 --random_in_air 1 --num_steps_per_env 24 --pref_scale 50.0
```

4. Example of training the go2 robot for bounding gait
```
python train_bounding_with_preference.py --task gpt_go2 --log_root logs/go2_bounding --rl_device cuda:0 --sim_device cuda:0 \
--max_iterations 5600 --reward_module_name go2bounding_prompt.reward.go2_forward_reward --headless --save_pairs --pref_scale 1.0 --prompt_init_task bounding_forward
```

5. Example of training the go2 robot for fast cadence
```
python train_cadence_with_preference.py --task gpt_go2 --log_root logs/go2_cadence --rl_device cuda:0 --sim_device cuda:0 --max_iterations 5000 \
--reward_module_name go2fast_prompt.reward.go2_forward_reward --headless --save_pairs --pref_scale 1.0 --prompt_init_task fast_cadence_forward \
--pref_pred_pool_models_num 6 --pref_pred_select_models_num 3 --pref_pred_input_mode 0 --pref_pred_seq_length 8 --pref_pred_epoch 90
```

6. Example of training the go2 robot for slow cadence
```
python train_cadence_with_preference.py --task gpt_go2 --log_root logs/go2_cadence --rl_device cuda:1 --sim_device cuda:1 --max_iterations 5000 \
--reward_module_name go2slow_prompt.reward.go2_forward_reward --headless --save_pairs --pref_scale 1.0 --prompt_init_task slow_cadence_forward \
--pref_pred_pool_models_num 6 --pref_pred_select_models_num 3 --pref_pred_input_mode 0 --pref_pred_seq_length 8 --pref_pred_epoch 90
```

7. Example of training the go2 robot for stairs locomotion
```
python train_stairs_locomotion_with_preference.py --task=go2_stairs --log_root logs/go2_stairs --rl_device cuda:0 --sim_device cuda:0 \
--max_iterations 5000 --reward_module_name go2stairs_prompt.reward.go2_forward_reward --terrain pyramid_stairs --headless --num_envs 6144 --save_pairs --pref_scale 1.0
```

8. Example of training the go2 robot for obstacles locomotion
```
python train_obstacles_with_preference.py --task=go2_obs_curr --log_root logs/go2_obstacles --rl_device cuda:0 --sim_device cuda:0 --max_iterations 2000 \
--reward_module_name go2obstacles_prompt.reward.go2_forward_reward --terrain curriculum_obs --headless --num_envs 4096 --save_interval 100 --seed 1 --save_pairs --pref_scale 1.0
```

9. Example of training the go2 robot for slope locomotion
```
python train_slope_with_preference.py --task=go2_slope_pref --log_root logs/go2_slope --rl_device cuda:0 --sim_device cuda:0 --max_iterations 2000 \
--reward_module_name go2slope_prompt.reward.go2_forward_reward --terrain curriculum_slope --headless --num_envs 5000 --save_interval 100 --save_pairs --pref_scale 2.0 --seed 101
```

## testing
1. Example of testing the go2 robot for flat ground locomotion
```
python test_flat_locomotion_with_preference.py --task=gpt_go2 --num_envs 2 --rl_device cuda:0 \
--sim_device cuda:0 --load_run ckpt --checkpoint=3999 --log_root logs/go2_flat --headless --record --test_direct forward
```
![Flat](https://github.com/generalroboticslab/LAPP/blob/main/test_videos/flat_forward_right3999.mp4)

2. Example of testing the go2 robot for backflip
```
python test_backflip_normal.py --task=go2_backflip --num_envs 2 --rl_device cuda:7 --sim_device cuda:7 \ 
--load_run <backflip_model_path> --checkpoint=5000 --log_root logs/go2_backflip --headless --record --random_in_air 0
```

3. Example of testing the go2 robot for bounding
```
python test_bounding_with_preference.py --task=gpt_go2 --num_envs 2 --rl_device cuda:0 --sim_device cuda:0 \
--load_run ckpt --checkpoint=5599 --log_root logs/go2_bounding --headless --record --test_direct forward
```

4. Example of testing the go2 robot for fast cadence
```
python test_cadence_with_preference.py --task=gpt_go2 --num_envs 2 --rl_device cuda:0 --sim_device cuda:0 \
--load_run fast_ckpt --checkpoint=4999 --log_root logs/go2_cadence --headless --record --test_direct forward
```

5. Example of testing the go2 robot for slow cadence
```
python test_cadence_with_preference.py --task=gpt_go2 --num_envs 2 --rl_device cuda:0 --sim_device cuda:0 \
--load_run slow_ckpt --checkpoint=4999 --log_root logs/go2_cadence --headless --record --test_direct forward
```

6. Example of testing the go2 robot for stairs
```
python test_stairs_with_preference.py --task=go2_stairs --num_envs 2 --rl_device cuda:0 --sim_device cuda:0 \
--load_run ckpt --checkpoint=4999 --log_root logs/go2_stairs --test_direct forward --terrain pyramid_stairs --headless --record
```

7. Example of testing the go2 robot for obstacles
```
python test_obstacles_with_preference.py --task=go2_terrain --log_root logs/go2_obstacles --rl_device cuda:0 --sim_device cuda:0 \
--reward_module_name go2obstacles_prompt.reward.go2_forward_reward --terrain discrete_obstacles --headless --checkpoint 1999 --load_run ckpt --silence --record --test_direct forward
```

8. Example of testing the go2 robot for slope
```
python test_slope_with_preference.py --task=go2_terrain --log_root logs/go2_slope --rl_device cuda:0 --sim_device cuda:0 \
--reward_module_name go2slope_prompt.reward.go2_forward_reward --terrain pyramid_sloped --headless --checkpoint 1999 --load_run ckpt --silence --record --test_direct forward
```

9. Example of testing the go2 robot for wave
```
python test_wave_with_preference.py --task=go2_terrain --log_root logs/go2_wave --rl_device cuda:0 --sim_device cuda:0 \
--reward_module_name go2wave_prompt.reward.go2_forward_reward --terrain wave --headless --checkpoint 1499 --load_run ckpt --silence --record --test_direct forward
```

## Manipulation

For the code of the manipulation tasks, check out: https://github.com/generalroboticslab/LAPP_manipulation

## License

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.

## Acknowledgement

This project refers to the github repositories [Unitree RL GYM](https://github.com/unitreerobotics/unitree_rl_gym), 
[RSL RL](https://github.com/leggedrobotics/rsl_rl), and 
[Isaac Gym](https://github.com/isaac-sim/IsaacGymEnvs).

