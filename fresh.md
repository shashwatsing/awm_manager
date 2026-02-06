# Clone using github 

# Source awm
python -m pip install -e source/awm

# To start training
HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py --task Template-Awm_Manager-v0 --num_envs 1024 --max_iterations 1000

# To play the trained policy 
HYDRA_FULL_ERROR=1 python scripts/rsl_rl/play.py --task Template-Awm_Manager-v0 --num_envs 10 --checkpoint /home/shashwat/awm_manager/logs/rsl_rl/awm_manager/2026-02-05_12-55-55/model_250.pt