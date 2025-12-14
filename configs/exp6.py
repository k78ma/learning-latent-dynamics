class E2C_Config:
    # env
    mjcf_file: str = "sim/arm.mjcf.xml"
    fps: int = 30
    traj_seconds: int = 5
    camera_name: str = "camera_1"
    num_trajectories: int = 100
    randomize_init: bool = True
    ctrl_max: float = 1.0

    data_dir: str = "data/raw"
    frames_filename: str = "frames.npy"
    actions_filename: str = "actions.npy"

    # VAE
    img_height: int = 32
    img_width: int = 32
    channels_per_img: int = 1 
    latent_dim: int = 16
    state_seq_len: int = 1 
    vae_type: str = "conv" 

    conv_channels: list[int] = [32, 64, 128]
    conv_kernel_size: int | list[int] = 4
    conv_stride: int | list[int] = 2
    conv_padding: int | list[int] = 1

    ## DYNAMICS
    latent_dynamics_model_type: str = "rnn"  
    mlp_hidden_dims: list[int] = [100, 100]
    # RNN
    rnn_hidden_dim: int = 128
    rnn_seq_len: int = 10 
    
    
    # Tracking
    experiment_name: str = None 
    experiment_base_dir: str = "experiments"

    # Training
    epochs: int = 40
    epochs_per_checkpoint: int = 10
    beta_kl: float = 1.0 
    lr: float = 3e-4
    batch_size: int = 128

    eps = 1e-9  

    random_seed:int = 0

    iterations_per_log: int = 1000


e2c_config = E2C_Config()
