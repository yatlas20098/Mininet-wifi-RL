class Mininet_Simulation_Config:
    def __init__(self, sensor_ids, num_clusters=1, sampling_freq=4, observation_time=10, transmission_size=1*1024, file_lines_per_chunk=5, transmission_frame_duration=1, num_transmission_frames=10000, local_simulation=True, remote_simulation_ip="", remote_simulation_port=""):
        self.sensor_ids = sensor_ids
        self.sampling_freq = sampling_freq
        self.observation_time = observation_time
        self.transmission_size = transmission_size
        self.file_lines_per_chunk = file_lines_per_chunk # Remove
        self.transmission_frame_duration = transmission_frame_duration # Remove
        self.local_simulation = local_simulation
        self.server_ip = remote_simulation_ip
        self.port = remote_simulation_port
        self.similarity_threshold = 4 # Should be arg
        self.num_transmission_frames = num_transmission_frames # Remove
        self.num_clusters = num_clusters
        self.transmission_frequencies = [50, 100, 200, 400]


class Multi_Agent_PPO_Config:
    def __init__(self, batch_size=64, memory_capacity=256, gamma=0.99, tau=0.005, actor_lr=3e-4, critic_lr=1e-3, gae_lambda=0.99, policy_clip = 0.2, max_steps=100, num_episodes=99999999, train_every=2048, n_epochs=10):
        self.batch_size = batch_size 
        self.gamma = gamma 
        self.tau = tau 
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.train_every = train_every
        self.n_epochs = n_epochs
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.00

