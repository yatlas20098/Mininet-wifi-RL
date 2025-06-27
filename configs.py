class Mininet_Simulation_Config:
    def __init__(self, sensor_ids, sampling_freq=3, observation_time=10, transmission_size=4*1024, file_lines_per_chunk=5, transmission_frame_duration=1, num_transmission_frames=10000, local_simulation=True, remote_simulation_ip="", remote_simulation_port=""):
        self.sensor_ids = sensor_ids
        self.sampling_freq = sampling_freq
        self.observation_time = observation_time
        self.transmission_size = transmission_size
        self.file_lines_per_chunk = file_lines_per_chunk
        self.transmission_frame_duration = transmission_frame_duration
        self.local_simulation = local_simulation
        self.server_ip = remote_simulation_ip
        self.port = remote_simulation_port
        self.similarity_threshold = 1
        self.num_transmission_frames = num_transmission_frames

class Multi_Agent_PPO_Config:
    def __init__(self, batch_size=64, memory_capacity=256, gamma=0.99, tau=0.01, lr=0.0003, gae_lambda=0.95, policy_clip = 0.1, max_steps=100, num_episodes=10, train_every=2048, n_epochs=10):
        self.batch_size = batch_size 
        self.gamma = gamma 
        self.tau = tau 
        self.lr = lr 
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.train_every = train_every
        self.n_epochs = n_epochs

