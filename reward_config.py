from rewards import throughput, similarity
from collections import OrderedDict

reward_config = OrderedDict()

reward_config["throughput"] = {
    "type":"shared",
    "compute_fn": throughput
}

reward_config["similarity"] = {
    "type":"individual",
    "compute_fn": similarity
}
