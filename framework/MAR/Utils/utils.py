import random
from collections import Counter


def find_mode(values):
    mode, _ = Counter(values).most_common(1)[0]
    return mode


def get_kwargs(mode, agent_count):
    spatial = None
    temporal = None
    rounds = 1

    if mode in {"DirectAnswer", "CoT", "IO", "Reflection"}:
        spatial = [[0 for _ in range(agent_count)] for _ in range(agent_count)]
        temporal = [[0 for _ in range(agent_count)] for _ in range(agent_count)]
    elif mode == "FullConnected":
        spatial = [[1 if i != j else 0 for i in range(agent_count)] for j in range(agent_count)]
        temporal = [[1 for _ in range(agent_count)] for _ in range(agent_count)]
    elif mode == "Random":
        spatial = [[random.randint(0, 1) if i != j else 0 for i in range(agent_count)] for j in range(agent_count)]
        temporal = [[random.randint(0, 1) for _ in range(agent_count)] for _ in range(agent_count)]
    elif mode == "Chain":
        spatial = [[1 if i == j + 1 else 0 for i in range(agent_count)] for j in range(agent_count)]
        temporal = [[1 if i == 0 and j == agent_count - 1 else 0 for i in range(agent_count)] for j in range(agent_count)]
    elif mode == "Debate":
        spatial = [[0 for _ in range(agent_count)] for _ in range(agent_count)]
        temporal = [[1 for _ in range(agent_count)] for _ in range(agent_count)]
        rounds = 2

    return {
        "initial_spatial_probability": 0.5,
        "fixed_spatial_masks": spatial,
        "initial_temporal_probability": 0.5,
        "fixed_temporal_masks": temporal,
        "node_kwargs": None,
        "num_rounds": rounds,
    }
