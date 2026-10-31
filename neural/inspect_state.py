import json
from neural.client import FlyQNeuralClient


client = FlyQNeuralClient()
state = client.state()

print("Top-level /state fields:")
print("========================")

for key, value in state.items():
    if key == "frame":
        print(f"{key}: <image omitted>")
    elif isinstance(value, dict):
        print(f"{key}: dict -> {list(value.keys())}")
    elif isinstance(value, list):
        print(f"{key}: list[{len(value)}]")
    else:
        print(f"{key}: {value!r}")
