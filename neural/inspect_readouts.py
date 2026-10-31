from pprint import pprint
from neural.client import FlyQNeuralClient


client = FlyQNeuralClient()
state = client.state()

print("\nACTION")
print("======")
pprint(state.get("action"))

print("\nREADOUTS")
print("========")
for i, readout in enumerate(state.get("readouts", [])):
    print(f"\n[{i}]")
    pprint(readout)

print("\nNEURON VOLTAGES")
print("===============")
pprint(state.get("neuron_voltage_mv"))
