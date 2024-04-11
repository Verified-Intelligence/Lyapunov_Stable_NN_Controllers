import torch
import os
import sys

if len(sys.argv) <= 1:
    print(f"Utility to split Lyapunov function from a full model.")
    print(f"Usage: {sys.argv[0]} input_pth_file")
    exit()

output_dict = {}
filename = sys.argv[1]
output_filename = os.path.splitext(filename)[0] + "_lyapunov.pth"

model = torch.load(sys.argv[1], map_location="cpu")
d = model["state_dict"]
for key in d:
    if key.startswith("lyapunov"):
        newkey = key.replace("lyapunov.", "")
        output_dict[newkey] = d[key]
        print(f"Saving {newkey} with size {output_dict[newkey].size()}")

print(f"saving to {output_filename}")
torch.save(output_dict, output_filename)
