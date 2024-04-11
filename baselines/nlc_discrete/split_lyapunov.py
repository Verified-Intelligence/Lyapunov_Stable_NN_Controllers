import torch
import os
import sys

if len(sys.argv) <= 1:
    print(f"Utility to split Lyapunov function from a full model.")
    print(f"Usage: {sys.argv[0]} input_pth_file")
    exit()

output_dict = {}
filename = sys.argv[1]
output_filename = os.path.splitext(filename)[0] + '_lyapunov.pth'

model = torch.load(sys.argv[1], map_location='cpu')

for key in model:
    if key.startswith('lyap_model.'):
        newkey = key.replace('lyap_model.', '')
        output_dict[newkey] = model[key]
        print(f'Saving {newkey} with size {output_dict[newkey].size()}')

print(f'saving to {output_filename}')
torch.save(output_dict, output_filename)
