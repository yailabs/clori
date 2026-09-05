#!/usr/bin/env python3
"""External HF oracle; never a production dependency. Emit output + conv + SSM state.

YVEX_SELECTIVE_SSD_ORACLE=OUTPUT makes unit.selective_ssd check these 96 values.
This is component evidence, not full-model qualification.
"""
import hashlib
import inspect
import json
import pathlib
import sys

import torch
import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer


def real_component(root, target):
    """Read only the already acquired local snapshot. No provider/network fallback."""
    from safetensors import safe_open
    config_data = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name):
        with safe_open(root / index[name], framework="pt", device="cpu") as source:
            return source.get_tensor(name).float()

    torch.set_num_threads(4)
    config = Mamba2Config(**config_data)
    # Allocate no random projection matrices: the tested boundary begins at projected input.
    with torch.device("meta"):
        mixer = Mamba2Mixer(config, 0)
    mixer.in_proj = torch.nn.Identity()
    mixer.out_proj = torch.nn.Identity()
    for name in ("conv1d.weight", "conv1d.bias", "A_log", "D", "dt_bias", "norm.weight"):
        value = torch.nn.Parameter(tensor("backbone.layers.0.mixer." + name), requires_grad=False)
        parent, _, leaf = name.rpartition(".")
        setattr(mixer.get_submodule(parent) if parent else mixer, leaf, value)
    mixer.eval()
    with torch.no_grad():
        hidden = tensor("backbone.embeddings.weight")[[1, 42]]
        hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + config.layer_norm_epsilon)
        hidden *= tensor("backbone.layers.0.norm.weight")
        projected = torch.nn.functional.linear(hidden, tensor("backbone.layers.0.mixer.in_proj.weight"))
        cache = DynamicCache(config=config)
        output = mixer.torch_forward(projected.unsqueeze(0), cache_params=cache)
        decode_cache = DynamicCache(config=config)
        decoded = torch.cat([mixer.torch_forward(row.reshape(1, 1, -1), cache_params=decode_cache)
                             for row in projected], dim=1)
        torch.testing.assert_close(output, decoded, atol=3e-5, rtol=3e-4)
        conv, state = cache.layers[0].conv_states[0], cache.layers[0].recurrent_states[0]
        torch.testing.assert_close(state, decode_cache.layers[0].recurrent_states[0], atol=3e-5, rtol=3e-4)
        torch.testing.assert_close(conv, decode_cache.layers[0].conv_states[0], atol=0, rtol=0)
        arrays = [projected, mixer.conv1d.weight, mixer.conv1d.bias, mixer.A_log, mixer.D,
                  mixer.dt_bias, mixer.norm.weight, output, conv, state]
        # Plain bounded diagnostic fixture; no checkpoint data is added to Git.
        with target.open("w") as out:
            out.write(f"{config.num_heads} {config.head_dim} {config.state_size} {config.n_groups} "
                      f"{config.conv_kernel} 2 1 0 {config.layer_norm_epsilon:.17g}\n")
            for array in arrays:
                for value in array.flatten().tolist():
                    out.write(f"{value:.9g}\n")
    print(json.dumps({"oracle": "transformers.Mamba2Mixer.torch_forward", "mode": "real-layer-0-component",
                      "repository": "mistralai/Mamba-Codestral-7B-v0.1",
                      "local_source": str(root), "tokens": [1, 42], "dtype": "source-BF16-expanded-F32",
                      "normalization_groups": 1, "norm_before_gate": False,
                      "source_config_norm_before_gate": config_data.get("norm_before_gate"),
                      "normalization_authority_resolved": False,
                      "transformers": transformers.__version__, "torch": torch.__version__,
                      "implementation_sha256": hashlib.sha256(pathlib.Path(inspect.getfile(Mamba2Mixer)).read_bytes()).hexdigest(),
                      "fixture_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "whole_model": False}, sort_keys=True))


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--acquired-source":
        real_component(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
        return
    torch.set_num_threads(1)
    torch.manual_seed(0)
    config = Mamba2Config(hidden_size=4, expand=2, num_heads=4, head_dim=2,
                          state_size=2, n_groups=2, num_hidden_layers=1,
                          conv_kernel=3, chunk_size=2, layer_norm_epsilon=1e-5)
    model = Mamba2Mixer(config, 0).float().eval()
    model.in_proj = torch.nn.Identity()
    model.out_proj = torch.nn.Identity()
    projection = ((torch.arange(112).float() % 17) - 8) * 0.125
    with torch.no_grad():
        model.conv1d.weight.copy_((((torch.arange(48).float() % 7) - 3) * 0.125).reshape(16, 1, 3))
        model.conv1d.bias.copy_((torch.arange(16).float() - 7) * 0.03125)
        model.A_log.copy_(torch.arange(4).float() * 0.125)
        model.D.copy_(0.5 + torch.arange(4).float() * 0.25)
        model.dt_bias.copy_(-0.5 + torch.arange(4).float() * 0.125)
        model.norm.weight.copy_(0.75 + torch.arange(8).float() * 0.0625)
        cache = DynamicCache(config=config)
        output = model.torch_forward(projection.reshape(1, 4, 28), cache_params=cache)
        conv = cache.layers[0].conv_states[0]
        state = cache.layers[0].recurrent_states[0]
        decode_cache = DynamicCache(config=config)
        decoded = torch.cat([model.torch_forward(row.reshape(1, 1, 28), cache_params=decode_cache)
                             for row in projection.reshape(4, 28)], dim=1)
        torch.testing.assert_close(output, decoded, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(state, decode_cache.layers[0].recurrent_states[0], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(conv, decode_cache.layers[0].conv_states[0], atol=0, rtol=0)
        values = torch.cat([output.flatten(), conv.flatten(), state.flatten()]).tolist()
    target = pathlib.Path(sys.argv[1])
    target.write_text("\n".join(format(value, ".9g") for value in values) + "\n")
    source = pathlib.Path(inspect.getfile(Mamba2Mixer))
    print(json.dumps({"oracle": "transformers.Mamba2Mixer.torch_forward",
                      "transformers": transformers.__version__, "torch": torch.__version__,
                      "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "fixture_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "dtype": "float32", "tokens": 4, "values": len(values),
                      "atol": 3e-6, "rtol": 3e-5, "whole_model": False}, sort_keys=True))


if __name__ == "__main__":
    main()
