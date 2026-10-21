import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from torch_policy import load_policy_from_files


class ApolloPolicyGRU(nn.Module):
    def __init__(self, policy, obs_dim):
        super().__init__()
        self.policy = policy
        self.obs_dim = obs_dim


    def forward(self, input):
        mean, _, next_carry = self.policy.forward_step(input[:, :self.obs_dim], input[:, self.obs_dim:])
        return torch.cat([mean, next_carry], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--output", type=str, default="policy.onnx")
    parser.add_argument("--obs-dim", type=int, default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--nr-check-inputs", type=int, default=5)
    args = parser.parse_args()

    policy, meta = load_policy_from_files(args.weights, args.meta, torch.device("cpu"))

    obs_dim = args.obs_dim if args.obs_dim is not None else meta["expected_policy_obs_dim"]
    indices = meta["policy_observation_indices"]
    if indices != list(range(obs_dim)):
        raise ValueError(
            f"--obs-dim {obs_dim} does not match the trained policy_observation_indices {indices}. "
            "The exported graph indexes its observation with those indices, so it expects the "
            "caller to feed the full environment observation rather than the pre-selected policy one."
        )

    carry_dim = policy.gru_hidden_dim
    action_dim = policy.action_dim
    input_dim = obs_dim + carry_dim
    output_dim = action_dim + carry_dim

    # Apollo's OnnxSession binds exactly one input and one output and never inspects
    # their structure, so the recurrent carry is packed next to the observation and
    # next to the action mean. The caller feeds the carry back in on every step.
    model = ApolloPolicyGRU(policy, obs_dim)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.zeros(1, input_dim, dtype=torch.float32),
            args.output,
            input_names=["obs"],
            output_names=["actions"],
            opset_version=args.opset,
            dynamic_axes=None,
            do_constant_folding=True,
        )

    # torch.onnx.export spills the weights into a .data sidecar once the graph is
    # large enough, but Apollo is given a single .onnx path, so inline them again.
    exported = onnx.load(args.output)

    # It also stamps every node with debug metadata carrying the wrapper class name
    # and the absolute path of this script, which would otherwise end up inside the
    # deployed artifact. Removing it makes the export deterministic on the weights.
    for node in exported.graph.node:
        del node.metadata_props[:]

    onnx.save_model(exported, args.output, save_as_external_data=False)
    sidecar = Path(f"{args.output}.data")
    if sidecar.exists():
        sidecar.unlink()

    graph_shapes = {}
    for value in list(exported.graph.input) + list(exported.graph.output):
        graph_shapes[value.name] = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
    if graph_shapes["obs"] != [1, input_dim]:
        raise ValueError(f"Exported input shape {graph_shapes['obs']} != [1, {input_dim}]")
    if graph_shapes["actions"] != [1, output_dim]:
        raise ValueError(f"Exported output shape {graph_shapes['actions']} != [1, {output_dim}]")

    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst_abs_diff = 0.0
    output_scale = 0.0
    for trial in range(args.nr_check_inputs):
        sample = rng.standard_normal((1, input_dim)).astype(np.float32)
        if trial == 0:
            sample[:, obs_dim:] = 0.0
        with torch.no_grad():
            reference = model(torch.from_numpy(sample)).numpy()
        onnx_output = session.run(None, {"obs": sample})[0]
        worst_abs_diff = max(worst_abs_diff, float(np.abs(reference - onnx_output).max()))
        output_scale = max(output_scale, float(np.abs(reference).max()))

    relative_error = worst_abs_diff / max(output_scale, 1e-9)
    if relative_error > 1e-5:
        raise ValueError(
            f"onnxruntime disagrees with the PyTorch reference by {relative_error:.3e} "
            f"relative to an output scale of {output_scale:.3f}"
        )

    print(f"obs_dim {obs_dim} + carry_dim {carry_dim} -> action_dim {action_dim} + carry_dim {carry_dim}")
    print(f"input  obs     {graph_shapes['obs']}")
    print(f"output actions {graph_shapes['actions']}")
    print(f"torch vs onnxruntime: max abs diff {worst_abs_diff:.3e} over {args.nr_check_inputs} inputs, "
          f"{relative_error:.3e} relative to output scale {output_scale:.3f}")
    print(f"policy std_dev {meta['std_dev']:.6f}")
    print(f"Saved {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
