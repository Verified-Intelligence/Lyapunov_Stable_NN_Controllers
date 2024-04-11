import unittest

import torch
import numpy as np
import neural_lyapunov_training.controllers as controllers


class TestNeuralNetworkController(unittest.TestCase):
    def forward_tester(self, dut):
        x_samples = torch.rand((10, 3))
        u_samples = dut.forward(x_samples)

        if dut.clip_output is None:
            f = lambda x: dut.net(x)
        elif dut.clip_output == "tanh":
            f = (
                lambda x: torch.tanh(dut.net(x)) * (dut.u_up - dut.u_lo) / 2
                + (dut.u_up + dut.u_lo) / 2
            )
        elif dut.clip_output == "clamp":
            f = lambda x: torch.clamp(dut.net(x), min=dut.u_lo, max=dut.u_up)

        if dut.u_equilibrium is None or dut.x_equilibrium is None:
            u_samples_expected = f(x_samples)
        else:
            u_samples_expected = f(x_samples) - f(dut.x_equilibrium) + dut.u_equilibrium
        np.testing.assert_allclose(
            u_samples.detach().numpy(), u_samples_expected.detach().numpy(), atol=1e-6
        )

    def test_forward(self):
        # no clip, no x_equilibrium/u_equilibrium
        dut1 = controllers.NeuralNetworkController(
            nlayer=3,
            in_dim=3,
            out_dim=2,
            hidden_dim=16,
            clip_output=None,
            u_lo=None,
            u_up=None,
            x_equilibrium=None,
            u_equilibrium=None,
        )
        self.forward_tester(dut1)

        # no clip, with x_equilibriu/u_equilibrium
        x_equilibrium = torch.tensor([0.5, 0.2, -0.4])
        u_equilibrium = torch.tensor([0.5, 1.4])
        dut2 = controllers.NeuralNetworkController(
            nlayer=3,
            in_dim=3,
            out_dim=2,
            hidden_dim=16,
            clip_output=None,
            u_lo=None,
            u_up=None,
            x_equilibrium=x_equilibrium,
            u_equilibrium=u_equilibrium,
        )
        self.forward_tester(dut2)

        # tanh clip, no x_equilibrium/u_equilibrium
        u_lo = torch.tensor([-2, -1.0])
        u_up = torch.tensor([1.0, 4.0])
        dut3 = controllers.NeuralNetworkController(
            nlayer=3,
            in_dim=3,
            out_dim=2,
            hidden_dim=16,
            clip_output="tanh",
            u_lo=u_lo,
            u_up=u_up,
            x_equilibrium=None,
            u_equilibrium=None,
        )
        self.forward_tester(dut3)

        # tanh clip, with x_equilibrium/u_equilibrium
        dut4 = controllers.NeuralNetworkController(
            nlayer=3,
            in_dim=3,
            out_dim=2,
            hidden_dim=16,
            clip_output="tanh",
            u_lo=u_lo,
            u_up=u_up,
            x_equilibrium=x_equilibrium,
            u_equilibrium=u_equilibrium,
        )
        self.forward_tester(dut4)

        # clamp clip, no x_equilibrium/u_equilibrium
        u_lo = torch.tensor([-2, -1.0])
        u_up = torch.tensor([1.0, 4.0])
        dut5 = controllers.NeuralNetworkController(
            nlayer=3,
            in_dim=3,
            out_dim=2,
            hidden_dim=16,
            clip_output="clamp",
            u_lo=u_lo,
            u_up=u_up,
            x_equilibrium=None,
            u_equilibrium=None,
        )
        self.forward_tester(dut5)

        # tanh clip, with x_equilibrium/u_equilibrium
        dut6 = controllers.NeuralNetworkController(
            nlayer=3,
            in_dim=3,
            out_dim=2,
            hidden_dim=16,
            clip_output="clamp",
            u_lo=u_lo,
            u_up=u_up,
            x_equilibrium=x_equilibrium,
            u_equilibrium=u_equilibrium,
        )
        self.forward_tester(dut6)


if __name__ == "__main__":
    unittest.main()
