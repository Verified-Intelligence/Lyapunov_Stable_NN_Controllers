import typing
import unittest

import torch
import torch.nn as nn
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov


def SetupLyapunovNN(
    goal_state: torch.Tensor,
    nominal: typing.Optional[typing.Callable],
    V_psd_form: str,
) -> lyapunov.NeuralNetworkLyapunov:
    lyap_nn = lyapunov.NeuralNetworkLyapunov(
        goal_state,
        hidden_widths=[3, 4],
        x_dim=2,
        R_rows=3,
        absolute_output=False,
        eps=0.1,
        activation=nn.ReLU,
        nominal=nominal,
        V_psd_form=V_psd_form,
    )
    lyap_nn.net[0].weight.data = torch.tensor([[1, 3], [2, -4], [3, 1.0]])
    lyap_nn.net[0].bias.data = torch.tensor([1, 3, 2.0])
    lyap_nn.net[2].weight.data = torch.tensor(
        [[1, 4, 2], [3, -1, 4], [0, 1, 4], [0.5, 1, 3]]
    )
    lyap_nn.net[2].bias.data = torch.tensor([-2, 1, 0, 3.0])
    return lyap_nn


def SetupLyapunovQuadratic(
    goal_state: torch.Tensor,
    R: typing.Optional[torch.Tensor],
) -> lyapunov.NeuralNetworkLyapunov:
    lyap_quad = lyapunov.NeuralNetworkQuadraticLyapunov(
        goal_state,
        x_dim=2,
        R_rows=3,
        eps=0.01,
        R=R,
    )
    return lyap_quad


class TestNeuralNetworkLyapunov(unittest.TestCase):
    def test(self):
        goal_state = torch.tensor([1.0, 2.0])
        for nominal in (
            lambda x: torch.sum((x - goal_state) ** 2, dim=-1, keepdim=True),
            None,
        ):
            for V_psd_form in ("L1", "quadratic"):
                dut = SetupLyapunovNN(goal_state, nominal, V_psd_form)
                self.assertEqual(len(dut.net), 5)
                self.assertEqual(dut.R.shape, (3, 2))
                # Check if load_state_dict works
                dut.load_state_dict(dut.state_dict())

                x = torch.tensor([[2, 3.0], [0, -1], [3.0, 2]])
                V = dut.forward(x)
                self.assertEqual(V.shape, (3, 1))
                for i in range(3):
                    V_nominal = nominal(x[i]) if nominal is not None else 0
                    eps_plus_RtR = dut.eps * torch.eye(2) + dut.R.T @ dut.R
                    if V_psd_form == "L1":
                        non_network_output = torch.norm(
                            eps_plus_RtR @ (x[i] - goal_state), p=1
                        )
                    elif V_psd_form == "quadratic":
                        non_network_output = (x[i] - goal_state) @ (
                            eps_plus_RtR @ (x[i] - goal_state)
                        )
                    Vi = (
                        V_nominal
                        + dut.net(x[i])
                        - dut.net(goal_state)
                        + non_network_output
                    )
                    self.assertAlmostEqual(V[i, 0].item(), Vi.item(), places=5)


class TestNeuralNetworkQuadraticLyapunov(unittest.TestCase):
    def test(self):
        goal_state = torch.tensor([1.0, 2.0])
        for R in (
            torch.rand(3, 2),
            None,
        ):
            dut = SetupLyapunovQuadratic(goal_state, R)

            x = torch.tensor([[2, 3.0], [0, -1], [3.0, 2]])
            V = dut.forward(x)
            self.assertEqual(V.shape, (3, 1))
            for i in range(x.shape[0]):
                Vi = dut.forward(x[i].unsqueeze(0))
                self.assertAlmostEqual(V[i, 0].item(), Vi.item(), places=5)


class TestLyapunovPositivityLoss(unittest.TestCase):
    def test(self):
        goal_state = torch.tensor([1.0, 2.0])
        lyapunov_nn = SetupLyapunovNN(goal_state, nominal=None, V_psd_form="L1")
        N = torch.tensor([[1, 3], [2, -1], [3, 2.0]])
        dut = lyapunov.LyapunovPositivityLoss(lyapunov_nn, Nt=N.T)

        x = torch.tensor([[1, 3], [-0.5, 2], [2.0, 3], [1, 4]])
        loss = dut(x)
        self.assertEqual(loss.shape, (4, 1))
        for i in range(4):
            loss_i = lyapunov_nn(x[i].unsqueeze(0)).squeeze() - torch.norm(
                N @ (x[i] - goal_state), p=1
            )
            self.assertAlmostEqual(loss[i, 0].item(), loss_i.item(), places=4)


class MockDynamics(dynamical_system.DiscreteTimeSystem):
    def __init__(self, goal_state):
        super(MockDynamics, self).__init__(2, 1)
        self.goal_state = goal_state

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        return 1.1 * x + u.repeat((1, 2)) - 0.1 * self.goal_state

    @property
    def x_equilibrium(self) -> torch.Tensor:
        return self.goal_state

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.tensor([0.0])


class TestLyapunovDerivativeSimpleLoss(unittest.TestCase):
    def forward_test(self, goal_state, nominal, V_psd_form):
        lyap_nn = SetupLyapunovNN(goal_state, nominal, V_psd_form)
        mock_dynamics = MockDynamics(goal_state)
        mock_controller = controllers.LinearController(
            torch.tensor([[-1, 3.0]]), mock_dynamics.u_equilibrium
        )
        kappa = 0.01
        dut = lyapunov.LyapunovDerivativeSimpleLoss(
            mock_dynamics, mock_controller, lyap_nn, kappa=kappa
        )
        x = torch.tensor([[1, 3], [1, 2], [-1, 0], [0.5, 0.3], [1.5, 2.1]])

        loss = dut(x)
        self.assertEqual(loss.shape, (x.shape[0], 1))
        for i in range(x.shape[0]):
            u = mock_controller.forward(x[i])
            Vi = lyap_nn.forward(x[i])
            x_next = mock_dynamics(x[i], u)
            loss_expected = (1 - kappa) * Vi - lyap_nn.forward(x_next)
            self.assertAlmostEqual(loss[i].item(), loss_expected.item(), places=3)

    def test(self):
        goal_state = torch.tensor([1.0, 2.0])
        for nominal in (
            lambda x: torch.sum((x - goal_state) ** 2, dim=-1, keepdim=True),
            None,
        ):
            for V_psd_form in ("L1", "quadratic"):
                self.forward_test(goal_state, nominal, V_psd_form)


if __name__ == "__main__":
    unittest.main()
