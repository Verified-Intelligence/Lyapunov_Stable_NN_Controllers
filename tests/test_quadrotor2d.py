import neural_lyapunov_training.quadrotor2d as quadrotor2d
import neural_lyapunov_training.dynamical_system as dynamical_system
import torch
import unittest
import numpy as np


class TestPendulumDiscreteTimeSystem(unittest.TestCase):
    def test_forward(self):
        quadrotor_continuous = quadrotor2d.Quadrotor2DDynamics()
        dt = 0.01
        dut = dynamical_system.SecondOrderDiscreteTimeSystem(quadrotor_continuous, dt)
        self.assertEqual(dut.nx, 6)
        self.assertEqual(dut.nu, 2)
        self.assertTrue(dut.x_equilibrium.equal(torch.zeros(6)))
        nq = dut.nq

        x = torch.tensor(
            [
                [0.5, 0.2, 1, 0.5, 0.3, 2],
                [0.1, 0.4, 0.4, -0.5, 0.2, 1.1],
                [0.3, -0.5, 0.3, 0.9, -1.5, 2],
            ]
        )
        u = torch.tensor([[1, 2], [3, 0.4], [0.5, 5]])
        x_next = dut.forward(x, u)
        self.assertEqual(x_next.shape, x.shape)
        for i in range(x.shape[0]):
            qddot = quadrotor_continuous.forward(
                x[i, :].unsqueeze(0), u[i, :].unsqueeze(0)
            ).squeeze()
            # Explicit Euler integration for velocity
            qdot_next = x[i, nq:] + qddot * dt
            np.testing.assert_allclose(
                x_next[i, nq:].detach().numpy(), qdot_next.detach().numpy()
            )
            # Midpoint integration for position
            np.testing.assert_allclose(
                x_next[i, :nq].detach().numpy(),
                x[i, :nq] + (x[i, nq:] + qdot_next).detach().numpy() / 2 * dt,
            )


if __name__ == "__main__":
    unittest.main()
