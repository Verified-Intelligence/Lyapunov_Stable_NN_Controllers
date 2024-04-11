import neural_lyapunov_training.pendulum as pendulum
import neural_lyapunov_training.dynamical_system as dynamical_system
import torch
import unittest


class TestPendulumDiscreteTimeSystem(unittest.TestCase):
    def test_forward(self):
        pendulum_continuous = pendulum.PendulumDynamics()
        dt = 0.01
        dut = dynamical_system.SecondOrderDiscreteTimeSystem(pendulum_continuous, dt)
        self.assertEqual(dut.nx, 2)
        self.assertEqual(dut.nu, 1)
        self.assertEqual(dut.nq, 1)
        self.assertTrue(dut.x_equilibrium.equal(torch.tensor([0.0, 0.0])))
        self.assertTrue(dut.u_equilibrium.equal(torch.tensor([0.0])))

        x = torch.tensor([[0.5, 0.2], [0.1, 0.4], [0.3, -0.5]])
        u = torch.tensor([[1], [-2], [0.5]])
        x_next = dut.forward(x, u)
        self.assertEqual(x_next.shape, x.shape)
        for i in range(x.shape[0]):
            qdot_next = pendulum_continuous.forward(
                x[i, :].unsqueeze(0), u[i, :].unsqueeze(0)
            )
            self.assertAlmostEqual(
                x_next[i, 1].item(), (x[i, 1] + qdot_next[0] * dt).item(), places=7
            )
            self.assertAlmostEqual(
                (x_next[i, 0] - x[i, 0]).item(),
                (x_next[i, 1] + x[i, 1]).item() / 2 * dt,
                places=7,
            )


if __name__ == "__main__":
    unittest.main()
