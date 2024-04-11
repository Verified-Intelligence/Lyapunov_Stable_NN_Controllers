import torch

from auto_LiRPA.bound_ops import (
    BoundActivation,
    BoundMatMul,
    BoundReduceSum,
    BoundMul,
    BoundSqr,
    BoundParams,
)


def optimize_graph(model, version="v1"):
    if version == "v1":
        merge_quadratic_term(model)
    elif version == "v2":
        merge_quadratic_term_v2(model)
    else:
        raise ValueError(version)


def merge_quadratic_term(model):
    nodes = list(model.nodes())
    for node in nodes:
        if (
            isinstance(node, BoundReduceSum)
            and isinstance(node.inputs[0], BoundMul)
            and isinstance(node.inputs[0].inputs[1], BoundMatMul)
            and node.inputs[0].inputs[1].inputs[0] == node.inputs[0].inputs[0]
            and not node.inputs[0].inputs[1].inputs[1].perturbed
        ):
            q = node.inputs[0].inputs[1].inputs[1].forward_value
            q_cholesky = torch.nn.Parameter(torch.linalg.cholesky(q))
            node_q_cholesky = BoundParams(f"{node.name}/q_cholesky", q_cholesky)
            node_q_cholesky.name = f"{node.name}/q_cholesky"

            node_x_matmul_q_cholesky = BoundMatMul(
                inputs=[node.inputs[0].inputs[0], node_q_cholesky]
            )
            node_x_matmul_q_cholesky.name = f"{node.name}/x_matmul_q_cholesky"
            node_x_matmul_q_cholesky_sqr = BoundSqr(inputs=[node_x_matmul_q_cholesky])
            node_x_matmul_q_cholesky_sqr.name = f"{node.name}/x_matmul_q_cholesky_sqr"
            node_reduce_sum = BoundReduceSum(
                inputs=[node_x_matmul_q_cholesky_sqr], attr={"axes": node.axis}
            )
            node_reduce_sum.name = f"{node.name}/reduce_sum"

            model.add_nodes(
                [
                    node_q_cholesky,
                    node_x_matmul_q_cholesky,
                    node_x_matmul_q_cholesky_sqr,
                    node_reduce_sum,
                ]
            )
            model.replace_node(node, node_reduce_sum)


def merge_quadratic_term_v2(model):
    nodes = list(model.nodes())
    for node in nodes:
        if (
            isinstance(node, BoundReduceSum)
            and isinstance(node.inputs[0], BoundMul)
            and isinstance(node.inputs[0].inputs[1], BoundMatMul)
            and node.inputs[0].inputs[1].inputs[0] == node.inputs[0].inputs[0]
            and not node.inputs[0].inputs[1].inputs[1].perturbed
        ):
            node_x = node.inputs[0].inputs[0]

            q = node.inputs[0].inputs[1].inputs[1].forward_value
            q_cholesky = torch.nn.Parameter(torch.linalg.cholesky(q))
            node_q_cholesky = BoundParams(f"{node.name}/q_cholesky", q_cholesky)
            node_q_cholesky.name = f"{node.name}/q_cholesky"

            node_x_matmul_q_cholesky = BoundMatMul(inputs=[node_x, node_q_cholesky])
            node_x_matmul_q_cholesky.name = f"{node.name}/x_matmul_q_cholesky"
            node_x_matmul_q_cholesky_sqr = BoundSqr(inputs=[node_x_matmul_q_cholesky])
            node_x_matmul_q_cholesky_sqr.name = f"{node.name}/x_matmul_q_cholesky_sqr"
            node_reduce_sum = BoundReduceSum(
                inputs=[node_x_matmul_q_cholesky_sqr], attr={"axes": node.axis}
            )
            node_reduce_sum.name = f"{node.name}/reduce_sum"

            node_quadratic = BoundQuadratic(
                inputs=[node_x, node_q_cholesky, node_reduce_sum]
            )
            node_quadratic.name = f"{node.name}/quadratic"

            model.add_nodes(
                [
                    node_q_cholesky,
                    node_x_matmul_q_cholesky,
                    node_x_matmul_q_cholesky_sqr,
                    node_reduce_sum,
                    node_quadratic,
                ]
            )
            model.replace_node(node, node_quadratic)


class BoundQuadratic(BoundActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.q = self.inputs[1].param.matmul(self.inputs[1].param.t())

    def forward(self, *x):
        return x[2]

    def interval_propagate(self, *v):
        return v[2]

    def bound_backward(self, last_lA, last_uA, x, *args, **kwargs):
        # x_m = (x.lower + x.upper) / 2
        # x_m = x.lower
        x_m = x.upper
        lower_k = 2 * x_m.matmul(self.q)
        lower_b = (-lower_k * x_m + x_m * x_m.matmul(self.q)).sum(dim=-1, keepdim=True)

        if last_lA is not None:
            last_lA_pos = last_lA.clamp(min=0)
            last_lA_neg = last_lA.clamp(max=0)
            lA_x = last_lA_pos * lower_k
            lA_reduce_sum = last_lA_neg
            lbias = (last_lA_pos * lower_b).sum(dim=-1)
        else:
            lA_x = lA_reduce_sum = None
            lbias = 0.0
        if last_uA is not None:
            last_uA_pos = last_uA.clamp(min=0)
            last_uA_neg = last_uA.clamp(max=0)
            uA_x = last_uA_neg * lower_k
            uA_reduce_sum = last_uA_pos
            ubias = (last_uA_neg * lower_b).sum(dim=-1)
        else:
            uA_x = uA_reduce_sum = None
            ubias = 0.0

        return (
            [
                (lA_x, uA_x),
                (None, None),
                (lA_reduce_sum, uA_reduce_sum),
            ],
            lbias,
            ubias,
        )
