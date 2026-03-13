import torch
import torch.nn as nn


class mHC(nn.Module):
    """Manifold-Constrained Hyper-Connections.

    Expected input shape: (batch_size, n_streams, d_model).
    """

    def __init__(self, n_streams, d_model, funcional_block, num_iters=20):
        super().__init__()
        self.n = n_streams
        self.C = d_model
        self.F = funcional_block

        flat_dim = self.n * self.C

        # Dynamic projection parameters from vec(x_l) in R^(n*C).
        self.phi_pre  = nn.Parameter(torch.randn(flat_dim, self.n) * 0.02)
        self.phi_post = nn.Parameter(torch.randn(flat_dim, self.n) * 0.02)
        self.phi_res  = nn.Parameter(torch.randn(flat_dim, self.n * self.n) * 0.02)

        # Static biases.
        self.bias_pre  = nn.Parameter(torch.zeros(self.n))
        self.bias_post = nn.Parameter(torch.zeros(self.n))
        self.bias_res  = nn.Parameter(torch.zeros(self.n, self.n))

        # Learnable gating factors
        self.alpha_pre  = torch.nn.Parameter(torch.tensor(0.1))
        self.alpha_post = torch.nn.Parameter(torch.tensor(0.1))
        self.alpha_res  = torch.nn.Parameter(torch.tensor(0.1))

        self.num_iter = num_iters

    def sinkhorn_knopp(self, M_logits):
        # exp(.) enforces positivity before alternating row/column normalization.
        M = torch.exp(M_logits)
        for _ in range(self.num_iter):
            M = M / (M.sum(dim=2, keepdim=True) + 1e-12)
            M = M / (M.sum(dim=1, keepdim=True) + 1e-12)
        return M

    def forward(self, x):
        # x shape: (batch_size, n_streams, d_model)
        batch_size = x.size(0)

        # vec(x_l) and RMSNorm over flattened residual stream
        x_flat = x.reshape(batch_size, self.flat_dim)
        rms    = torch.sqrt(torch.mean(x_flat.pow(2), dim=1, keepdim=True) + 1e-8)
        x_norm = x_flat / rms

        # Dynamic mapping logits
        H_pre_tilde  = self.alpha_pre  * (x_norm @ self.phi_pre) + self.bias_pre
        H_post_tilde = self.alpha_post * (x_norm @ self.phi_post) + self.bias_post
        H_res_tilde  = self.alpha_res  * (x_norm @ self.phi_res)
        
        H_res_tilde  = H_res_tilde.reshape(batch_size, self.n, self.n) + self.bias_res

        # Constrained mappings
        H_pre  = torch.sigmoid(H_pre_tilde).unsqueeze(1)            # (B, 1, n)
        H_post = (2.0 * torch.sigmoid(H_post_tilde)).unsqueeze(-1)  # (B, n, 1)
        H_res  = self.sinkhorn_knopp(H_res_tilde)                   # (B, n, n)

        # Aggregate n streams to a single functional input
        function_input = torch.bmm(H_pre, x)

        # Pass through the functional block
        function_output = self.F(function_input)
        if function_output.dim() == 2:
            function_output = function_output.unsqueeze(1)

        # Distribute block output back to n streams.
        update = H_post * function_output

        # Residual transport through doubly stochastic mapping.
        residuals = torch.bmm(H_res, x)

        return residuals + update