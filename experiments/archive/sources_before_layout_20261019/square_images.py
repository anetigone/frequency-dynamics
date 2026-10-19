"""Real spatial kernels, periodic images, exact moments and empirical GUPA.

The kernel basis is orthonormal in REAL pixel coordinates. Circular correlation
is divided by sqrt(H*W/2), so B_ck = g_k*(h_cos**2+h_sin**2).
This makes pixel-kernel SGD have exactly the paper's half-spectrum metric.
No bias, normalization layers, output normalization or hidden regularization.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def real_basis(size, modes, *, dtype=torch.float64, device="cpu"):
    yy, xx = torch.meshgrid(torch.arange(size, dtype=dtype, device=device),
                            torch.arange(size, dtype=dtype, device=device), indexing="ij")
    waves = torch.stack([2 * math.pi * (kx * xx + ky * yy) / size for kx, ky in modes])
    return torch.stack((waves.cos(), waves.sin()), dim=1) * math.sqrt(2 / size**2)


class SquareImages(nn.Module):
    def __init__(self, size, modes, g, channels, outputs=1, kernel_size=None,
                 dtype=torch.float64, device="cpu", backend="spatial"):
        super().__init__()
        self.size, self.backend = size, backend
        self.modes = tuple(tuple(m) for m in modes)
        self.register_buffer("basis", real_basis(size, modes, dtype=dtype, device=device))
        self.register_buffer("g", torch.as_tensor(g, dtype=dtype, device=device))
        k = kernel_size or size
        self.kernel = nn.Parameter(torch.zeros(channels, 1, k, k, dtype=dtype, device=device))
        self.weight = nn.Parameter(torch.zeros(outputs, channels, dtype=dtype, device=device))
        # Each pure cosine/sine probe has frequency energy exactly g_k.
        probes = self.basis * size * self.g.sqrt()[:, None, None, None]
        self.register_buffer("probes", probes.reshape(-1, 1, size, size))

    def features(self, images):
        k = self.kernel.shape[-1]
        if self.backend == "spatial":
            padded = F.pad(images, (0, k - 1, 0, k - 1), mode="circular")
            response = F.conv2d(padded, self.kernel)
        elif self.backend == "fft":
            padded_kernel = F.pad(self.kernel[:, 0], (0, self.size - k, 0, self.size - k))
            xf = torch.fft.rfft2(images[:, 0])
            hf = torch.fft.rfft2(padded_kernel)
            response = torch.fft.irfft2(xf[:, None] * hf.conj()[None],
                                       s=(self.size, self.size))
        else:
            raise ValueError(self.backend)
        response = response / math.sqrt(self.size**2 / 2)
        return response.square().mean(dim=(-2, -1))

    def forward(self, images):
        return self.features(images) @ self.weight.T

    def effective(self):
        # Actual image -> convolution -> square -> GAP, with no Fourier shortcut.
        b = self.features(self.probes).reshape(len(self.g), 2, -1).mean(dim=1).T
        return self.weight @ b, b

    def spectral_coefficients(self):
        if self.kernel.shape[-1] != self.size:
            raise ValueError("Full-support kernels required for this metric check")
        return torch.einsum("cij,dpij->cdp", self.kernel[:, 0], self.basis)

    @torch.no_grad()
    def set_spectral(self, coefficients, weight):
        if self.kernel.shape[-1] != self.size:
            raise ValueError("Full-support initialization required")
        self.kernel[:, 0].copy_(torch.einsum("cdp,dpij->cij", coefficients, self.basis))
        self.weight.copy_(weight)

    def images(self, coefficients):
        # iid N(0,1) coefficients give independent CN power coordinates q_k.
        scaled = coefficients * self.g.sqrt()[None, :, None] * (self.size / math.sqrt(2))
        return torch.einsum("bdp,dpij->bij", scaled, self.basis)[:, None]


def population_loss(a, noise=0., s=None, d=None):
    if s is None:
        s = torch.full_like(a[0], (1 + noise)**2)
        d = 2 * (s - 1)
    covariance = (a * s) @ a.T
    return gupa(covariance, .5 * (a.square() * d).sum())


def gupa(covariance, alignment):
    eye = torch.eye(len(covariance), dtype=covariance.dtype, device=covariance.device)
    shifted = eye + covariance  # epsilon=1; keep both uniformity terms.
    return alignment - .5 * torch.linalg.slogdet(shifted)[1] - .5 * torch.trace(
        torch.linalg.solve(shifted, covariance))


def empirical_loss(z1, z2):
    n = len(z1)
    if n < 2:
        raise ValueError("Covariance requires batch size >= 2")
    c1, c2 = z1 - z1.mean(0), z2 - z2.mean(0)
    covariance = (c1.T @ c1 + c2.T @ c2) / (2 * (n - 1))
    return gupa(covariance, .5 * (z1 - z2).square().sum(1).mean())


def excess_risks(a, tasks):
    """SVD row-space risk; no ridge. tasks is [number_of_tasks, d]."""
    _, values, vh = torch.linalg.svd(a, full_matrices=False)
    tolerance = torch.finfo(a.dtype).eps * max(a.shape) * values.max()
    rank = int((values > tolerance).sum())
    retained = (tasks @ vh[:rank].T).square().sum(1)
    return (tasks.square().sum(1) - retained).clamp_min(0), values, rank


def smooth_risk(a, beta):
    # For intervention derivatives only: requires full row rank, checked by caller.
    signal = a @ beta
    return beta.square().sum() - signal @ torch.linalg.solve(a @ a.T, signal)


def texture_spec():
    modes, groups = [], []
    for direction in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        for radii in [(2, 3), (6, 7)]:
            group = []
            for r in radii:
                group.append(len(modes))
                modes.append((r * direction[0], r * direction[1]))
            groups.append(group)
    return modes, groups


def texture_tasks(dtype, device):
    _, groups = texture_spec()
    tasks = torch.zeros(10, 16, dtype=dtype, device=device)
    tasks[0, 0], tasks[1, 4] = 1, 1
    names = ["single_x_low", "single_y_low"]
    for i, indices in enumerate(groups):
        tasks[i + 2, indices] = 1 / math.sqrt(len(indices))
        names.append(["x", "y", "diagonal", "antidiagonal"][i // 2] +
                     ("_low" if i % 2 == 0 else "_high"))
    return names, tasks
