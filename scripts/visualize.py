import matplotlib.pyplot as plt
import torch


def visualize_lattice(U: torch.Tensor, title: str = None):
    """Visualise a 2D Z₂ link configuration.

    Parameters
    ----------
    U
        Link tensor of shape ``(D=2, L, L, nc=1, nc=1)`` or ``(D=2, L, L)``.
        Green = +1, red = −1.
    """
    if U.ndim == 5:
        # (D, L, L, 1, 1) → (D, L, L)
        U = U.squeeze(-1).squeeze(-1)
    assert U.ndim == 3 and U.shape[0] == 2, (
        f"Expected shape (2, L, L); got {tuple(U.shape)}"
    )

    L = U.shape[1]
    fig, ax = plt.subplots(figsize=(max(4, L), max(4, L)))

    for x in range(L):
        for y in range(L):
            for direction, (dx, dy) in enumerate([(1, 0), (0, 1)]):
                val = U[direction, x, y].item()
                color = "tab:green" if val > 0 else "tab:red"
                ax.plot([x, x + dx], [y, y + dy], color=color, lw=2.5)

    for x in range(L):
        for y in range(L):
            ax.plot(x, y, "ko", ms=7, zorder=5)

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, L)
    ax.set_ylim(-0.5, L)
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from lgt.lattice import Z2, random_links

    U = random_links(L=5, D=2, group=Z2())
    visualize_lattice(U)
