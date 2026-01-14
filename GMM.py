"""
Gaussian Mixture Model Fitting with PyTorch
=============================================
Fits a 4-mode GMM to data sampled from a 3-mode GMM.
Visualizes the fitting process during training.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class GaussianMixtureModel(nn.Module):
    """GMM as a PyTorch module with learnable parameters."""

    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.weight_logits = nn.Parameter(torch.zeros(n_components))
        self.means = nn.Parameter(torch.zeros(n_components))
        self.log_vars = nn.Parameter(torch.zeros(n_components))

    def initialize_from_data(self, X):
        """Initialize parameters based on data statistics."""
        with torch.no_grad():
            data_min, data_max = X.min().item(), X.max().item()
            init_means = torch.linspace(data_min, data_max, self.n_components + 2)[1:-1]
            self.means.copy_(init_means + torch.randn_like(init_means) * 0.1)
            self.log_vars.fill_(np.log(X.var().item() / 2))
            self.weight_logits.zero_()

    @property
    def weights(self):
        return torch.softmax(self.weight_logits, dim=0)

    @property
    def variances(self):
        return torch.exp(self.log_vars)

    def gaussian_log_pdf(self, x, mean, var):
        return -0.5 * (torch.log(2 * np.pi * var) + (x - mean) ** 2 / var)

    def gaussian_pdf(self, x, mean, var):
        """Compute PDF (not log) of Gaussian distribution."""
        return (1.0 / torch.sqrt(2 * np.pi * var)) * torch.exp(
            -0.5 * (x - mean) ** 2 / var
        )

    def neg_log_likelihood(self, x):
        """Compute negative log-likelihood using log-sum-exp for stability."""
        weights = self.weights
        variances = self.variances
        log_probs = torch.stack(
            [
                torch.log(weights[k])
                + self.gaussian_log_pdf(x, self.means[k], variances[k])
                for k in range(self.n_components)
            ],
            dim=1,
        )
        return -torch.logsumexp(log_probs, dim=1).sum() / len(x)

    def neg_likelihood(self, x, alpha: float | None = None):
        """Compute negative likelihood (product of mixture PDFs)."""
        weights = self.weights
        variances = self.variances
        mixture_pdf = torch.zeros_like(x)
        for k in range(self.n_components):
            mixture_pdf += weights[k] * self.gaussian_pdf(
                x, self.means[k], variances[k]
            )
        if alpha is not None:
            assert alpha > 0.0
            detached_mixture_pdf = mixture_pdf.detach().clone()
            detached_mixture_pdf[detached_mixture_pdf < alpha] = alpha
            mixture_pdf /= detached_mixture_pdf
        return -mixture_pdf.sum() / len(x)

    def se(self, x):
        """Compute estimated squared error."""
        weights = self.weights
        variances = self.variances

        mixture_pdf_p = torch.zeros_like(x)
        for k in range(self.n_components):
            mixture_pdf_p += weights[k] * self.gaussian_pdf(
                x, self.means[k], variances[k]
            )
        mixture_pdf_p = mixture_pdf_p.sum() / len(x)

        sampled_x = self.sample_data(len(x))
        mixture_pdf_q = torch.zeros_like(sampled_x)
        for k in range(self.n_components):
            mixture_pdf_q += weights[k] * self.gaussian_pdf(
                sampled_x, self.means[k], variances[k]
            )
        mixture_pdf_q = mixture_pdf_q.sum() / len(sampled_x)

        return mixture_pdf_q - mixture_pdf_p

    def mse_kde(self, x):
        """Compute mean squared error."""
        weights = self.weights
        variances = self.variances

        mixture_pdf = torch.zeros_like(x)
        for k in range(self.n_components):
            mixture_pdf += weights[k] * self.gaussian_pdf(
                x, self.means[k], variances[k]
            )

        # make data pdf by kernel smoothing data points
        bandwidth = 0.5 * 128 / len(x)
        data_pdf = torch.zeros_like(x)
        for xi in x:
            data_pdf += self.gaussian_pdf(x, xi, bandwidth**2 * torch.ones_like(x))
        data_pdf /= len(x)

        return ((mixture_pdf - data_pdf) ** 2).sum()

    def reinforce(self, x):
        """Compute REINFORCE loss."""
        weights = self.weights
        variances = self.variances

        # sample data points from the model
        sampled_x = self.sample_data(len(x))

        # get rewards from kde data pdf
        bandwidth = 0.5 * 128 / len(x)
        rewards = torch.zeros_like(sampled_x)
        for xi in x:
            rewards += self.gaussian_pdf(sampled_x, xi, bandwidth**2 * torch.ones_like(sampled_x))
        rewards /= len(x)

        # compute log probs of sampled data points
        log_probs = torch.stack(
            [
                torch.log(weights[k])
                + self.gaussian_log_pdf(sampled_x, self.means[k], variances[k])
                for k in range(self.n_components)
            ],
            dim=1,
        )

        log_mixture_probs = torch.logsumexp(log_probs, dim=1)
        loss = -(log_mixture_probs * rewards).sum() / len(x)
        return loss

    def get_params(self):
        """Get current parameters as numpy arrays."""
        with torch.no_grad():
            return {
                "weights": self.weights.numpy().copy(),
                "means": self.means.numpy().copy(),
                "stds": torch.sqrt(self.variances).numpy().copy(),
            }

    def sample_pdf(self, x):
        """Evaluate mixture PDF at given points."""
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            weights = self.weights
            variances = self.variances
            pdf = torch.zeros_like(x_tensor)
            for k in range(self.n_components):
                pdf += weights[k] * torch.exp(
                    self.gaussian_log_pdf(x_tensor, self.means[k], variances[k])
                )
            return pdf.numpy()

    def sample_data(self, n_samples):
        """Generate samples"""
        with torch.no_grad():
            component_assignments = np.random.choice(
                self.n_components, size=n_samples, p=self.weights.numpy()
            )
            samples = np.array(
                [
                    np.random.normal(self.means[c], np.sqrt(self.variances[c]))
                    for c in component_assignments
                ]
            )
            return torch.tensor(samples, dtype=torch.float32)


def generate_3mode_gmm_data(n_samples=1000, seed=42):
    """Generate samples from a 3-mode GMM."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    true_means = np.array([-4.0, 0.0, 5.0])
    true_stds = np.array([1.0, 1.5, 0.8])
    true_weights = np.array([0.7, 0.25, 0.05])

    component_assignments = np.random.choice(3, size=n_samples, p=true_weights)
    samples = np.array(
        [np.random.normal(true_means[c], true_stds[c]) for c in component_assignments]
    )

    return torch.tensor(samples, dtype=torch.float32), {
        "means": true_means,
        "stds": true_stds,
        "weights": true_weights,
    }


def generate_4mode_gmm_data(n_samples=1000, seed=42):
    """Generate samples from a 4-mode GMM."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    true_means = np.array([-6.0, -2.0, 2.0, 6.0])
    true_stds = np.array([0.8, 1.0, 1.0, 0.8])
    true_weights = np.array([0.4, 0.2, 0.1, 0.3])

    component_assignments = np.random.choice(4, size=n_samples, p=true_weights)
    samples = np.array(
        [np.random.normal(true_means[c], true_stds[c]) for c in component_assignments]
    )

    return torch.tensor(samples, dtype=torch.float32), {
        "means": true_means,
        "stds": true_stds,
        "weights": true_weights,
    }


def generate_4th_mode_data(n_samples=500, mean=10.0, std=0.5, seed=123):
    """Generate samples from a 4th Gaussian mode for post-training."""
    np.random.seed(seed)
    samples = np.random.normal(mean, std, size=n_samples)
    return torch.tensor(samples, dtype=torch.float32), {
        "means": np.array([mean]),
        "stds": np.array([std]),
        "weights": np.array([1.0]),
    }


def print_params(model, title):
    """Print GMM parameters."""
    print(f"\n{title}")
    print("=" * 60)
    params = model.get_params()
    order = np.argsort(params["means"])
    for idx in order:
        print(
            f"  Component {idx + 1}: w={params['weights'][idx]:.3f}, "
            f"μ={params['means'][idx]:.3f}, σ={params['stds'][idx]:.3f}"
        )


def plot_model_shapes(
    model, x_plot, true_density, data_np, n_components, epoch, title_prefix="GMM"
):
    """Plot model shapes (fitted density and components) for a given epoch."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data histogram
    ax.hist(
        data_np, bins=50, density=True, alpha=0.5, color="steelblue", edgecolor="white"
    )

    # Plot true density
    ax.plot(x_plot, true_density, "k--", linewidth=2, label="True (3-mode)")

    # Plot fitted density
    fitted_pdf = model.sample_pdf(x_plot)
    ax.plot(
        x_plot, fitted_pdf, "r-", linewidth=2, label=f"Fitted ({n_components}-mode)"
    )

    # Plot individual components
    params = model.get_params()
    colors = plt.cm.tab10(np.linspace(0, 1, n_components))
    for k in range(n_components):
        comp_pdf = (
            params["weights"][k]
            * (1.0 / np.sqrt(2 * np.pi * params["stds"][k] ** 2))
            * np.exp(-0.5 * (x_plot - params["means"][k]) ** 2 / params["stds"][k] ** 2)
        )
        ax.plot(
            x_plot,
            comp_pdf,
            "--",
            color=colors[k],
            alpha=0.7,
            label=f"Comp {k + 1}: w={params['weights'][k]:.2f}, μ={params['means'][k]:.1f}",
        )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim(x_plot.min(), x_plot.max())
    ax.set_ylim(0, max(true_density) * 1.3)
    ax.set_title(f"{title_prefix} - Epoch {epoch}", fontsize=14)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _setup_training(n_samples=2000, seed=42, underfit=False):
    """Common setup for all training functions.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed
        underfit: If True, generate 4-mode data (for 3-mode model to fit)
    """
    if underfit:
        print(f"\nGenerating {n_samples} samples from 4-mode GMM...")
        data, true_params = generate_4mode_gmm_data(n_samples, seed=seed)
        print("\nTrue 4-Mode GMM Parameters:")
        print("-" * 40)
        for i in range(4):
            print(
                f"  Component {i + 1}: w={true_params['weights'][i]:.2f}, "
                f"μ={true_params['means'][i]:.1f}, σ={true_params['stds'][i]:.1f}"
            )
    else:
        print(f"\nGenerating {n_samples} samples from 3-mode GMM...")
        data, true_params = generate_3mode_gmm_data(n_samples, seed=seed)
        print("\nTrue 3-Mode GMM Parameters:")
        print("-" * 40)
        for i in range(3):
            print(
                f"  Component {i + 1}: w={true_params['weights'][i]:.2f}, "
                f"μ={true_params['means'][i]:.1f}, σ={true_params['stds'][i]:.1f}"
            )

    return data, true_params


def _get_hyperparams():
    """Default training hyperparameters."""
    return {
        "lr": 0.01,
        "batch_size": 128,
        "max_epochs": 50,
        "plot_every": 5,
        "mixce_eta": 0.3,
        "trsft_alpha": 0.05,
        "post_epochs": 50,
    }


def main_train(objective="nll", save_plot=True, show_plot=False, underfit=False):
    """Train GMM using specified objective.

    Args:
        objective: Training objective
        save_plot: Whether to save the plot to file
        show_plot: Whether to display the plot
        underfit: If True, use 3-mode model on 4-mode data (underfitting scenario)

    Returns:
        model: Trained GaussianMixtureModel
        data: Training data tensor
        true_params: Dictionary of true parameters
    """
    # Determine model/data configuration
    if underfit:
        n_model_components = 3
        n_data_modes = 4
        scenario = "3-Mode GMM Fitting 4-Mode Data"
    else:
        n_model_components = 4
        n_data_modes = 3
        scenario = "4-Mode GMM Fitting 3-Mode Data"

    print("=" * 70)
    print(f"{scenario} - {objective.upper()}")
    print("=" * 70)

    data, true_params = _setup_training(underfit=underfit)
    hp = _get_hyperparams()
    plot_every = hp["plot_every"]

    torch.manual_seed(42)
    model = GaussianMixtureModel(n_components=n_model_components)
    model.initialize_from_data(data)

    # Compute plotting data
    data_np = data.numpy()
    x_plot = np.linspace(data_np.min() - 2, data_np.max() + 2, 500)
    true_density = np.zeros_like(x_plot)
    for mean, std, weight in zip(
        true_params["means"], true_params["stds"], true_params["weights"]
    ):
        true_density += (
            weight
            * (1.0 / np.sqrt(2 * np.pi * std**2))
            * np.exp(-0.5 * (x_plot - mean) ** 2 / std**2)
        )

    optimizer = optim.Adam(model.parameters(), lr=hp["lr"])
    n_samples = len(data)

    # Collect snapshots for plotting
    snapshots = []

    for epoch in range(hp["max_epochs"]):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        for start_idx in range(0, n_samples, hp["batch_size"]):
            batch = data[perm[start_idx : min(start_idx + hp["batch_size"], n_samples)]]
            optimizer.zero_grad()

            # Choose objective
            if objective == "nll":
                loss = model.neg_log_likelihood(batch)
            elif objective == "nl":
                loss = model.neg_likelihood(batch)
            elif objective == "se":
                loss = model.se(batch)
            elif objective == "mse_kde":
                loss = model.mse_kde(batch)
            elif objective == "mixce":
                loss = hp["mixce_eta"] * model.neg_log_likelihood(batch) + (1 - hp["mixce_eta"]) * model.neg_likelihood(batch)
            elif objective == "trsft":
                loss = model.neg_likelihood(batch, hp["trsft_alpha"])
            elif objective == "reinforce":
                loss = model.reinforce(batch)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % plot_every == 0:
            print(f"Epoch {epoch + 1}: loss = {epoch_loss / n_batches:.4f}")
            snapshots.append(
                {
                    "epoch": epoch + 1,
                    "params": model.get_params(),
                    "fitted_pdf": model.sample_pdf(x_plot).copy(),
                }
            )

    # Plot all epochs in one figure
    n_plots = len(snapshots)
    n_cols = 5
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i, snapshot in enumerate(snapshots):
        ax = axes[i]
        ax.hist(
            data_np,
            bins=50,
            density=True,
            alpha=0.5,
            color="steelblue",
            edgecolor="white",
        )
        true_label = f"True ({n_data_modes}-mode)" if underfit else "True"
        fitted_label = f"Fitted ({n_model_components}-mode)" if underfit else "Fitted"
        ax.plot(x_plot, true_density, "k--", linewidth=2, label=true_label)
        ax.plot(x_plot, snapshot["fitted_pdf"], "r-", linewidth=2, label=fitted_label)
        params = snapshot["params"]
        colors = plt.cm.tab10(np.linspace(0, 1, n_model_components))
        for k in range(n_model_components):
            comp_pdf = (
                params["weights"][k]
                * (1.0 / np.sqrt(2 * np.pi * params["stds"][k] ** 2))
                * np.exp(
                    -0.5 * (x_plot - params["means"][k]) ** 2 / params["stds"][k] ** 2
                )
            )
            ax.plot(x_plot, comp_pdf, "--", color=colors[k], alpha=0.7)
        ax.set_title(f"Epoch {snapshot['epoch']}", fontsize=12)
        ax.set_xlim(x_plot.min(), x_plot.max())
        ax.set_ylim(0, max(true_density) * 1.3)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    title_suffix = " (Underfit)" if underfit else ""
    fig.suptitle(f"{objective.upper()} Training Progress{title_suffix}", fontsize=16)
    plt.tight_layout()
    if save_plot:
        os.makedirs("images", exist_ok=True)
        if objective == "mixce":
            filename = f"images/gmm_underfit_{objective}_eta{hp['mixce_eta']}_epochs{hp['max_epochs']}_training.png" if underfit else f"images/gmm_{objective}_eta{hp['mixce_eta']}_epochs{hp['max_epochs']}_training.png"
        elif objective == "trsft":
            filename = f"images/gmm_underfit_{objective}_alpha{hp['trsft_alpha']}_epochs{hp['max_epochs']}_training.png" if underfit else f"images/gmm_{objective}_alpha{hp['trsft_alpha']}_epochs{hp['max_epochs']}_training.png"
        else:
            filename = f"images/gmm_underfit_{objective}_epochs{hp['max_epochs']}_training.png" if underfit else f"images/gmm_{objective}_epochs{hp['max_epochs']}_training.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {filename}")
    if show_plot:
        plt.show()
    else:
        plt.close()

    print_params(model, f"{objective.upper()} Final Parameters")
    return model, data, true_params


def main_post_training(
    objective="nll",
    fourth_mode_mean=1,
    fourth_mode_std=1,
    fourth_mode_samples=500,
):
    """
    Post-training experiment: Train on 3-mode data, then continue training on 4th mode.

    Args:
        post_epochs: Number of epochs for post-training
        fourth_mode_mean: Mean of the 4th Gaussian mode
        fourth_mode_std: Std of the 4th Gaussian mode
        fourth_mode_samples: Number of samples from 4th mode
    """
    print("=" * 70)
    print("POST-TRAINING EXPERIMENT")
    print(f"Pre-train on 3-mode data, then continue on 4th mode (μ={fourth_mode_mean})")
    print("=" * 70)

    # ========== Phase 1: NLL Pre-training on 3-mode data ==========
    model, data_3mode, true_params_3mode = main_train(
        objective="nll", save_plot=False, show_plot=False
    )
    data_np_pre = data_3mode.numpy()
    hp = _get_hyperparams()
    post_epochs = hp["post_epochs"]
    plot_every = hp["plot_every"]

    # ========== Phase 2: Post-training on 4th mode data ==========
    print("\n" + "=" * 70)
    print(
        f"PHASE 2: Post-training on 4th mode (μ={fourth_mode_mean}, σ={fourth_mode_std})"
    )
    print("=" * 70)

    data_4th, params_4th = generate_4th_mode_data(
        n_samples=fourth_mode_samples, mean=fourth_mode_mean, std=fourth_mode_std
    )

    # Combined true params for visualization (3 original + 1 new)
    combined_true_params = {
        "means": np.concatenate([true_params_3mode["means"], params_4th["means"]]),
        "stds": np.concatenate([true_params_3mode["stds"], params_4th["stds"]]),
        "weights": np.concatenate(
            [
                true_params_3mode["weights"] * 0.8,  # Scale down original
                np.array([0.2]),  # New mode weight
            ]
        ),
    }

    # Compute plotting data for post-training (extended range)
    data_np_4th = data_4th.numpy()
    x_plot_post = np.linspace(
        min(data_np_pre.min(), data_np_4th.min()) - 2,
        max(data_np_pre.max(), data_np_4th.max()) + 2,
        500,
    )
    true_density_post = np.zeros_like(x_plot_post)
    for mean, std, weight in zip(
        combined_true_params["means"],
        combined_true_params["stds"],
        combined_true_params["weights"],
    ):
        true_density_post += (
            weight
            * (1.0 / np.sqrt(2 * np.pi * std**2))
            * np.exp(-0.5 * (x_plot_post - mean) ** 2 / std**2)
        )

    optimizer = optim.Adam(model.parameters(), lr=hp["lr"])
    n_samples_4th = len(data_4th)

    # Collect snapshots for post-training
    snapshots_post = []

    for epoch in range(post_epochs):
        perm = torch.randperm(n_samples_4th)
        epoch_loss = 0.0
        n_batches = 0
        batch_size = min(hp["batch_size"], fourth_mode_samples)
        for start_idx in range(0, n_samples_4th, batch_size):
            batch = data_4th[
                perm[start_idx : min(start_idx + batch_size, n_samples_4th)]
            ]
            optimizer.zero_grad()

            # Choose objective
            if objective == "nll":
                loss = model.neg_log_likelihood(batch)
            elif objective == "nl":
                loss = model.neg_likelihood(batch)
            elif objective == "se":
                loss = model.se(batch)
            elif objective == "mse_kde":
                loss = model.mse_kde(batch)
            elif objective == "mixce":
                loss = hp["mixce_eta"] * model.neg_log_likelihood(batch) + (1 - hp["mixce_eta"]) * model.neg_likelihood(batch)
            elif objective == "trsft":
                loss = model.neg_likelihood(batch, hp["trsft_alpha"])
            elif objective == "reinforce":
                loss = model.reinforce(batch)
            else:
                raise ValueError(f"Unknown objective: {objective}")

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % plot_every == 0:
            print(f"Post-train Epoch {epoch + 1}: loss = {epoch_loss / n_batches:.4f}")
            snapshots_post.append(
                {
                    "epoch": epoch + 1,
                    "params": model.get_params(),
                    "fitted_pdf": model.sample_pdf(x_plot_post).copy(),
                }
            )

    # Plot post-training progress
    n_plots = len(snapshots_post)
    n_cols = 5
    n_rows = max(1, (n_plots + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Compute global y max across all snapshots for consistent ylim
    global_y_max = max(true_density_post)
    for snapshot in snapshots_post:
        global_y_max = max(global_y_max, max(snapshot["fitted_pdf"]))

    for i, snapshot in enumerate(snapshots_post):
        ax = axes[i]
        ax.hist(
            data_np_pre,
            bins=50,
            density=True,
            alpha=0.3,
            color="steelblue",
            edgecolor="white",
            label="Original",
        )
        ax.hist(
            data_np_4th,
            bins=30,
            density=True,
            alpha=0.5,
            color="orange",
            edgecolor="white",
            label="4th mode",
        )
        ax.plot(x_plot_post, true_density_post, "k--", linewidth=2, label="True")
        ax.plot(x_plot_post, snapshot["fitted_pdf"], "r-", linewidth=2, label="Fitted")
        params = snapshot["params"]
        colors = plt.cm.tab10(np.linspace(0, 1, 4))
        for k in range(4):
            comp_pdf = (
                params["weights"][k]
                * (1.0 / np.sqrt(2 * np.pi * params["stds"][k] ** 2))
                * np.exp(
                    -0.5
                    * (x_plot_post - params["means"][k]) ** 2
                    / params["stds"][k] ** 2
                )
            )
            ax.plot(x_plot_post, comp_pdf, "--", color=colors[k], alpha=0.7)
        ax.set_title(f"Post Epoch {snapshot['epoch']}", fontsize=12)
        ax.set_xlim(x_plot_post.min(), x_plot_post.max())
        ax.set_ylim(0, 0.5)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Post-training Progress ({objective.upper()})", fontsize=16)
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    if objective == "mixce":
        filename = f"images/gmm_post_{objective}_eta{hp['mixce_eta']}_epochs{post_epochs}_posttrain.png"
    elif objective == "trsft":
        filename = f"images/gmm_post_{objective}_alpha{hp['trsft_alpha']}_epochs{post_epochs}_posttrain.png"
    else:
        filename = f"images/gmm_post_{objective}_epochs{post_epochs}_posttrain.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")

    print_params(model, "Post-training Final Parameters")

    return model


if __name__ == "__main__":
    # main_train(objective="mixce", underfit=True)
    # for objective in ["nll", "nl", "se", "mixce", "trsft", "reinforce"]:
    #     main_train(objective=objective, underfit=True)
    # for objective in ["nll", "nl", "se", "mixce", "trsft", "reinforce"]:
    #     main_post_training(objective=objective)
    main_post_training(objective="trsft")
