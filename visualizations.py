import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_experiment_json(
    strategy: str,
    seat_time_dist: str,
    results_dir: str = os.path.join("results", "json"),
) -> pd.DataFrame:
    """
    Loads the JSON file for the given boarding strategy and seat time distribution, parses it to a pandas DataFrame, and returns it.
    """
    filename = f"{strategy}_{seat_time_dist}.json"
    path = os.path.join(results_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def plot_seat_time_histogram(
    df: pd.DataFrame, strategy: str, seat_time_dist: str, output_dir: str = "plots"
):
    """
    Plots an aggregate histogram of the total time needed to take a seat (seat_end_t - enter_t) over all replications.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    seat_times = df["seat_end_t"] - df["enter_t"]
    plt.figure(figsize=(8, 6))
    sns.histplot(seat_times, bins=30, kde=True, color="skyblue")
    plt.title(
        f"Seat Time Distribution\nStrategy: {strategy}, Seat Time Dist: {seat_time_dist}"
    )
    plt.xlabel("Time to Take Seat (s)")
    plt.ylabel("Passenger Count")
    plt.tight_layout()
    out_file = os.path.join(
        output_dir, f"hist_seat_time_{strategy}_{seat_time_dist}.png"
    )
    plt.savefig(out_file)
    plt.close()


def plot_avg_seat_time_histogram(
    df: pd.DataFrame, strategy: str, seat_time_dist: str, ax=None
):
    """
    Plots a histogram where each bin's height is the average count across all replications for that bin (i.e., average histogram over replications).
    If ax is provided, plot on that axis; otherwise, create a new figure.
    """
    df["seat_time"] = df["seat_end_t"] - df["enter_t"]
    # Get all seat times and replications
    replications = df["replication"].unique()
    seat_times = df["seat_time"]
    # Compute global bin edges
    bin_count = 20
    min_time = seat_times.min()
    max_time = seat_times.max()
    bins = np.linspace(min_time, max_time, bin_count + 1)
    # For each replication, compute histogram
    histograms = []
    for rep in replications:
        rep_seat_times = df[df["replication"] == rep]["seat_time"]
        hist, _ = np.histogram(rep_seat_times, bins=bins)
        histograms.append(hist)
    avg_hist = np.mean(histograms, axis=0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        bin_centers,
        avg_hist,
        width=(bins[1] - bins[0]),
        color="mediumseagreen",
        alpha=0.8,
        edgecolor="black",
        label="Mean",
    )
    ax.set_title(f"{strategy}, {seat_time_dist}")
    ax.set_xlabel("Time to Take Seat (s)")
    ax.set_ylabel("Avg Passenger Count per Bin")
    ax.legend()


# Columns [base_seat_time,interference_penalty,luggage_add_time,p_luggage,strategy,reps,mean_time_s,sd_s,ci95_halfwidth_s]
def plot_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_file: str,
):
    pivot_table = df.pivot_table(index=y_col, columns=x_col, values=value_col)
    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(pivot_table, cmap="viridis", aspect="auto", origin="lower")
    plt.colorbar(heatmap, label=value_col)
    plt.xticks(ticks=range(len(pivot_table.columns)), labels=pivot_table.columns)
    plt.yticks(ticks=range(len(pivot_table.index)), labels=pivot_table.index)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Plot value in every cell
    for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
            value = pivot_table.iloc[i, j]
            plt.text(
                j,
                i,
                f"{value:.2f}" if pd.notnull(value) else "",
                ha="center",
                va="center",
                color="white"
                if heatmap.cmap(heatmap.norm(value))[:3] < (0.5, 0.5, 0.5)
                else "black",
            )

    plt.savefig(os.path.join("plots", output_file))
    plt.close()


if __name__ == "__main__":
    # data_file = os.path.join("results", "task2_sweep_seat_times.csv")
    # df = pd.read_csv(data_file)

    # plot_heatmap(
    #     df,
    #     x_col="base_seat_time",
    #     y_col="interference_penalty",
    #     value_col="mean_time_s",
    #     title="Mean Boarding Time vs Seat Time Parameters",
    #     xlabel="Base Seat Time",
    #     ylabel="Interference Penalty",
    #     output_file="mean_boarding_time_heatmap.png",
    # )

    # plot_heatmap(
    #     df,
    #     x_col="base_seat_time",
    #     y_col="interference_penalty",
    #     value_col="ci95_halfwidth_s",
    #     title="95% CI Half-Width vs Seat Time Parameters",
    #     xlabel="Base Seat Time",
    #     ylabel="Interference Penalty",
    #     output_file="ci95_halfwidth_heatmap.png",
    # )

    # Load experiment JSON data
    
    
    strategies = ["random", "wilma", "back_to_front", "steffen_boarding"]
    seat_time_dists = ["exponential", "lognormal", "triangular"]
    n_rows = len(strategies)
    n_cols = len(seat_time_dists)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    for i, strategy in enumerate(strategies):
        for j, seat_time_dist in enumerate(seat_time_dists):
            df = load_experiment_json(strategy, seat_time_dist)
            ax = axes[i, j] if n_rows > 1 and n_cols > 1 else axes[max(i, j)]
            plot_avg_seat_time_histogram(
                df,
                strategy=strategy,
                seat_time_dist=seat_time_dist,
                ax=ax,
            )
    plt.tight_layout()
    fig.savefig(os.path.join("plots", "avg_hist_seat_time_grid.png"))
    plt.close(fig)
