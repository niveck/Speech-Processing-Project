import pandas as pd
import matplotlib.pyplot as plt


def llm_performance_graph(all_rows_raw_text):
    rows = all_rows_raw_text.split("\n")
    rows = rows[1:]  # remove first line with column titles
    rows_by_epoch = {}
    epoch = "0"
    for row in rows:
        if "Epoch:" in row:
            epoch = row
            rows_by_epoch[epoch] = []
            continue
        rows_by_epoch[epoch].append(row.split(",\t"))
    ctc_model_original_exact_match_accuracy_by_epoch = {}
    llm_exact_match_accuracy_by_epoch = {}
    for epoch, epoch_rows in rows_by_epoch.items():
        if not epoch_rows:
            continue
        ctc_model_original_exact_match_accuracy_by_epoch[epoch] = 0
        llm_exact_match_accuracy_by_epoch[epoch] = 0
        for ctc_model_output, llm_output, true_label, my_assertion in epoch_rows:
            ctc_model_original_exact_match_accuracy_by_epoch[epoch] += int(ctc_model_output == true_label)
            llm_exact_match_accuracy_by_epoch[epoch] += int(llm_output == true_label)
            assert bool(llm_output == true_label) == eval(my_assertion)
        ctc_model_original_exact_match_accuracy_by_epoch[epoch] /= len(epoch_rows)
        llm_exact_match_accuracy_by_epoch[epoch] /= len(epoch_rows)
    return rows_by_epoch, ctc_model_original_exact_match_accuracy_by_epoch, llm_exact_match_accuracy_by_epoch


def plot_metric_across_epochs(results_csv_path, metric_name,
                              with_last_experiment=False, start_from_epoch=0,
                              ylim=None):
    df = pd.read_csv(results_csv_path)
    df = df[df["epoch"] >= start_from_epoch]
    title = f"{metric_name} Across Training Epochs"
    plt.title(title)
    columns_to_plot = list(df.columns)[1:]  # index 0 is "epoch"
    if not with_last_experiment:
        columns_to_plot = columns_to_plot[:-1]
    for col in columns_to_plot:
        plt.plot(df["epoch"], df[col], label=col)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig("./graphs/" + title.lower().replace(" ", "_") +
                ("with_last" if with_last_experiment else "") + f"_{start_from_epoch}.png")
    plt.cla()


def main():
    plot_metric_across_epochs("./results/loss.csv", "Loss")
    plot_metric_across_epochs("./results/loss.csv", "Loss", start_from_epoch=6, ylim=(0, 1.5))
    plot_metric_across_epochs("./results/loss.csv", "Loss", with_last_experiment=True)
    plot_metric_across_epochs("./results/loss.csv", "Loss", start_from_epoch=6, with_last_experiment=True, ylim=(0, 1.5))
    plot_metric_across_epochs("./results/accuracy.csv", "Accuracy", ylim=(0, 1))
    plot_metric_across_epochs("./results/accuracy.csv", "Accuracy", start_from_epoch=6, ylim=(0, 1))
    plot_metric_across_epochs("./results/accuracy.csv", "Accuracy", with_last_experiment=True, ylim=(0, 1))
    plot_metric_across_epochs("./results/accuracy.csv", "Accuracy", start_from_epoch=6, with_last_experiment=True, ylim=(0, 1))
    plot_metric_across_epochs("./results/exact_match_accuracy.csv", "Exact-Match Accuracy", ylim=(0, 0.35))
    plot_metric_across_epochs("./results/exact_match_accuracy.csv", "Exact-Match Accuracy", start_from_epoch=6, ylim=(0, 0.35))
    plot_metric_across_epochs("./results/exact_match_accuracy.csv", "Exact-Match Accuracy", with_last_experiment=True, ylim=(0, 0.35))
    plot_metric_across_epochs("./results/exact_match_accuracy.csv", "Exact-Match Accuracy", start_from_epoch=6, with_last_experiment=True, ylim=(0, 0.35))
    plot_metric_across_epochs("./results/labeling_accuracy.csv", "Labeling Accuracy", with_last_experiment=True)


if __name__ == '__main__':
    main()
