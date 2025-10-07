#!/usr/bin/python3

from typing import NamedTuple, Any
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker
import numpy as np
import os
import math
import sys

data_dir: str = ""
gpus: list[str] = []
valueCache: dict[str, np.ndarray] = {}

def toSnakeCase(string: str):
    sanitized = "".join([c if c.isalnum() else " " for c in string])
    return "_".join([word.lower() for word in sanitized.split() if len(word) > 0])

def parseFile(path, scale):
    """
    unit of value is nanoseconds
    """
    res = np.fromfile(path, np.uint64)
    print("\n" + path + " ", scale, np.median(res) * scale, np.mean(res) * scale, np.std(res) * scale, end='\n', flush=True)
    return np.fromfile(path, np.uint64) * scale

def parseAll(gpu, name, scale):
    id = "{}_{}".format(gpu, name)

    print("Reading {}...".format(id), end='', flush=True)
    values = parseFile("{}/{}/{}.bin".format(data_dir, gpu, name), scale)
    print("done")

    return values

class CustomGraph(NamedTuple):
    title: str
    subfolder: str
    gpu: str
    names: list[str]
    markers: list[str]
    colors: list[str]
    xvalues: list[list[float]]
    yvalues: list[list[float]]
    xscale: str
    xlabel: str
    yscale: str
    ylabel: str
    fontsize: str
    cols: str
    xbase: int = 10
    ybase: int = 10
    xscale_log_clip: bool = False
    xticks: list[float] = []
    xticklabels: list[str] = []
    error: list[list[float]] = []
    ybounds: tuple[float, float] = None
    legend_title: str = None
    xticks_rotation: float = None

def plotCustomGraph(custom_graph: CustomGraph, plot: None | tuple[Any,Any] = None):
    print("Plotting custom graph '{} ({})'...".format(custom_graph.title, custom_graph.gpu))

    if plot is None:
        fig, ax = plt.subplots(1, 1, tight_layout=True)
    else:
        fig, ax = plot
    fig.set_figheight(6)

    if (len(custom_graph.error) > 0):
        for (name, marker, color, xvalues, yvalues, yerr) in zip(custom_graph.names, custom_graph.markers, custom_graph.colors, custom_graph.xvalues, custom_graph.yvalues, custom_graph.error):
            ax.errorbar(xvalues, yvalues, yerr=yerr, label=name, alpha=0.75, marker=marker, color=color, linewidth=0.75, markersize=3)
    else:
        for (name, marker, color, xvalues, yvalues) in zip(custom_graph.names, custom_graph.markers, custom_graph.colors, custom_graph.xvalues, custom_graph.yvalues):
            ax.plot(xvalues, yvalues, label=name, alpha=0.75, marker=marker, color=color, linewidth=0.75, markersize=3)

    if (all([label != None for label in custom_graph.names])):
        offset = -0.30 if (custom_graph.xticks_rotation != None and custom_graph.xlabel != "") else -0.15
        ax.legend(fontsize=13, ncols=custom_graph.cols, loc='upper center', bbox_to_anchor=(0.5, offset), title=custom_graph.legend_title, title_fontsize=13)

    ax.set(
        xlabel=custom_graph.xlabel,
        ylabel=custom_graph.ylabel,
    )
    ax.xaxis.get_label().set_fontsize(13)
    ax.yaxis.get_label().set_fontsize(13)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(13) # Size here overrides font_prop


    if (custom_graph.xscale_log_clip):
        ax.set_xscale(custom_graph.xscale, base=custom_graph.xbase, nonpositive='clip')
    elif (custom_graph.xscale != 'linear'):
        ax.set_xscale(custom_graph.xscale, base=custom_graph.xbase)
    else:
        ax.set_xscale(custom_graph.xscale)

    if (len(custom_graph.xticklabels) > 0):
        ax.set_xticks(custom_graph.xticks)

        if (custom_graph.xticks_rotation == None):
            ax.set_xticklabels(custom_graph.xticklabels)
        else:
            ax.set_xticklabels(custom_graph.xticklabels, rotation=custom_graph.xticks_rotation)

        if (custom_graph.xscale == 'log'):
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FixedFormatter(custom_graph.xticklabels))

    if (custom_graph.yscale == 'log'):
        subs = None
        if (custom_graph.ybase == 10):
            subs = [2, 3, 4, 5, 6, 7, 8, 9]
        ax.set_yscale(custom_graph.yscale, base=custom_graph.ybase, subs=subs)
    else:
        ax.set_yscale(custom_graph.yscale, base=custom_graph.ybase)

    if (custom_graph.ybounds != None):
        ax.set_ylim(custom_graph.ybounds[0], custom_graph.ybounds[1])

    if plot is None:
        os.makedirs(name=os.path.join('.', 'plots', custom_graph.subfolder, custom_graph.gpu), exist_ok=True)
        print(f"writing to file {"./plots/{}/{}/custom_graph_{}.svg".format(custom_graph.subfolder, custom_graph.gpu, toSnakeCase(custom_graph.title))}")
        fig.savefig("./plots/{}/{}/custom_graph_{}.svg".format(custom_graph.subfolder, custom_graph.gpu, toSnakeCase(custom_graph.title)))
        plt.close(fig)
    print("done")

def plot_baseline(plot_combined: bool = False):
    if plot_combined:
        plot = plt.subplots(1, 1, tight_layout=True)
    else:
        plot = None

    colorcounter = 1

    for gpu in gpus:
        data = parseAll(gpu, "bm_contention_baseline_add_relaxed_device_long", 1.0 / 1024.0)
        graph_1024 = CustomGraph(
            title="Mean Time Per Op. (add_relaxed-device), Baseline, Individual Runs (1024 Ops.)",
            subfolder="add_relaxed/device",
            gpu=gpu,
            names=[f"{gpu}-long"],
            markers=['o'],
            colors=[f"C{colorcounter}"],
            xvalues=[[i for i in range(1, len(data) + 1)]],
            yvalues=[data],
            xscale="linear",
            xbase=10,
            xlabel="Run",
            yscale="log",
            ylabel="Mean Time Per Operation [ns]",
            fontsize="x-small",
            cols=2,
            error=[],
            ybounds=(0.9, 200)
        )
        plotCustomGraph(graph_1024, plot)

        data = parseAll(gpu, "bm_contention_baseline_add_relaxed_device_short", 1.0 / 128.0)
        graph = CustomGraph(
            title="Mean Time Per Op. (add_relaxed-device), Baseline, Individual Runs",
            subfolder="add_relaxed/device",
            gpu=gpu,
            names=[f"{gpu}-short"],
            markers=['x'],
            colors=[f"C{colorcounter}"],
            xvalues=[[i for i in range(1, len(data) + 1)]],
            yvalues=[data],
            xscale="linear",
            xbase=10,
            xlabel="Run",
            yscale="log",
            ylabel="Mean Time Per Operation [ns]",
            fontsize="x-small",
            cols=2,
            error=[],
            ybounds=(0.9, 200)
        )
        plotCustomGraph(graph, plot)

        if plot is not None:
            colorcounter += 1

    if plot_combined:
        os.makedirs(name=os.path.join('.', 'plots', "add_relaxed", "device", "combined"), exist_ok=True)
        plot[0].savefig("./plots/add_relaxed/device/{}/baseline_combined_{}.svg".format("combined", toSnakeCase("Mean Time Per Op. (add_relaxed-device), Combined Baseline, Individual Runs")))
        plt.close(plot[0])

def plot_contention():
    strides = [0, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    groupings = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    large_target = 1024 * 1024 * 32

    for gpu in gpus:
        add_stride_mean = CustomGraph(
            title="Mean Time Per Fetch-Add (add_relaxed), Variable Grouping and Stride",
            subfolder="add_relaxed",
            gpu=gpu,
            names=[str(group) for group in groupings],
            markers=['o'] * len(groupings),
            colors=["C{}".format(int(math.log2(group))) for group in groupings],
            xvalues=[list(map(lambda x: 2 if x == 0 else x, strides))] * len(groupings),
            yvalues=[[np.mean(parseAll(gpu, "bm_contention_add_relaxed_t_32_32768_m_{}_{}".format(group, stride), (32 * 32768) / large_target)) for stride in strides] for group in groupings],
            xscale="log",
            xbase=2,
            xlabel="Memory Stride [B]",
            yscale="log",
            ylabel="Mean Time Per Operation [ns]",
            fontsize="x-small",
            cols=5,
            xticks=list(map(lambda x: 2 if x == 0 else x, strides)),
            xticklabels=list(map(lambda x: "0" if x == 0 else f"{x}", strides)),
            error=[[np.std(parseAll(gpu, "bm_contention_add_relaxed_t_32_32768_m_{}_{}".format(group, stride), (32 * 32768) / large_target)) for stride in strides] for group in groupings],
            #ybounds=[10**3, 4*10**4],
            legend_title="#atomics",
            xticks_rotation=-45
        )
        plotCustomGraph(add_stride_mean)

        add_stride_mean_transposed = CustomGraph(
            title="Mean Time Per Fetch-Add (add_relaxed), Variable Grouping and Stride (Transposed)",
            subfolder="add_relaxed",
            gpu=gpu,
            names=[str(group) for group in groupings],
            markers=['o'] * len(groupings),
            colors=["C{}".format(int(math.log2(group))) for group in groupings],
            xvalues=[list(map(lambda x: 1 if x == 0 else x, strides))] * len(groupings),
            yvalues=[[np.mean(parseAll(gpu, "bm_contention_add_relaxed_t_32_32768_m_{}_{}_transposed".format(group, stride), (32 * 32768) / large_target)) for stride in strides] for group in groupings],
            xscale="log",
            xbase=2,
            xlabel="Memory Stride [B]",
            yscale="log",
            ylabel="Mean Time Per Operation [ns]",
            fontsize="x-small",
            cols=5,
            xticks=list(map(lambda x: 2 if x == 0 else x, strides)),
            xticklabels=list(map(lambda x: "0" if x == 0 else f"{x}", strides)),
            error=[[np.std(parseAll(gpu, "bm_contention_add_relaxed_t_32_32768_m_{}_{}_transposed".format(group, stride), (32 * 32768) / large_target)) for stride in strides] for group in groupings],
            #ybounds=[5*10**2, 5*10**4],
            legend_title="#atomics",
            xticks_rotation=-45
        )
        plotCustomGraph(add_stride_mean_transposed)

def plot_lane_scaling():
    def color(val):
        return {1<<(i): f"C{i}" for i in range(17)}[val]

    def all_warps(): return (1 << i for i in range(15))
    def all_lanes(): return (1 << i for i in range(6))

    def make_filename(mode, lanes, warps, groups, stride, additions: None | list[Any] = None, append_bin: bool = False):
        return f"bm_contention_{mode}_t_{lanes}_{warps}_m_{groups}_{stride}{"".join(f"_{el}" for el in additions) if additions is not None else ""}{".bin" if append_bin else ""}"

    def num_iterations_per_thread(lanes, warps, large):
        if large:
            MaxBlockSize = 1024
            MaxGridSize = 1024
        else:
            MaxBlockSize = 512
            MaxGridSize = 128
        AtomicAddMinOperations = 32

        target_value = MaxGridSize * MaxBlockSize * AtomicAddMinOperations
        total_threads = lanes * warps
        return target_value / total_threads

    for gpu in gpus:
        configs = [(num_warps, num_blocks) \
            for num_warps in all_warps() \
                for num_blocks in [1] ]

        time_count_mean = CustomGraph(
            title=f"Mean Runtime, Variable Lanes and Warps ({gpu}: add_relaxed)",
            subfolder="add_relaxed/lane_scale",
            gpu=gpu,
            names=["{}".format(warp, group) for (warp, group) in configs],
            markers=['o', '^', 's', 'P', 'X', 'D'] * 6,
            colors=[color(warp * group) for (warp, group) in configs],
            xvalues=[list(all_lanes())] * len(configs),
            yvalues=[[np.mean(parseAll(gpu, make_filename("add_relaxed", lane, warp, group, 4), 1.0)) / num_iterations_per_thread(lane, warp, True) for lane in all_lanes()] for (warp, group) in configs],
            xscale="log",
            xbase=2,
            xlabel="#(active threads/warp)",
            yscale="log",
            ylabel="Mean Runtime per Access [ns]",
            fontsize="x-small",
            cols=5,
            #ybounds=[10**1, 10**5],
            xticklabels=[str(el) for el in all_lanes()],
            xticks=list(all_lanes()),
            legend_title="#warps"
        )
        plotCustomGraph(time_count_mean)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: {} <data_path> <gpus>".format(sys.argv[0]))
        exit(-1)

    data_dir = sys.argv[1]
    gpus = str.split(sys.argv[2], ',')

    plot_baseline()
    plot_baseline(True)
    plot_contention()
    plot_lane_scaling()
