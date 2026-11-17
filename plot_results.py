"""Plot all paired held-out scores."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / "results/paired-results.csv").open() as stream:
    rows = list(csv.DictReader(stream))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
fig.patch.set_facecolor("#faf9f6")
fig.subplots_adjust(left=.12, right=.96, bottom=.16, top=.77, wspace=.32)
fig.text(.035,.94,"One correction, six paired comparisons",fontsize=18,weight="bold",va="center",color="#172936")
fig.text(.035,.86,"Each row uses the same split, starting weights, and sequence of mini-batches.",fontsize=9.2,color="#475b67")
for ax,suffix,title,limits in zip(axes,["ewt_pct","mae_mev"],["EwT (%)  ·  higher is better","MAE (meV)  ·  lower is better"],[(0,100),(0,50)]):
    ax.set_facecolor("#faf9f6"); ax.set_axisbelow(True); ax.grid(axis="x",color="#dfdfd8",linewidth=.7)
    for y,r in enumerate(rows):
        before,after=(float(r[f"{v}_{suffix}"]) for v in ("baseline","corrected"))
        ax.plot([before,after],[y,y],color="#b4c9c3",lw=3,solid_capstyle="round")
        ax.scatter(before,y,s=45,marker="o",facecolor="#faf9f6",edgecolor="#586f7c",linewidth=1.5,zorder=3)
        ax.scatter(after,y,s=45,marker="D",facecolor="#087c70",edgecolor="#087c70",zorder=4)
    ax.set_title(title,loc="left",fontsize=10,weight="bold",color="#172936")
    ax.set_xlim(*limits);ax.set_ylim(5.6,-.6);ax.set_yticks(range(6))
    ax.set_yticklabels([f"{r['split']} / {r['seed']}" for r in rows],fontsize=8,color="#405562")
    ax.tick_params(axis="both",length=0,labelsize=8,labelcolor="#405562",pad=4)
    for spine in ax.spines.values():spine.set_visible(False)
fig.text(.12,.045,"○  Original coordinate check",fontsize=9,color="#586f7c")
fig.text(.40,.045,"◆  Corrected row-wise check",fontsize=9,color="#087c70")
fig.text(.73,.045,"Row label: split / seed",fontsize=8,color="#566671")
fig.savefig(root / "docs/paired-comparison.png",facecolor=fig.get_facecolor(),dpi=150)
