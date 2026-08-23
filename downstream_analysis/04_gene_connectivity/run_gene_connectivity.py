#!/usr/bin/env python3
from pathlib import Path
import csv
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image

# Usage:
#   Run from any working directory:
#       python run_gene_connectivity.py
#
# Definition:
#   direct = unique Pathway Commons direct Gene-Gene neighbours
#   indirect = unique Gene-Gene neighbours connected through the same SNV
#   indirect-only = indirect minus direct
#   total = union(direct, indirect)
#
# A pair supported by both mechanisms is counted once in total and
# assigned to the direct segment of the stacked bar.

BASE = Path(__file__).resolve().parent
INPUT = BASE / "data" / "edges_snv_gene.csv"
OUTDIR = BASE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

PC_EDGES = [
    # metabolic-sequence (6)
    ("ENSG00000175164","ENSG00000174951","ABO","FUT1","metabolic_sequence"),
    ("ENSG00000175164","ENSG00000176920","ABO","FUT2","metabolic_sequence"),
    ("ENSG00000147003","ENSG00000198743","CLTRN","SLC5A3","metabolic_sequence"),
    ("ENSG00000164400","ENSG00000089127","CSF2","OAS1","metabolic_sequence"),
    ("ENSG00000160326","ENSG00000198743","SLC2A6","SLC5A3","metabolic_sequence"),
    ("ENSG00000198743","ENSG00000163817","SLC5A3","SLC6A20","metabolic_sequence"),

    # regulatory (5)
    ("ENSG00000121691","ENSG00000164400","CAT","CSF2","regulatory"),
    ("ENSG00000121691","ENSG00000186868","CAT","MAPT","regulatory"),
    ("ENSG00000241186","ENSG00000204531","CRIPTO","POU5F1","regulatory"),
    ("ENSG00000164400","ENSG00000185499","CSF2","MUC1","regulatory"),
    ("ENSG00000105550","ENSG00000137310","FGF21","TCF19","regulatory"),

    # physical / complex (23)
    ("ENSG00000130234","ENSG00000121691","ACE2","CAT","physical_complex"),
    ("ENSG00000130234","ENSG00000204539","ACE2","CDSN","physical_complex"),
    ("ENSG00000130234","ENSG00000012223","ACE2","LTF","physical_complex"),
    ("ENSG00000130234","ENSG00000133661","ACE2","SFTPD","physical_complex"),
    ("ENSG00000130234","ENSG00000163817","ACE2","SLC6A20","physical_complex"),
    ("ENSG00000121691","ENSG00000105559","CAT","PLEKHA4","physical_complex"),
    ("ENSG00000204536","ENSG00000204539","CCHCR1","CDSN","physical_complex"),
    ("ENSG00000163823","ENSG00000204539","CCR1","CDSN","physical_complex"),
    ("ENSG00000163823","ENSG00000012223","CCR1","LTF","physical_complex"),
    ("ENSG00000121807","ENSG00000160791","CCR2","CCR5","physical_complex"),
    ("ENSG00000121807","ENSG00000180739","CCR2","S1PR5","physical_complex"),
    ("ENSG00000182504","ENSG00000105559","CEP97","PLEKHA4","physical_complex"),
    ("ENSG00000147003","ENSG00000163817","CLTRN","SLC6A20","physical_complex"),
    ("ENSG00000164400","ENSG00000159110","CSF2","IFNAR2","physical_complex"),
    ("ENSG00000164400","ENSG00000243646","CSF2","IL10RB","physical_complex"),
    ("ENSG00000172215","ENSG00000105559","CXCR6","PLEKHA4","physical_complex"),
    ("ENSG00000186803","ENSG00000159110","IFNA10","IFNAR2","physical_complex"),
    ("ENSG00000186803","ENSG00000243646","IFNA10","IL10RB","physical_complex"),
    ("ENSG00000228083","ENSG00000159110","IFNA14","IFNAR2","physical_complex"),
    ("ENSG00000228083","ENSG00000243646","IFNA14","IL10RB","physical_complex"),
    ("ENSG00000159110","ENSG00000243646","IFNAR2","IL10RB","physical_complex"),
    ("ENSG00000012223","ENSG00000186868","LTF","MAPT","physical_complex"),
    ("ENSG00000163818","ENSG00000186868","LZTFL1","MAPT","physical_complex"),
]

GENE_SYMBOL = {}
for a_id,b_id,a_name,b_name,_ in PC_EDGES:
    GENE_SYMBOL[a_id] = a_name
    GENE_SYMBOL[b_id] = b_name

# Extra symbols needed by the top-ranking shared-SNV-only genes.
GENE_SYMBOL.update({
    "ENSG00000204540":"PSORS1C1",
    "ENSG00000204538":"PSORS1C2",
    "ENSG00000163825":"RTP3",
})

with INPUT.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

pairs = {(r["snv_id"], r["gene_id"]) for r in rows}
snvs = {s for s,_ in pairs}
genes = {g for _,g in pairs}

assert len(pairs) == 248
assert len(snvs) == 187
assert len(genes) == 79
assert len(PC_EDGES) == 34

cat_count = defaultdict(int)
for *_, cls in PC_EDGES:
    cat_count[cls] += 1
assert cat_count["physical_complex"] == 23
assert cat_count["regulatory"] == 5
assert cat_count["metabolic_sequence"] == 6

by_snv = defaultdict(set)
for snv,gene in pairs:
    by_snv[snv].add(gene)

indirect = defaultdict(set)
for gs in by_snv.values():
    for a,b in itertools.combinations(sorted(gs),2):
        indirect[a].add(b)
        indirect[b].add(a)

direct = defaultdict(set)
for a,b,_,_,_ in PC_EDGES:
    direct[a].add(b)
    direct[b].add(a)

ranking = []
for gene_id in genes:
    d = direct.get(gene_id,set())
    i = indirect.get(gene_id,set())
    io = i - d
    overlap = d & i
    total = d | i
    ranking.append({
        "gene_id":gene_id,
        "gene":GENE_SYMBOL.get(gene_id,gene_id),
        "direct_neighbors":len(d),
        "indirect_only_neighbors":len(io),
        "dual_supported_neighbors":len(overlap),
        "total_unique_neighbors":len(total),
    })

ranking.sort(key=lambda r:(
    -r["total_unique_neighbors"],
    -r["direct_neighbors"],
    -r["indirect_only_neighbors"],
    r["gene_id"]
))
for n,r in enumerate(ranking,1):
    r["rank"] = n

top20 = ranking[:20]
fields = [
    "rank","gene_id","gene","direct_neighbors",
    "indirect_only_neighbors","dual_supported_neighbors",
    "total_unique_neighbors"
]

for path,data in [
    (OUTDIR/"coding_gene_integrated_connectivity_full_79.csv", ranking),
    (OUTDIR/"coding_gene_integrated_connectivity_top20.csv", top20),
]:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)

max_total = max(r["total_unique_neighbors"] for r in top20)
xmax = max_total + 1.35

def panel(data,title,path,legend=False):
    data = list(reversed(data))
    genes = [r["gene"] for r in data]
    d = [r["direct_neighbors"] for r in data]
    i = [r["indirect_only_neighbors"] for r in data]
    totals = [r["total_unique_neighbors"] for r in data]

    fig = plt.figure(figsize=(7.0,6.5))
    ax = fig.add_axes([0.22,0.13,0.73,0.78])
    ax.barh(genes,d,color="#3C78B5",label="Direct (Pathway Commons)")
    ax.barh(genes,i,left=d,color="#E39A45",label="Indirect only (shared SNV)")

    for y,total in enumerate(totals):
        ax.text(total+0.10,y,str(total),va="center",ha="left",fontsize=10)

    ax.set_xlim(0,xmax)
    ax.set_xlabel("Unique connected coding-gene neighbours")
    ax.set_title(title,fontsize=13,pad=10)
    ax.grid(axis="x",alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(frameon=False,loc="lower right",fontsize=9)

    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close(fig)

p1 = OUTDIR/"panel_A_ranks_1_10.png"
p2 = OUTDIR/"panel_B_ranks_11_20.png"
panel(top20[:10],"A   Ranks 1–10",p1,True)
panel(top20[10:20],"B   Ranks 11–20",p2,False)

im1 = Image.open(p1).convert("RGB")
im2 = Image.open(p2).convert("RGB")
h = max(im1.height,im2.height)

if im1.height != h:
    im1 = im1.resize((round(im1.width*h/im1.height),h))
if im2.height != h:
    im2 = im2.resize((round(im2.width*h/im2.height),h))

gap = 24
canvas = Image.new("RGB",(im1.width+gap+im2.width,h),"white")
canvas.paste(im1,(0,0))
canvas.paste(im2,(im1.width+gap,0))

canvas.save(OUTDIR/"coding_gene_integrated_connectivity_top20_two_panel.png",dpi=(300,300))
canvas.save(OUTDIR/"coding_gene_integrated_connectivity_top20_two_panel.pdf","PDF",resolution=300)

print("Top 20:")
for r in top20:
    print(
        r["rank"], r["gene"],
        "direct=",r["direct_neighbors"],
        "indirect-only=",r["indirect_only_neighbors"],
        "total=",r["total_unique_neighbors"]
    )
