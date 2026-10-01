"""
Evolution Detective: Find the Missing Ancestor
------------------------------------------------
An interactive bioinformatics app that takes 3-6 species' DNA sequences and:
  1. Aligns them (pairwise-based "star" multiple alignment)
  2. Detects mutation / variable positions
  3. Calculates pairwise genetic similarity
  4. Builds a phylogenetic tree (UPGMA)
  5. Reconstructs the ancestral sequence at every internal node (Fitch's
     parsimony algorithm) and flags positions that are genuinely uncertain
  6. Lets you "replay" evolution along a slider from the common ancestor
     down to any modern species, watching mutations accumulate

Dependencies: streamlit, biopython, plotly, pandas
    pip install streamlit biopython plotly pandas
Run with:
    streamlit run evolution_detective.py

Only Biopython is used for the bioinformatics itself (Bio.Align.PairwiseAligner
for alignment, Bio.Phylo.TreeConstruction for the tree) -- no external tools
like Clustal/MUSCLE/MAFFT are required.
"""

from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment, PairwiseAligner
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

st.set_page_config(page_title="Evolution Detective", page_icon="🧬", layout="wide")

# ----------------------------------------------------------------------
# Example data (illustrative sequences, NOT real database accessions --
# built so Human/Chimp are closest, Gibbon most distant, matching the
# textbook primate phylogeny, and Gibbon carries one deletion so the
# alignment step has a real indel to resolve).
# ----------------------------------------------------------------------
EXAMPLE_SEQUENCES = {
    "Human":     "ATGGCCAACCTCCTACTCCTCATCGTACCCATTCTAATCG",
    "Chimp":     "ATGGCCGACCTCCTACTCCTCATCGTACCCATTTTAATCG",
    "Gorilla":   "ATCGCCAACCTGCTACTCCTCATCGTATCCGTTCTAATCA",
    "Orangutan": "ACGGACAATCCCCTATTCCTCATGGTGCCCATACTAACCG",
    "Gibbon":    "GTGACCTACATACTGCTACTAATCATACACAGTCTGATG",
}


# ========================================================================
# 1. PAIRWISE ALIGNMENT + STAR MULTIPLE SEQUENCE ALIGNMENT
# ========================================================================

def get_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    return aligner


def _merge_into_profile(profile_seqs, ref_old, ref_new, other_new):
    """Merge a new pairwise alignment (ref_new/other_new) into an existing
    profile whose reference row is ref_old, producing one unified frame.
    This is the standard 'star alignment' merge step: walk both gapped
    copies of the reference together, and whenever one has a gap the
    other doesn't, that column is unique to one side."""
    i, j = 0, 0
    len_old, len_new = len(ref_old), len(ref_new)
    new_cols = [[] for _ in profile_seqs]
    other_col = []

    while i < len_old or j < len_new:
        old_is_gap = i < len_old and ref_old[i] == "-"
        new_is_gap = j < len_new and ref_new[j] == "-"

        if old_is_gap:
            for k, seq in enumerate(profile_seqs):
                new_cols[k].append(seq[i])
            other_col.append("-")
            i += 1
        elif new_is_gap:
            for k in range(len(profile_seqs)):
                new_cols[k].append("-")
            other_col.append(other_new[j])
            j += 1
        else:
            for k, seq in enumerate(profile_seqs):
                new_cols[k].append(seq[i])
            other_col.append(other_new[j])
            i += 1
            j += 1

    result = ["".join(c) for c in new_cols]
    result.append("".join(other_col))
    return result


def build_star_msa(seq_dict: dict) -> dict:
    """Build a multiple sequence alignment using the first sequence as the
    reference ('star' / center-star alignment): every other sequence is
    pairwise-aligned to the reference, and the pairwise alignments are
    merged into one shared coordinate frame. Good for closely related
    sequences, and needs nothing but Bio.Align.PairwiseAligner."""
    names = list(seq_dict.keys())
    ref_seq = seq_dict[names[0]]
    aligner = get_aligner()

    first_other = names[1]
    best = aligner.align(ref_seq, seq_dict[first_other])[0]
    profile_names = [names[0], first_other]
    profile_seqs = [str(best[0]), str(best[1])]

    for name in names[2:]:
        best = aligner.align(ref_seq, seq_dict[name])[0]
        ref_new, other_new = str(best[0]), str(best[1])
        profile_seqs = _merge_into_profile(profile_seqs, profile_seqs[0], ref_new, other_new)
        profile_names.append(name)

    return dict(zip(profile_names, profile_seqs))


# ========================================================================
# 2. DISTANCE / SIMILARITY + PHYLOGENETIC TREE
# ========================================================================

def compute_similarity_matrix(aligned: dict) -> pd.DataFrame:
    """Percent identity between every pair, ignoring columns where either
    sequence has a gap."""
    names = list(aligned.keys())
    df = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            sa, sb = aligned[a], aligned[b]
            valid = [(x, y) for x, y in zip(sa, sb) if x != "-" and y != "-"]
            if not valid:
                df.loc[a, b] = 0.0
                continue
            matches = sum(1 for x, y in valid if x == y)
            df.loc[a, b] = round(100 * matches / len(valid), 1)
    return df


def build_tree(aligned: dict):
    """Build a UPGMA phylogenetic tree using only Biopython's built-in,
    pure-Python distance calculator and tree constructor (no external
    tools like RAxML/FastTree needed)."""
    records = [SeqRecord(Seq(seq), id=name) for name, seq in aligned.items()]
    alignment = MultipleSeqAlignment(records)
    calculator = DistanceCalculator("identity")
    dm = calculator.get_distance(alignment)
    constructor = DistanceTreeConstructor()
    tree = constructor.upgma(dm)
    return tree


def get_tree_plot_data(tree):
    """Assign (x, y) plotting coordinates to every clade: y evenly spaces
    the leaves and averages internal nodes; x is cumulative branch length
    from the root, exactly like a standard dendrogram/phylogram layout."""
    terminals = tree.get_terminals()
    y_pos = {leaf: i for i, leaf in enumerate(terminals)}

    def assign_y(clade):
        if clade.is_terminal():
            return y_pos[clade]
        ys = [assign_y(c) for c in clade.clades]
        y_pos[clade] = sum(ys) / len(ys)
        return y_pos[clade]

    assign_y(tree.root)

    x_pos = {tree.root: 0.0}

    def assign_x(clade):
        for child in clade.clades:
            bl = child.branch_length if child.branch_length is not None else 0.0
            x_pos[child] = x_pos[clade] + bl
            assign_x(child)

    assign_x(tree.root)
    return y_pos, x_pos


def build_tree_figure(tree) -> go.Figure:
    y_pos, x_pos = get_tree_plot_data(tree)
    edge_x, edge_y = [], []

    def add_edges(clade):
        for child in clade.clades:
            edge_x.extend([x_pos[clade], x_pos[child], None])
            edge_y.extend([y_pos[child], y_pos[child], None])
            add_edges(child)

        if len(clade.clades) > 1:
            ys = [y_pos[c] for c in clade.clades]
            edge_x.extend([x_pos[clade], x_pos[clade], None])
            edge_y.extend([min(ys), max(ys), None])

    add_edges(tree.root)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#4a5568", width=2),
            hoverinfo="skip"
        )
    )

    terminals = tree.get_terminals()

    fig.add_trace(
        go.Scatter(
            x=[x_pos[leaf] for leaf in terminals],
            y=[y_pos[leaf] for leaf in terminals],
            mode="markers+text",
            text=[leaf.name for leaf in terminals],
            textposition="middle right",
            marker=dict(size=11, color="#e53e3e"),
            hoverinfo="text"
        )
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Genetic distance (substitutions per site)",
        yaxis=dict(showticklabels=False, zeroline=False),
        margin=dict(l=20, r=140, t=20, b=40),
        height=120 + 60 * len(terminals),
        plot_bgcolor="white",
    )

    return fig

# ========================================================================
# 3. ANCESTRAL RECONSTRUCTION (Fitch's small-parsimony algorithm)
# ========================================================================

def fitch_ancestral_states(tree, aligned: dict, length: int):
    """For every node (leaf and internal) and every alignment column,
    compute the Fitch state set. A set of size 1 means that position is
    unambiguous under parsimony; size > 1 means the ancestral base at
    that position genuinely cannot be pinned down from this tree alone."""
    node_states = {clade: [None] * length for clade in tree.find_clades()}

    def postorder(clade):
        if clade.is_terminal():
            seq = aligned[clade.name]
            for pos in range(length):
                node_states[clade][pos] = {seq[pos]}
        else:
            for child in clade.clades:
                postorder(child)
            for pos in range(length):
                child_sets = [node_states[c][pos] for c in clade.clades]
                inter = child_sets[0]
                for s in child_sets[1:]:
                    inter = inter & s
                if inter:
                    node_states[clade][pos] = inter
                else:
                    union = set()
                    for s in child_sets:
                        union |= s
                    node_states[clade][pos] = union

    postorder(tree.root)
    return node_states


def format_ancestral_sequence(state_sets):
    """Render a list of Fitch state-sets as a sequence string, with
    genuinely ambiguous positions shown as e.g. [A/G]."""
    parts, uncertain = [], []
    for i, s in enumerate(state_sets):
        if len(s) == 1:
            parts.append(next(iter(s)))
        else:
            parts.append("[" + "/".join(sorted(s)) + "]")
            uncertain.append(i)
    return "".join(parts), uncertain


def get_path_to_leaf(tree, leaf_name):
    """Root-to-leaf path (inclusive of both ends) as a list of clades."""
    path = []

    def recurse(clade, trail):
        if clade.is_terminal() and clade.name == leaf_name:
            path.extend(trail + [clade])
            return True
        return any(recurse(child, trail + [clade]) for child in clade.clades)

    recurse(tree.root, [])
    return path


def highlight_diff_html(seq_a: str, seq_b: str) -> str:
    """seq_b rendered as HTML with positions that differ from seq_a
    highlighted -- used to show mutations accumulating step by step."""
    out = []
    for a, b in zip(seq_a, seq_b):
        if a != b:
            out.append(f"<span style='background-color:#ffd43b;font-weight:700'>{b}</span>")
        else:
            out.append(b)
    return ("<pre style='font-family:monospace;font-size:16px;letter-spacing:1px;"
            "white-space:pre-wrap;word-break:break-all'>" + "".join(out) + "</pre>")


def render_alignment_html(aligned: dict) -> str:
    """Monospace block of every aligned sequence, with columns that vary
    across species highlighted."""
    names = list(aligned.keys())
    length = len(next(iter(aligned.values())))
    pad = max(len(n) for n in names)
    lines = []
    for name in names:
        seq = aligned[name]
        line = f"{name.ljust(pad)}  "
        for pos in range(length):
            col = {aligned[n][pos] for n in names}
            ch = seq[pos]
            if len(col) > 1:
                line += f"<mark style='background-color:#ffd43b'>{ch}</mark>"
            else:
                line += ch
        lines.append(line)
    return ("<pre style='font-family:monospace;font-size:14px;line-height:1.7;"
            "overflow-x:auto'>" + "<br>".join(lines) + "</pre>")


# ========================================================================
# STREAMLIT UI
# ========================================================================

st.title("🧬 Evolution Detective")
st.caption("Find the Missing Ancestor — sequence alignment, mutation analysis, "
           "phylogeny and ancestral reconstruction in one workflow.")

with st.expander("📥 Input sequences", expanded=True):
    mode = st.radio("Source", ["Use example primate sequences", "Paste FASTA", "Upload FASTA file"],
                     horizontal=True)

    sequences = {}
    if mode == "Use example primate sequences":
        sequences = dict(EXAMPLE_SEQUENCES)
        st.code("\n".join(f">{n}\n{s}" for n, s in sequences.items()), language="text")
    elif mode == "Paste FASTA":
        fasta_text = st.text_area(
            "Paste 3-6 sequences in FASTA format",
            height=200,
            placeholder=">Human\nATGCATGC...\n>Chimp\nATGCATGG...",
        )
        if fasta_text.strip():
            try:
                records = list(SeqIO.parse(StringIO(fasta_text), "fasta"))
                sequences = {r.id: str(r.seq).upper().replace(" ", "") for r in records}
            except Exception as e:
                st.error(f"Couldn't parse that FASTA text: {e}")
    else:
        uploaded = st.file_uploader("FASTA file", type=["fasta", "fa", "txt"])
        if uploaded:
            try:
                records = list(SeqIO.parse(StringIO(uploaded.read().decode("utf-8")), "fasta"))
                sequences = {r.id: str(r.seq).upper().replace(" ", "") for r in records}
            except Exception as e:
                st.error(f"Couldn't parse that file: {e}")

    st.caption("The first sequence entered is used as the alignment reference.")

    run = st.button("🔬 Run analysis", type="primary", use_container_width=True)

if run:
    if len(sequences) < 3:
        st.warning("Add at least 3 species for a meaningful tree and ancestral reconstruction.")
    else:
        st.session_state["sequences"] = sequences

if "sequences" in st.session_state:
    seqs = st.session_state["sequences"]

    with st.spinner("Aligning sequences and building the tree..."):
        aligned = build_star_msa(seqs)
        length = len(next(iter(aligned.values())))
        tree = build_tree(aligned)
        node_states = fitch_ancestral_states(tree, aligned, length)

    variable_positions = sum(
        1 for pos in range(length) if len({aligned[n][pos] for n in aligned}) > 1
    )
    root_seq, root_uncertain = format_ancestral_sequence(
        [node_states[tree.root][p] for p in range(length)]
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Species", len(aligned))
    c2.metric("Alignment length", length)
    c3.metric("Variable positions", variable_positions)
    c4.metric("Uncertain ancestral sites", len(root_uncertain))

    tab_align, tab_mut, tab_dist, tab_tree, tab_anc = st.tabs(
        ["Alignment", "Mutations", "Similarity", "Phylogenetic Tree", "Ancestor & Replay"]
    )

    with tab_align:
        st.markdown(render_alignment_html(aligned), unsafe_allow_html=True)
        st.caption("Highlighted columns differ across at least one species.")

    with tab_mut:
        rows = []
        for pos in range(length):
            col = {n: aligned[n][pos] for n in aligned}
            if len(set(col.values())) > 1:
                rows.append({"Position": pos + 1, **col})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Position"), use_container_width=True)
        else:
            st.info("No variable positions found — sequences are identical.")

    with tab_dist:
        sim = compute_similarity_matrix(aligned)
        fig = go.Figure(data=go.Heatmap(
            z=sim.values, x=sim.columns, y=sim.index,
            colorscale="RdYlGn", zmin=0, zmax=100,
            text=sim.values, texttemplate="%{text}%", hoverinfo="skip",
        ))
        fig.update_layout(height=120 + 60 * len(sim), margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Percent identity between each pair of aligned sequences.")

    with tab_tree:
        st.plotly_chart(build_tree_figure(tree), use_container_width=True)
        st.caption("UPGMA tree built from the identity distance matrix.")

    with tab_anc:
        st.subheader("Estimated common ancestor")
        st.code(root_seq, language="text")
        st.caption(
            f"{len(root_uncertain)} of {length} positions are genuinely ambiguous under "
            "parsimony (shown as [X/Y]) — the available sequences alone can't resolve them."
        )

        st.divider()
        st.subheader("🎞️ Evolution Replay")
        target = st.selectbox("Replay evolution toward:", list(aligned.keys()))
        path = get_path_to_leaf(tree, target)
        step = st.slider("Step", 0, len(path) - 1, 0,
                          help="0 = common ancestor, last = modern sequence")

        clade = path[step]
        if clade.is_terminal():
            current_seq = aligned[clade.name]
            label = f"Modern {clade.name}"
        else:
            current_seq, _ = format_ancestral_sequence([node_states[clade][p] for p in range(length)])
            label = "Common ancestor" if step == 0 else f"Ancestral node — step {step}"

        st.markdown(f"**{label}**")
        if step == 0:
            st.markdown(
                f"<pre style='font-family:monospace;font-size:16px'>{current_seq}</pre>",
                unsafe_allow_html=True,
            )
        else:
            # every node before the final leaf in a root-to-leaf path is an
            # ancestral (internal) node, so the previous step always has a
            # Fitch-reconstructed sequence to diff against
            prev_clade = path[step - 1]
            prev_seq, _ = format_ancestral_sequence([node_states[prev_clade][p] for p in range(length)])
            st.markdown(highlight_diff_html(prev_seq, current_seq), unsafe_allow_html=True)
            st.caption("Highlighted bases changed since the previous step.")
else:
    st.info("Choose or paste sequences above, then click **Run analysis**.")
