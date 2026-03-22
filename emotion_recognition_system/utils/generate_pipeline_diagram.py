"""Generate pipeline diagrams for the report methodology section.

Run from the repo root:
    python emotion_recognition_system/utils/generate_pipeline_diagram.py

Outputs:
    emotion_recognition_system/results/pipeline_diagram.png          (experimental)
    emotion_recognition_system/results/deployment_diagram.png        (inference / live demo)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Shared colours ────────────────────────────────────────────────────────────
C_DATA   = "#2C3E50"
C_PROC   = "#1A5276"
C_FEAT   = "#196F3D"
C_LABEL  = "#7D6608"
C_TRAIN  = "#922B21"
C_DEPLOY = "#0E6251"   # deployment / live demo steps
C_APP    = "#4A235A"   # application output (DDA)
C_ARROW  = "#95A5A6"
BG       = "#F4F6F7"
WHITE    = "#FEFEFE"


# ── Shared helpers ────────────────────────────────────────────────────────────
def box(ax, cy, title, subtitle=None, color=C_PROC, w=0.66, h=0.072):
    cx = 0.5
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.014",
        linewidth=1.3, edgecolor=color, facecolor=color, zorder=3,
    ))
    ty = cy + 0.014 if subtitle else cy
    ax.text(cx, ty, title, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=WHITE, zorder=4)
    if subtitle:
        ax.text(cx, cy - 0.020, subtitle, ha="center", va="center",
                fontsize=7.4, color=WHITE, alpha=0.90, zorder=4)


def arrow(ax, y_from, y_to):
    ax.annotate("", xy=(0.5, y_to), xytext=(0.5, y_from),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.6, mutation_scale=13), zorder=2)


def _make_canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.02, 1.02)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    return fig, ax


# ── Diagram 1: Experimental pipeline ─────────────────────────────────────────
def draw_experimental(out="emotion_recognition_system/results/pipeline_diagram.png"):
    H  = 0.072
    HF = 0.090   # feature extraction (two-line subtitle)
    HE = 0.140   # experiments block
    GAP = 0.050

    Y_out   = 0.086
    Y_exp   = Y_out   + H/2  + GAP + HE/2
    Y_cv    = Y_exp   + HE/2 + GAP + H/2
    Y_label = Y_cv    + H/2  + GAP + H/2
    Y_feat  = Y_label + H/2  + GAP + HF/2
    Y_sync  = Y_feat  + HF/2 + GAP + H/2
    Y_raw   = Y_sync  + H/2  + GAP + H/2
    TITLE_Y = Y_raw   + H/2  + 0.038

    fig, ax = _make_canvas(8, 11)

    # Title
    ax.text(0.5, TITLE_Y, "Experimental Pipeline",
            ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=C_DATA)

    # Boxes + arrows
    box(ax, Y_raw,  "Raw Recordings",
        "14 files   |   7 subjects x 2 conditions   |   EDA + PPG @ 64 Hz",
        color=C_DATA)
    arrow(ax, Y_raw - H/2, Y_sync + H/2)

    box(ax, Y_sync, "Sensor-Video Synchronisation",
        "alignment of labels using timestamps",
        color=C_PROC)
    arrow(ax, Y_sync - H/2, Y_feat + HF/2)

    box(ax, Y_feat, "Feature Extraction",
        "EDA: 8 features (tonic/phasic, SCR)   |   PPG: 13 HRV features (BPM, IBI, RMSSD...)\n"
        "30 s sliding windows, 15 s stride @ 64 Hz   ->   121 windows",
        color=C_FEAT, h=HF)
    arrow(ax, Y_feat - HF/2, Y_label + H/2)

    box(ax, Y_label, "Annotation Mapping",
        "77 labelled moments -> 88 windows   |   excitement / frustration / neutral",
        color=C_LABEL)
    arrow(ax, Y_label - H/2, Y_cv + H/2)

    box(ax, Y_cv, "Leave-One-Subject-Out Cross-Validation   (7 folds)",
        "train on 6 subjects, test on 1   |   train-fold-only median imputation",
        color=C_TRAIN)
    arrow(ax, Y_cv - H/2, Y_exp + HE/2)

    # Experiments block (custom — two-column list inside)
    ax.add_patch(FancyBboxPatch(
        (0.5 - 0.66/2, Y_exp - HE/2), 0.66, HE,
        boxstyle="round,pad=0.01,rounding_size=0.014",
        linewidth=1.3, edgecolor=C_TRAIN, facecolor=C_TRAIN, zorder=3,
    ))
    ax.text(0.5, Y_exp + HE/2 - 0.022,
            "14 Experiments   (Random Forest + SVM x 7 configurations)",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=WHITE, zorder=4)
    configs_left  = ["No fusion — EDA-only  (n=88)",
                     "No fusion — PPG-only  (n=39)",
                     "Early Fusion          (n=88)"]
    configs_right = ["Intermediate Fusion        (n=88)",
                     "Intermediate Fusion Comp.  (n=39)",
                     "Late Fusion                (n=88)",
                     "Late Fusion Comp.          (n=39)"]
    base_y = Y_exp + HE/2 - 0.052
    for i, txt in enumerate(configs_left):
        ax.text(0.20, base_y - i * 0.026, "• " + txt,
                ha="left", va="center", fontsize=7.2, color=WHITE, zorder=4)
    for i, txt in enumerate(configs_right):
        ax.text(0.50, base_y - i * 0.026, "• " + txt,
                ha="left", va="center", fontsize=7.2, color=WHITE, zorder=4)
    arrow(ax, Y_exp - HE/2, Y_out + H/2)

    box(ax, Y_out, "Results",
        "confusion matrices   |   feature importance   |   per-subject accuracy   |   metrics",
        color=C_DATA)

    # Legend
    legend = [
        mpatches.Patch(facecolor=C_DATA,  label="Data"),
        mpatches.Patch(facecolor=C_PROC,  label="Pre-processing"),
        mpatches.Patch(facecolor=C_FEAT,  label="Feature extraction"),
        mpatches.Patch(facecolor=C_LABEL, label="Labelling"),
        mpatches.Patch(facecolor=C_TRAIN, label="Training / evaluation"),
    ]
    ax.legend(handles=legend, loc="lower center", fontsize=7.5,
              framealpha=0.9, ncol=5, bbox_to_anchor=(0.5, -0.005),
              edgecolor="#BDC3C7")

    plt.tight_layout(pad=0.4)
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved to {out}")


# ── Diagram 2: Deployment / inference pipeline ────────────────────────────────
def draw_deployment(out="emotion_recognition_system/results/deployment_diagram.png"):
    H   = 0.072
    GAP = 0.060   # slightly wider gap — fewer boxes, more breathing room

    Y_dda    = 0.100
    Y_pred   = Y_dda   + H/2 + GAP + H/2
    Y_infer  = Y_pred  + H/2 + GAP + H/2
    Y_model  = Y_infer + H/2 + GAP + H/2
    Y_sensor = Y_model + H/2 + GAP + H/2
    TITLE_Y  = Y_sensor + H/2 + 0.055

    fig, ax = _make_canvas(8, 7)

    ax.text(0.5, TITLE_Y, "Deployment Pipeline",
            ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=C_DATA)

    box(ax, Y_sensor, "Live EDA Sensor",
        "PhysioKit wrist sensor   |   continuous EDA stream @ 64 Hz",
        color=C_DATA)
    arrow(ax, Y_sensor - H/2, Y_model + H/2)

    box(ax, Y_model, "EmotionPredictor",
        "trained model loaded from .pkl   |   internal EDA feature extraction",
        color=C_DEPLOY)
    arrow(ax, Y_model - H/2, Y_infer + H/2)

    box(ax, Y_infer, "predict_stream()",
        "30 s sliding windows, 15 s stride   |   feature extraction per window",
        color=C_FEAT)
    arrow(ax, Y_infer - H/2, Y_pred + H/2)

    box(ax, Y_pred, "Emotion Classification",
        "excitement  /  frustration  /  neutral   |   confidence score per window",
        color=C_TRAIN)
    arrow(ax, Y_pred - H/2, Y_dda + H/2)

    box(ax, Y_dda, "Dynamic Difficulty Adjustment",
        "excitement -> maintain / increase difficulty   |   frustration -> reduce difficulty",
        color=C_APP)

    # Legend
    legend = [
        mpatches.Patch(facecolor=C_DATA,   label="Hardware / data"),
        mpatches.Patch(facecolor=C_DEPLOY, label="Inference module"),
        mpatches.Patch(facecolor=C_FEAT,   label="Feature extraction"),
        mpatches.Patch(facecolor=C_TRAIN,  label="Classification"),
        mpatches.Patch(facecolor=C_APP,    label="Application"),
    ]
    ax.legend(handles=legend, loc="lower center", fontsize=7.5,
              framealpha=0.9, ncol=5, bbox_to_anchor=(0.5, -0.005),
              edgecolor="#BDC3C7")

    plt.tight_layout(pad=0.4)
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved to {out}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    draw_experimental()
    draw_deployment()