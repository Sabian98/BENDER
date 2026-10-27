"""palette.py: one color scheme for every BENDER JCTC figure, after figures4papers
(github.com/ChenLiu-1996/figures4papers), in the pastel style of its ImmunoStruct bar charts
(figure_ImmunoStruct/figures/bars_comparison_IEDB.png: blue #3775BA, tan #DAA87C, mauve #D9B9D4,
greens #DDF3DE #AADCA9 #8BCF8B, grays #CFCECE).

History (2026-10-21): first pass navy/maroon (too deep; figures/backup_2026-10-21_deep_f4p/), then a lighter
blue/red pass, then this pastel blue/tan scheme, chosen by the user from a side-by-side comparison. The blue/red
pass is still available with BENDER_PALETTE=light.

Roles (color always encodes the same thing):
  CALVADOS-2       blue   (C2 #3775BA lines/text/markers, C2_FILL #8FB3DD fills)
  MPIPI-Recharged  tan    (MR #B8834F lines/text/markers, MR_FILL #DAA87C fills)
  MPIPI-GG         mauve  (GG #C4A0BE)
  *_DARK           one step deeper, only for thin rules, medians and small text that need contrast
  differences      DIVERGING: tan where MPIPI-Recharged is stickier / more compact, blue where CALVADOS-2 is
  density (sequences per bin)  pastel teal ramp
  sequence-derived quantities  grays
  residue chemistry: Arg dark gray, aromatic gold, Lys green, acidic light gray

Checked with simulated protan/deutan/tritan vision (Machado 2009, OKLab dE x100):
  pastel  lines/markers C2/MR/GG pairwise dE >= 10.9 in every vision type (C2 vs MR >= 19.6);
          fills C2_FILL/MR_FILL >= 13.7 (normal 15.5)
  light   lines/markers >= 14.2; fills >= 12.5 (normal 16.5)
"""
import os

from matplotlib.colors import LinearSegmentedColormap

VARIANT = os.environ.get("BENDER_PALETTE", "pastel")   # pastel (paper) or light (blue/red alternative)

C2, C2_FILL, C2_LIGHT, C2_PALE, C2_DARK = "#3775BA", "#8FB3DD", "#C3D6EE", "#E3ECF7", "#2B62A3"
if VARIANT == "light":
    MR, MR_FILL, MR_LIGHT, MR_PALE, MR_DARK = "#B64342", "#DE8A84", "#F2C4C0", "#FAE6E4", "#9C3837"
    GG, GG_DARK = "#5FAFB7", "#3F8F98"
    DIV_WARM = "#E9A6A1"
else:
    MR, MR_FILL, MR_LIGHT, MR_PALE, MR_DARK = "#B8834F", "#DAA87C", "#EBCBAE", "#F6E7DA", "#966637"
    GG, GG_DARK = "#C4A0BE", "#9A7494"
    DIV_WARM = "#E6C3A0"                       # light step on the MPIPI-Recharged side of DIVERGING
INK, DARK, MUTED, RULE, PALE = "#272727", "#4D4D4D", "#767676", "#CFCECE", "#EDEDED"
SEQ, SEQ_DARK = "#CFCECE", "#767676"          # sequence-derived quantities
GOLD = "#FFD700"
ARG, ARO, LYS_C, ACID, OTHER_RES = DARK, "#F2C200", "#8BCF8B", "#CFCECE", "#EDEDED"
GREEN_DARK = "#4E9A55"                        # darkest step of the green ramp (lines on white)

# sequences per bin: pastel teal (no force field uses teal); gray-to-black density was rejected 2026-10-21
DENSITY = LinearSegmentedColormap.from_list("density", ["#EEF7F7", "#CFE8EA", "#9FD0D5", "#6BB2BA", "#3F8F98"])
DIVERGING = LinearSegmentedColormap.from_list(   # low (negative) = MPIPI-Recharged side, high = CALVADOS-2 side
    "diverging", [MR, DIV_WARM, "#F4F4F4", "#A9C5E6", C2])
C2_MAP = LinearSegmentedColormap.from_list("c2", ["#FFFFFF", C2_PALE, C2_LIGHT, C2_FILL, C2])
MR_MAP = LinearSegmentedColormap.from_list("mr", ["#FFFFFF", MR_PALE, MR_LIGHT, MR_FILL, MR])
LYS_MAP = LinearSegmentedColormap.from_list("lys", [GREEN_DARK, "#7DBF7F", "#AADCA9", "#DDF3DE"])
