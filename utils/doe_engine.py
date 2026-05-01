"""DOE design matrix generation engine."""
import numpy as np
import pandas as pd

try:
    import pyDOE3 as _pydoe
    PYDOE_AVAILABLE = True
except ImportError:
    try:
        import pyDOE2 as _pydoe
        PYDOE_AVAILABLE = True
    except ImportError:
        PYDOE_AVAILABLE = False
        _pydoe = None

# Design type metadata
DESIGN_CATALOG = {
    "Full Factorial (2^k)": {
        "key": "full_factorial",
        "type": "Screening",
        "model": "Main effects + All interactions",
        "resolution": "Full",
        "min_factors": 2,
        "max_factors": 6,
        "description": "Tests all combinations of factor levels. Best for up to 5 factors.",
        "icon": "🔲",
    },
    "Fractional Factorial (2^k-p)": {
        "key": "fractional_factorial",
        "type": "Screening",
        "model": "Main effects + Some 2FI",
        "resolution": "III–V",
        "min_factors": 3,
        "max_factors": 15,
        "description": "Efficient screening for many factors. Some interactions are aliased.",
        "icon": "📐",
    },
    "Plackett-Burman": {
        "key": "plackett_burman",
        "type": "Screening",
        "model": "Main effects only",
        "resolution": "III",
        "min_factors": 2,
        "max_factors": 23,
        "description": "Maximum factor coverage with minimum runs. Best for preliminary screening.",
        "icon": "🎯",
    },
    "Central Composite (CCD)": {
        "key": "ccd",
        "type": "Response Surface",
        "model": "Full quadratic (linear + 2FI + quadratic)",
        "resolution": "V",
        "min_factors": 2,
        "max_factors": 8,
        "description": "Gold standard for response surface. Supports full quadratic model.",
        "icon": "⭕",
    },
    "Box-Behnken (BBD)": {
        "key": "box_behnken",
        "type": "Response Surface",
        "model": "Full quadratic",
        "resolution": "V",
        "min_factors": 3,
        "max_factors": 7,
        "description": "No corner points — ideal when extreme combinations are impractical.",
        "icon": "🔷",
    },
}

# Standard fractional factorial generators
_FF_GENERATORS = {
    3:  "a b ab",
    4:  "a b c abc",
    5:  "a b c d abcd",
    6:  "a b c d e abcde",
    7:  "a b c d e f abcdef",
    8:  "a b c d e f g abcdefg",
    9:  "a b c d e f g h abcdefgh",
    10: "a b c d e f g h i abcdefghi",
}

# Plackett-Burman base designs (rows are runs, cols are factors)
_PB_BASE = {
    4:  [[ 1, 1,-1], [ 1,-1, 1], [-1, 1, 1], [-1,-1,-1]],
    8:  [[ 1, 1, 1,-1, 1,-1,-1], [ 1, 1,-1, 1,-1,-1, 1],
         [ 1,-1, 1,-1,-1, 1, 1], [-1, 1,-1,-1, 1, 1, 1],
         [ 1,-1,-1, 1, 1, 1,-1], [-1,-1, 1, 1, 1,-1, 1],
         [-1, 1, 1, 1,-1, 1,-1], [-1,-1,-1,-1,-1,-1,-1]],
    12: [[ 1, 1,-1, 1, 1, 1,-1,-1,-1, 1,-1],
         [-1, 1, 1,-1, 1, 1, 1,-1,-1,-1, 1],
         [ 1,-1, 1, 1,-1, 1, 1, 1,-1,-1,-1],
         [-1, 1,-1, 1, 1,-1, 1, 1, 1,-1,-1],
         [-1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1],
         [-1,-1,-1, 1,-1, 1, 1,-1, 1, 1, 1],
         [ 1,-1,-1,-1, 1,-1, 1, 1,-1, 1, 1],
         [ 1, 1,-1,-1,-1, 1,-1, 1, 1,-1, 1],
         [ 1, 1, 1,-1,-1,-1, 1,-1, 1, 1,-1],
         [-1, 1, 1, 1,-1,-1,-1, 1,-1, 1, 1],
         [ 1,-1, 1, 1, 1,-1,-1,-1, 1,-1, 1],
         [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]],
}


def _pb_runs_needed(n_factors: int) -> int:
    n = 4
    while n - 1 < n_factors:
        n += 4
    return n


def _ccd_runs(n_factors: int) -> int:
    return 2**n_factors + 2 * n_factors + 5


def _bbd_runs(n_factors: int) -> int:
    bbd = {3: 15, 4: 27, 5: 46, 6: 54, 7: 62}
    return bbd.get(n_factors, 3 * n_factors * (n_factors - 1) + 3)


def get_run_count(design_key: str, n_factors: int) -> int:
    if design_key == "full_factorial":
        return 2 ** n_factors
    elif design_key == "fractional_factorial":
        return max(8, 2 ** (n_factors - n_factors // 3))
    elif design_key == "plackett_burman":
        return _pb_runs_needed(n_factors)
    elif design_key == "ccd":
        return _ccd_runs(n_factors)
    elif design_key == "box_behnken":
        return _bbd_runs(n_factors)
    return 0


def generate_design(design_key: str, n_factors: int, center_points: int = 3,
                    ccd_face: str = "ccf", randomize: bool = True,
                    random_seed: int = 42) -> np.ndarray:
    """Generate a coded (-1 to 1) design matrix."""
    if not PYDOE_AVAILABLE:
        raise ImportError("pyDOE3 is required. Run: pip install pyDOE3")

    rng = np.random.default_rng(random_seed)

    if design_key == "full_factorial":
        design = _pydoe.ff2n(n_factors)
        if center_points > 0:
            design = np.vstack([design, np.zeros((center_points, n_factors))])

    elif design_key == "fractional_factorial":
        if n_factors in _FF_GENERATORS:
            gen = _FF_GENERATORS[n_factors]
            design = _pydoe.fracfact(gen)
        else:
            design = _pydoe.ff2n(n_factors)
        if center_points > 0:
            design = np.vstack([design, np.zeros((center_points, n_factors))])

    elif design_key == "plackett_burman":
        n_runs = _pb_runs_needed(n_factors)
        design = _pydoe.pbdesign(n_runs)
        design = design[:, :n_factors]

    elif design_key == "ccd":
        design = _pydoe.ccdesign(n_factors, face=ccd_face)

    elif design_key == "box_behnken":
        if n_factors < 3:
            raise ValueError("Box-Behnken requires at least 3 factors.")
        design = _pydoe.bbdesign(n_factors, center=max(1, center_points))

    else:
        raise ValueError(f"Unknown design type: {design_key}")

    if randomize:
        idx = rng.permutation(len(design))
        design = design[idx]

    return design


def coded_to_natural(coded: np.ndarray, factor_ranges: list) -> np.ndarray:
    """Convert coded (-1..1) values to natural (actual) values."""
    natural = np.empty_like(coded, dtype=float)
    for i, (lo, hi) in enumerate(factor_ranges):
        center = (hi + lo) / 2.0
        half = (hi - lo) / 2.0
        natural[:, i] = center + coded[:, i] * half
    return natural


def natural_to_coded(natural: np.ndarray, factor_ranges: list) -> np.ndarray:
    """Convert natural values to coded (-1..1) values."""
    coded = np.empty_like(natural, dtype=float)
    for i, (lo, hi) in enumerate(factor_ranges):
        center = (hi + lo) / 2.0
        half = (hi - lo) / 2.0
        coded[:, i] = (natural[:, i] - center) / half
    return coded


def build_design_dataframe(design: np.ndarray, factor_names: list,
                           factor_ranges: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (coded_df, natural_df) from a design matrix."""
    coded_df = pd.DataFrame(design, columns=factor_names)
    coded_df.index = pd.RangeIndex(1, len(coded_df) + 1)
    coded_df.index.name = "Run"

    natural_vals = coded_to_natural(design, factor_ranges)
    natural_df = pd.DataFrame(natural_vals, columns=factor_names)
    natural_df.index = coded_df.index

    return coded_df, natural_df


def design_to_excel(natural_df: pd.DataFrame, response_names: list) -> bytes:
    """Return Excel bytes with design matrix + empty response columns."""
    import io
    df = natural_df.copy()
    for r in response_names:
        df[r] = np.nan

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Design Matrix", index=True)
        workbook = writer.book
        worksheet = writer.sheets["Design Matrix"]
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#1e40af", "font_color": "white"})
        for col_num, col_name in enumerate(["Run"] + list(df.columns)):
            worksheet.write(0, col_num, col_name, header_fmt)
        worksheet.set_column(0, len(df.columns), 14)
    return buf.getvalue()


def get_design_recommendations(n_factors: int, objective: str = "screening") -> list[str]:
    """Return recommended design names for given factors and objective."""
    if objective == "screening":
        if n_factors <= 5:
            return ["Full Factorial (2^k)", "Plackett-Burman"]
        else:
            return ["Fractional Factorial (2^k-p)", "Plackett-Burman"]
    else:  # optimization / RSM
        if n_factors == 2:
            return ["Central Composite (CCD)", "Full Factorial (2^k)"]
        elif n_factors <= 7:
            return ["Central Composite (CCD)", "Box-Behnken (BBD)"]
        else:
            return ["Central Composite (CCD)"]
