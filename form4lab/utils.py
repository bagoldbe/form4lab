import re


def to_python_float(v):
    """Cast numpy scalars to Python float for Postgres compatibility.

    psycopg2 cannot serialize np.float64 directly. This casts any
    numeric value to a native Python float, passing through None.
    """
    if v is None:
        return None
    return float(v)


# C-suite role detection — shared between portfolio_simulator and alpaca_service
_CSUITE_PHRASES = [
    "CEO", "CFO", "COO", "CTO",
    "CHIEF EXECUTIVE OFFICER", "CHIEF FINANCIAL OFFICER",
    "CHIEF OPERATING OFFICER", "CHIEF TECHNOLOGY OFFICER",
    "PRESIDENT", "CHAIRMAN",
]

_CSUITE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _CSUITE_PHRASES) + r")\b"
)


def is_csuite(role_title: str | None) -> bool:
    """Check if a role title indicates a C-suite position.

    C-suite: CEO, CFO, COO, CTO, President, Chairman.
    NOT C-suite: Vice President, EVP, SVP, VP, Director, 10% Owner.
    """
    if not role_title:
        return False
    title_upper = role_title.upper()
    if "VICE" in title_upper:
        return False
    return bool(_CSUITE_RE.search(title_upper))


_CEO_RE = re.compile(r"\b(?:CEO|CHIEF EXECUTIVE OFFICER)\b")
_CFO_RE = re.compile(r"\b(?:CFO|CHIEF FINANCIAL OFFICER)\b")


def is_ceo(role_title: str | None) -> bool:
    """Check if a role title indicates the Chief Executive Officer."""
    if not role_title:
        return False
    title_upper = role_title.upper()
    if "VICE" in title_upper:
        return False
    return bool(_CEO_RE.search(title_upper))


def is_cfo(role_title: str | None) -> bool:
    """Check if a role title indicates the Chief Financial Officer."""
    if not role_title:
        return False
    title_upper = role_title.upper()
    if "VICE" in title_upper:
        return False
    return bool(_CFO_RE.search(title_upper))
