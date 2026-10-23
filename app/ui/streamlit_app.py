"""Streamlit entry point: routes between the analysis pages.

Run with:  streamlit run app/ui/streamlit_app.py

* Process capability (SPC) -- ``capability_page.py``
* Capability, non-normal   -- ``nonnormal_page.py``
* Type-1 gauge (MSA)       -- ``type1_page.py``
* Gage R&R (MSA)           -- ``gage_rr_page.py``

Measurement system analysis comes first in practice (a capability study on an
unvalidated gauge measures the gauge as much as the process), but capability
stays the default page so existing links keep working.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="SPC / Process Capability Analyzer",
    page_icon="📐",
    layout="wide",
)

page = st.navigation(
    [
        st.Page(_HERE / "capability_page.py", title="Process capability (SPC)", icon="📐", default=True),
        st.Page(_HERE / "nonnormal_page.py", title="Capability, non-normal", icon="📈", url_path="non-normal"),
        st.Page(_HERE / "type1_page.py", title="Type-1 gauge (MSA)", icon="🎯", url_path="type-1"),
        st.Page(_HERE / "gage_rr_page.py", title="Gage R&R (MSA)", icon="📏", url_path="gage-rr"),
    ]
)
page.run()
