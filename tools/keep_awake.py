"""Visit the hosted demos with a real browser so Streamlit Community Cloud keeps them awake.

Community Cloud puts an app to sleep after a period without visitors; the next
visitor then waits for a cold start. A plain HTTP request does not count as a
visit (the app only runs once a browser opens its websocket), so this opens
each URL in headless Chromium, presses the wake-up button if the app is
asleep, and waits until the app itself has rendered.

Run by .github/workflows/keep-awake.yml on a schedule. Locally:
    python tools/keep_awake.py [--channel msedge] [URL ...]
Exit code 1 if any app did not come up.
"""

from __future__ import annotations

import argparse
import sys

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout, sync_playwright

DEFAULT_URLS = (
    "https://spc-analyzer.streamlit.app/",
    "https://fiber-composite-calculator.streamlit.app/",
    "https://hoffmann-line-balancing.streamlit.app/",
)
WAKE_BUTTON = "Yes, get this app back up"
APP_SELECTOR = '[data-testid="stAppViewContainer"]'
COLD_START_MS = 240_000


def _app_rendered(page: Page) -> bool:
    return any(frame.locator(APP_SELECTOR).count() > 0 for frame in page.frames)


def visit(page: Page, url: str) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    state = "awake"
    waited = 0
    while waited < COLD_START_MS:
        if _app_rendered(page):
            page.wait_for_timeout(5_000)  # let the session register
            return state
        button = page.get_by_role("button", name=WAKE_BUTTON)
        if state == "awake" and button.count() > 0:
            button.first.click()
            state = "woken"
        page.wait_for_timeout(2_000)
        waited += 2_000
    raise PlaywrightTimeout(f"app did not render within {COLD_START_MS // 1000} s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("urls", nargs="*", default=list(DEFAULT_URLS))
    parser.add_argument("--channel", default=None, help="browser channel, e.g. msedge (default: bundled Chromium)")
    args = parser.parse_args()

    failed = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True)
        for url in args.urls:
            page = browser.new_page()
            try:
                print(f"{url}: {visit(page, url)}", flush=True)
            except Exception as exc:  # report every app, fail at the end
                failed += 1
                print(f"{url}: FAILED - {exc}", flush=True)
            finally:
                page.close()
        browser.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
