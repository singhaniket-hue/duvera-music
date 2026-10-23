"""A visible, dedicated browser profile for manual Google sign-in on YOUR computer."""
import argparse
from pathlib import Path
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=Path("chrome_google_flow_profile"))
    parser.add_argument("--url", default="https://labs.google/fx/tools/music-fx")
    args = parser.parse_args()
    url = urlparse(args.url)
    if url.scheme != "https" or url.hostname != "labs.google":
        parser.error("Use an https://labs.google URL")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        parser.error("Install .[browser] and run python -m playwright install chromium")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(str(args.profile.resolve()), headless=False)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(args.url)
        print("Sign in manually in the dedicated window. Close the window when finished.")
        print("No credentials are read by this script. A stored profile is NOT a verified login.")
        try:
            page.wait_for_event("close", timeout=0)
        finally:
            context.close()


if __name__ == "__main__":
    main()
