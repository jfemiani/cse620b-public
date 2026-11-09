import re
import shutil
from pathlib import Path
from os.path import relpath


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "_site"


def localize_notebook_assets(html: str, prefix: str) -> str:
    html = re.sub(
        r"https://cdnjs\.cloudflare\.com/ajax/libs/require\.js/[\d.]+/require\.min\.js",
        prefix + "require.js",
        html,
    )
    html = re.sub(
        r"https://cdnjs\.cloudflare\.com/ajax/libs/mathjax/[\d.]+/latest\.js",
        prefix + "mathjax/MathJax.js",
        html,
    )
    html = html.replace("https://unpkg.com/reveal.js@4.0.2/dist/", prefix + "reveal/")
    return html.replace("https://unpkg.com/reveal.js@4.0.2/plugin/", prefix + "reveal/plugin/")


def main() -> None:
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    DESTINATION.mkdir(exist_ok=True)
    for filename in ("index.html", "presentation.html"):
        shutil.copy2(ROOT / filename, DESTINATION / filename)
    for directory in ("pages", "notes", "assets", "Lessons"):
        shutil.copytree(ROOT / directory, DESTINATION / directory, dirs_exist_ok=True)
    for page in (DESTINATION / "Lessons").rglob("*.html"):
        prefix = Path(relpath(DESTINATION / "assets" / "vendor", page.parent)).as_posix() + "/"
        original = page.read_text(encoding="utf-8")
        localized = localize_notebook_assets(original, prefix)
        if original != localized:
            page.write_text(localized, encoding="utf-8")


if __name__ == "__main__":
    main()
