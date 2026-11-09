from html import escape
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "pages"
LESSONS_DIR = ROOT / "Lessons"
REPOSITORY = "jfemiani/cse620b-public"


def display_name(value: str) -> str:
    name = value.removeprefix("demo-").replace("-", " ").replace("_", " ").title()
    for before, after in (
        ("Deepglobe", "DeepGlobe"),
        ("Kmeans", "k-means"),
        ("Knn", "k-NN"),
        ("Lulc", "LULC"),
        ("Mlp", "MLP"),
        ("Osm", "OSM"),
        ("Pgd", "PGD"),
        ("Pytorch", "PyTorch"),
    ):
        name = name.replace(before, after)
    return name


def relative_url(path: Path) -> str:
    return quote(path.relative_to(ROOT).as_posix(), safe="/-._~")


def github_url(path: Path) -> str:
    return f"https://github.com/{REPOSITORY}/blob/main/{relative_url(path)}"


def page_links() -> str:
    pages = sorted(PAGES_DIR.glob("*.html"), key=lambda path: path.name.casefold())
    if not pages:
        raise SystemExit(f"No course pages found in {PAGES_DIR}")

    return "\n".join(
        f'        <li><a href="{relative_url(page)}">{escape(page.stem)}</a></li>'
        for page in pages
    )


def resource_link(path: Path) -> str | None:
    filename = path.name
    if filename.endswith(".slides.html"):
        label = f"Browser slides: {display_name(filename.removesuffix('.slides.html'))}"
        link = f'<a href="{relative_url(path)}">Open slides</a>'
    elif path.suffix == ".ipynb":
        label = f"Notebook: {display_name(path.stem)}"
        colab_url = (
            "https://colab.research.google.com/github/"
            f"{REPOSITORY}/blob/main/{relative_url(path)}"
        )
        link = (
            f'<a href="{colab_url}">Open in Colab</a>'
            f' <span aria-hidden="true">·</span> '
            f'<a href="{github_url(path)}">View notebook source</a>'
        )
    elif path.suffix == ".pdf":
        label = f"PDF: {display_name(path.stem)}"
        link = f'<a href="{relative_url(path)}">Open PDF</a>'
    elif path.suffix == ".html":
        label = f"Rendered demo: {display_name(path.stem)}"
        link = f'<a href="{relative_url(path)}">Open demo</a>'
    elif path.suffix == ".py":
        label = f"Python source: {display_name(path.stem)}"
        link = f'<a href="{github_url(path)}">View source</a>'
    else:
        return None

    return f"            <li><strong>{escape(label)}</strong> — {link}</li>"


def lesson_sections() -> str:
    sections = []
    for lesson in sorted(LESSONS_DIR.iterdir(), key=lambda path: path.name.casefold()):
        if not lesson.is_dir():
            continue
        resources = [
            item
            for path in sorted(lesson.rglob("*"), key=lambda path: path.as_posix().casefold())
            if path.is_file()
            if (item := resource_link(path)) is not None
        ]
        if not resources:
            continue
        title = lesson.name.split("-", 1)[-1].replace("-", " ")
        sections.append(
            f"""        <article class="module">
          <h3>{escape(title)}</h3>
          <ul class="resource-list">
{chr(10).join(resources)}
          </ul>
        </article>"""
        )
    if not sections:
        raise SystemExit(f"No lesson resources found in {LESSONS_DIR}")
    return "\n".join(sections)


def main() -> None:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Course pages, lecture slides, and Colab notebooks for CSE 620B at Miami University.">
  <title>CSE 620B | Computer Vision for Remote Sensing</title>
  <style>
    :root {
      color-scheme: light;
      --red: #941728;
      --ink: #242424;
      --muted: #555;
      --line: #dedede;
      --panel: #f7f5f3;
      --link: #075b85;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: #fff;
      font: 17px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    a { color: var(--link); text-underline-offset: 3px; overflow-wrap: anywhere; }
    a:focus-visible { outline: 3px solid #e6a700; outline-offset: 3px; }
    .skip-link {
      position: absolute;
      top: -5rem;
      left: 1rem;
      z-index: 1;
      padding: .5rem .75rem;
      background: #fff;
    }
    .skip-link:focus { top: 1rem; }
    header {
      padding: 3rem max(1.25rem, calc((100% - 1100px) / 2));
      color: #fff;
      background: var(--red);
    }
    header p { max-width: 720px; margin: .75rem 0 0; }
    .eyebrow { margin: 0; font-size: .9rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    h1 { max-width: 850px; margin: .5rem 0 0; font-size: clamp(2rem, 5vw, 3.25rem); line-height: 1.15; }
    main { max-width: 1100px; margin: 0 auto; padding: 1.5rem 1.25rem 4rem; }
    h2 { margin: 2.5rem 0 1rem; color: var(--red); font-size: 1.6rem; line-height: 1.3; }
    h3 { margin: 0; font-size: 1.25rem; line-height: 1.35; }
    .notice, .quiz-note {
      padding: 1rem 1.25rem;
      border-left: 5px solid var(--red);
      background: var(--panel);
    }
    .notice { margin: 0 0 2rem; }
    .notice p, .quiz-note p { margin: 0; }
    .page-list, .resource-list { margin: .8rem 0 0; padding-left: 1.25rem; }
    .page-list li, .resource-list li { margin: .45rem 0; }
    .module-list { display: grid; gap: 1rem; }
    .module {
      padding: 1.25rem;
      border: 1px solid var(--line);
      border-radius: .5rem;
      background: #fff;
    }
    .resource-list { color: var(--muted); }
    .resource-list a { font-weight: 600; }
    footer {
      padding: 1.5rem max(1.25rem, calc((100% - 1100px) / 2));
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: .95rem;
    }
    @media (min-width: 760px) {
      .module-list { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <a class="skip-link" href="#main">Skip to course materials</a>
  <header>
    <p class="eyebrow">Miami University</p>
    <h1>CSE 620B: Computer Vision for Remote Sensing</h1>
    <p>Lecture pages, slides, and hands-on notebooks for exploring computer vision in remote sensing.</p>
  </header>
  <main id="main">
    <aside class="notice" aria-label="Course site information">
      <p><strong>Read-only course resource site.</strong> This site does not collect assignment submissions or quiz attempts. Use Canvas when coursework must be submitted.</p>
    </aside>
    <section aria-labelledby="course-pages-heading">
      <h2 id="course-pages-heading">Course pages</h2>
      <p>Student-facing lesson pages, including the course content maintained for Canvas.</p>
      <ul class="page-list">
{page_links}
      </ul>
    </section>
    <section aria-labelledby="materials-heading">
      <h2 id="materials-heading">Slides, notebooks, and demos</h2>
      <p>Open notebooks in Google Colab or browse the slides and rendered demonstrations by lesson.</p>
      <div class="module-list">
{lesson_sections}
      </div>
    </section>
    <section aria-labelledby="practice-heading">
      <h2 id="practice-heading">Practice quizzes</h2>
      <div class="quiz-note">
        <p>Optional self-check practice quizzes are planned. They will be separate from Canvas assessments and will not submit answers or grades.</p>
      </div>
    </section>
  </main>
  <footer>
    <p>CSE 620B course materials · <a href="https://github.com/jfemiani/cse620b-public">Public course repository</a></p>
  </footer>
</body>
</html>
"""
    (ROOT / "index.html").write_text(
        template.replace("{page_links}", page_links()).replace(
            "{lesson_sections}", lesson_sections()
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
