import json
from html import escape
from pathlib import Path
from urllib.parse import quote, urlencode


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "jfemiani/cse620b-public"
CATEGORIES = ("Lecture notes", "Slides", "Demos", "Labs", "Practice quizzes")


def url(path: Path) -> str:
    return quote(path.relative_to(ROOT).as_posix(), safe="/-._~")


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"Course resource does not exist: {path.relative_to(ROOT)}")
    return path


def action(label: str, href: str, *, embed: bool = False, download: bool = False) -> str:
    attributes = ""
    if embed:
        attributes = ' data-embed="true"'
    elif download:
        attributes = " download"
    else:
        attributes = ' target="_blank" rel="noopener"'
    return f'<a href="{escape(href, quote=True)}"{attributes}>{escape(label)}</a>'


def card(title: str, actions: list[str]) -> str:
    return (
        f'<article class="resource"><h3>{escape(title)}</h3>'
        f'<div class="actions">{"".join(actions)}</div></article>'
    )


def slide_cards(topic: dict, prefix: str) -> str:
    cards = []
    lesson = ROOT / "Lessons" / topic["lesson"]
    for slide in topic["slides"]:
        pdf = require_file(lesson / slide["file"])
        viewer = prefix + "presentation.html?" + urlencode(
            {"pdf": url(pdf), "title": slide["title"]}
        )
        cards.append(card(slide["title"], [
            action("View slideshow", viewer, embed=True),
            action("Download PDF", prefix + url(pdf), download=True),
        ]))
    for slide in topic.get("canvas_slides", []):
        cards.append(card(slide["title"], [
            action("Open on Canvas (course access required)", slide["url"]),
        ]))
    for demo in topic["demos"]:
        rendered = lesson / (demo["stem"] + ".slides.html")
        if rendered.is_file():
            cards.append(card(demo["title"], [
                action("View slideshow", prefix + url(rendered), embed=True),
                action("Open separately", prefix + url(rendered)),
            ]))
    return "".join(cards) or '<p class="empty">Lecture slides have not been published for this topic.</p>'


def demo_cards(topic: dict, prefix: str) -> str:
    cards = []
    lesson = ROOT / "Lessons" / topic["lesson"]
    for demo in topic["demos"]:
        notebook = require_file(lesson / (demo["stem"] + ".ipynb"))
        colab = f"https://colab.research.google.com/github/{REPOSITORY}/blob/main/{url(notebook)}"
        actions = [
            action("Open in Google Colab", colab),
            action("Download notebook", prefix + url(notebook), download=True),
        ]
        for suffix, label in ((".html", "View rendered output"), (".slides.html", "View as slides")):
            rendered = lesson / (demo["stem"] + suffix)
            if rendered.is_file():
                actions.append(action(label, prefix + url(rendered), embed=True))
        source = lesson / (demo["stem"] + ".py")
        if source.is_file():
            actions.append(action("Download Python", prefix + url(source), download=True))
        cards.append(card(demo["title"], actions))
    return "".join(cards) or '<p class="empty">Demos have not been published for this topic.</p>'


def page(topic: dict, topics: list[dict], *, home: bool = False) -> str:
    prefix = "" if home else "../"
    nav = "".join(
        f'<a href="{prefix}pages/{item["slug"]}.html"'
        + (' aria-current="page"' if item["slug"] == topic["slug"] else "")
        + f'>{escape(item["title"])}</a>'
        for item in topics
    )
    tabs = "".join(
        f'<button type="button" role="tab" id="tab-{index}" aria-controls="panel-{index}"'
        f' aria-selected="{"true" if index == 0 else "false"}"'
        f' tabindex="{0 if index == 0 else -1}">{label}</button>'
        for index, label in enumerate(CATEGORIES)
    )
    notes = '<p class="empty">Lecture notes have not been published for this topic.</p>'
    if topic.get("notes"):
        note = require_file(ROOT / topic["notes"])
        notes = card("Lecture notes", [action("Read lecture notes", prefix + url(note), embed=True)])
    content = (
        notes,
        slide_cards(topic, prefix),
        demo_cards(topic, prefix),
        '<p class="empty">Labs have not been published for this topic. This site does not accept submissions.</p>',
        '<p class="empty">Practice quizzes are to be developed. They will be separate from Canvas assessments.</p>',
    )
    panels = "".join(
        f'<section role="tabpanel" id="panel-{index}" aria-labelledby="tab-{index}"'
        f' tabindex="0"{"" if index == 0 else " hidden"}>{body}</section>'
        for index, body in enumerate(content)
    )
    first_notes = ""
    if topic.get("notes"):
        first_notes = (
            f'<iframe id="material-frame" title="{escape(topic["title"])}: lecture notes"'
            f' src="{prefix}{url(ROOT / topic["notes"])}" allowfullscreen></iframe>'
        )
    else:
        first_notes = '<iframe id="material-frame" title="Selected course material" allowfullscreen hidden></iframe>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{escape(topic['summary'], quote=True)}">
  <title>{escape(topic['title'])} | CSE 620B</title>
  <link rel="stylesheet" href="{prefix}assets/course.css">
  <script src="{prefix}assets/course.js" defer></script>
</head>
<body>
  <a class="skip-link" href="#main">Skip to topic</a>
  <header class="course-header">
    <a href="{prefix}index.html"><strong>CSE 620B</strong><span>Computer Vision for Remote Sensing</span></a>
    <p>Miami University</p>
  </header>
  <div class="course-layout">
    <aside class="sidebar">
      <h2>Topics</h2>
      <nav aria-label="Course topics">{nav}</nav>
      <p class="site-note">Course materials only.<br>No submissions or grades.</p>
    </aside>
    <main id="main">
      <h1>{escape(topic['title'])}</h1>
      <p class="summary">{escape(topic['summary'])}</p>
      <div class="category-tabs" role="tablist" aria-label="Topic materials">{tabs}</div>
      <div class="resource-panels">{panels}</div>
      <section id="viewer" aria-label="Material viewer"{"" if topic.get("notes") else " hidden"}>
        <div class="viewer-toolbar">
          <h2 id="viewer-title">Lecture notes</h2>
          <div class="actions">
            <a id="open-material" href="{prefix + topic['notes'] if topic.get('notes') else '#'}" target="_blank" rel="noopener">Open separately</a>
            <button type="button" id="fullscreen-material">Full screen</button>
          </div>
        </div>
        <p id="viewer-error" role="alert" hidden></p>
        {first_notes}
      </section>
      <noscript><p>Enable JavaScript to switch material categories and use the embedded viewer.</p></noscript>
    </main>
  </div>
</body>
</html>
"""


def redirect(target: str, title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url={target}">
  <link rel="canonical" href="{target}">
  <title>{escape(title)} | CSE 620B</title>
</head>
<body><p>This topic has moved to <a href="{target}">{escape(title)}</a>.</p></body>
</html>
"""


def main() -> None:
    topics = json.loads((ROOT / "course.json").read_text(encoding="utf-8"))
    if not topics or len({topic["slug"] for topic in topics}) != len(topics):
        raise ValueError("Course topics must be nonempty and have unique slugs")
    pages = ROOT / "pages"
    pages.mkdir(exist_ok=True)
    for topic in topics:
        (pages / (topic["slug"] + ".html")).write_text(page(topic, topics), encoding="utf-8")
        if topic.get("legacy"):
            (pages / topic["legacy"]).write_text(
                redirect(topic["slug"] + ".html", topic["title"]), encoding="utf-8"
            )
    (ROOT / "index.html").write_text(page(topics[0], topics, home=True), encoding="utf-8")


if __name__ == "__main__":
    main()
