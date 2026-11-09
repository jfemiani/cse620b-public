import importlib.util
import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build", ROOT / "scripts/build_course_index.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)
STAGE_SPEC = importlib.util.spec_from_file_location("stage", ROOT / "scripts/stage_site.py")
stage = importlib.util.module_from_spec(STAGE_SPEC)
STAGE_SPEC.loader.exec_module(stage)


class Document(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.links = []
        self.ids = set()
        self.headings = []
        self.in_heading = False
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for attribute in ("href", "src"):
            if attrs.get(attribute):
                self.links.append(attrs[attribute])
        self.in_heading = tag in ("title", "h1", "h2", "h3")

    def handle_endtag(self, tag):
        if tag in ("title", "h1", "h2", "h3"):
            self.in_heading = False

    def handle_data(self, data):
        if self.in_heading:
            self.headings.append(data)


class CourseSiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build.main()
        cls.topics = json.loads((ROOT / "course.json").read_text(encoding="utf-8"))

    def test_links_files_and_fragments_exist(self):
        pages = [ROOT / "index.html", ROOT / "presentation.html"]
        pages += list((ROOT / "pages").glob("*.html"))
        pages += list((ROOT / "notes").glob("*.html"))
        for page in pages:
            for link in Document(page).links:
                parsed = urlsplit(link)
                if parsed.scheme or parsed.netloc:
                    continue
                target = (page.parent / unquote(parsed.path)).resolve() if parsed.path else page
                with self.subTest(page=page.relative_to(ROOT), link=link):
                    self.assertTrue(target.is_relative_to(ROOT))
                    self.assertTrue(target.is_file(), f"Missing target: {target}")
                    if parsed.fragment and target.suffix == ".html":
                        self.assertIn(parsed.fragment, Document(target).ids)
                    if target.name == "presentation.html" and parsed.query:
                        pdf = parse_qs(parsed.query)["pdf"][0]
                        self.assertTrue((ROOT / unquote(pdf)).is_file())

    def test_topic_names_and_order(self):
        for topic in self.topics:
            page = ROOT / "pages" / (topic["slug"] + ".html")
            document = Document(page)
            self.assertEqual(document.headings[0], topic["title"] + " | CSE 620B")
            self.assertFalse(any(re.search(r"\b(?:Week|Chapter)\s+\d", name) for name in document.headings))
            navigation = [
                link for link in document.links
                if link.startswith("../pages/") and "#" not in link
            ]
            self.assertEqual(
                navigation,
                [f'../pages/{item["slug"]}.html' for item in self.topics],
            )
            for index in range(5):
                self.assertIn(f"tab-{index}", document.ids)
                self.assertIn(f"panel-{index}", document.ids)

    def test_all_notebooks_have_download_and_colab_links(self):
        notebooks = {path.resolve() for path in (ROOT / "Lessons").rglob("*.ipynb")}
        linked = set()
        for topic in self.topics:
            page = ROOT / "pages" / (topic["slug"] + ".html")
            links = Document(page).links
            for link in links:
                path = urlsplit(link).path
                if not link.startswith("../Lessons/") or not path.endswith(".ipynb"):
                    continue
                notebook = (page.parent / unquote(path)).resolve()
                linked.add(notebook)
                colab = f"https://colab.research.google.com/github/{build.REPOSITORY}/blob/main/{build.url(notebook)}"
                self.assertIn(colab, links)
        self.assertEqual(notebooks, linked)

    def test_legacy_pages_redirect_to_named_topics(self):
        for topic in self.topics:
            if topic.get("legacy"):
                page = (ROOT / "pages" / topic["legacy"]).read_text(encoding="utf-8")
                self.assertIn(f'url={topic["slug"]}.html', page)

    def test_missing_resource_fails_build(self):
        with self.assertRaisesRegex(ValueError, "Course resource does not exist"):
            build.require_file(ROOT / "Lessons" / "missing.pdf")

    def test_build_is_reproducible(self):
        before = (ROOT / "index.html").read_bytes()
        build.main()
        self.assertEqual(before, (ROOT / "index.html").read_bytes())

    def test_notebook_dependencies_are_localized(self):
        source = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe"></script>
<link href="https://unpkg.com/reveal.js@4.0.2/dist/reveal.css">
<script src="https://unpkg.com/reveal.js@4.0.2/dist/reveal.js"></script>
<script src="https://unpkg.com/reveal.js@4.0.2/plugin/notes/notes.js"></script>
"""
        localized = stage.localize_notebook_assets(source, "../../assets/vendor/")
        self.assertNotIn("https://", localized)
        self.assertIn("../../assets/vendor/reveal/reveal.js", localized)
        self.assertIn("../../assets/vendor/reveal/plugin/notes/notes.js", localized)
        self.assertIn("MathJax.js?config=TeX-AMS_CHTML-full,Safe", localized)

    def test_staged_notebook_local_links_exist(self):
        self.assertTrue((ROOT / "_site").is_dir(), "Run npm run build before npm test")
        for page in (ROOT / "_site" / "Lessons").rglob("*.html"):
            for link in Document(page).links:
                parsed = urlsplit(link)
                if "assets/vendor" not in parsed.path or parsed.scheme:
                    continue
                with self.subTest(page=page.relative_to(ROOT), asset=link):
                    self.assertTrue((page.parent / unquote(parsed.path)).is_file())


if __name__ == "__main__":
    unittest.main()
