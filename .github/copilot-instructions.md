# CSE 620B Public (Student-Facing Repo) — Copilot Instructions

## What this repo is

The **public, student-facing** materials for **CSE 620B: Computer Vision for
Remote Sensing** at Miami University (instructor: John Femiani). Everything
here is safe for students (and the public internet) to see: built slides,
demo notebooks, handouts, and HTML mirrors of the Canvas course pages.

**Nothing here should ever contain quiz/exam answer keys, grading rubrics,
correct-answer flags, or instructor-only notes.** If a file looks like it
might contain any of those, it belongs in the sibling private repo instead —
do not add it here even temporarily.

Two related repos exist side by side on disk (`../cse620b` and
`../cse620b-public`) and on GitHub under the `jfemiani` account, with
**independent git histories** (this is not a fork or a subtree of the other):
- **`cse620b`** (private) — slide sources (Marp Markdown), the quiz question
  bank with answers, in-progress demos, and course planning docs. This repo
  does not read that repo directly; content flows over via the private
  repo's `Makefile` copying files by path, or by direct copy-paste review.
- **`cse620b-public`** (this repo) — the reviewed, rendered subset meant for
  students: built PDFs, cleaned notebooks, and (see below) HTML mirrors of
  Canvas modules.

## Current structure

- Slide PDFs/PPTX and demo notebooks currently sit flat in the repo root
  (e.g. `clouds.pdf`, `IRS-Chapter-18-Slides.pdf`, `demo-lulc.ipynb`,
  `demo-lulc-pytorch.py`). `Makefile` documents how each file is pulled from
  the private repo (`../cse620b/Slides/src/<Chapter>-Slides/...`) — check it
  before assuming a file's origin.
- `README.md` — the student-facing syllabus/schedule; keep dates and topics
  in sync with the private repo's `SCHEDULE.md`, but do not copy grading
  policy or instructor-only notes into it.

## Canvas modules: HTML mirrors (new convention)

This repo should hold a **rendered HTML mirror of each live Canvas module**,
in a `pages/` folder at the repo root, one file per Canvas page:

```
pages/N. Lecture Title.html
```

- The number and title in the filename mirror the live Canvas item order and
  title, the same convention `cse534-course-demos` uses for its per-module
  `pages/` folders — keep them in sync when reordering or renaming lessons.
- These are **working copies of what's live on Canvas**, authored here and
  pushed to Canvas, not the reverse: treat this repo as the source of truth
  for page content, and Canvas as the publish target.
- Follow the same page-authoring conventions established for CSE 534
  (documented in the `cse534-page-template` skill): Miami-red section
  headers, numbered MathML equations with plain-English `aria-label`s, and
  code embedded via emgithub iframes pointing at **this repo's** GitHub
  blob URLs (`https://github.com/jfemiani/cse620b-public/blob/main/...`),
  never at the private repo — Canvas students must never be pointed at a
  private-repo URL they can't access.
- Images referenced from a page must use absolute
  `https://raw.githubusercontent.com/jfemiani/cse620b-public/main/...` URLs
  (Canvas cannot resolve relative paths), and must be committed/pushed
  before a page referencing them is uploaded.
- `CANVAS_COURSE_ID` for this course is not yet recorded anywhere in either
  repo — get it from the Canvas course URL
  (`https://miamioh.instructure.com/courses/<id>`) and set it in this repo's
  `.env` (gitignored) before using the upload/download scripts below. Do
  not assume the CSE 534 course ID (243761) that ships as the script's
  default.

## Skills and agents to use

These are user-level skills/agents (available in any repo; some are named
after CSE 534 but are course-agnostic in what they actually do):

- **`cse534-page-template`** skill — HTML/CSS templates for Canvas pages
  and the `upload_to_canvas.py` / `download_canvas_pages.py` scripts.
  ```bash
  # Download all live Canvas pages for comparison against pages/ here:
  python3 ~/.copilot/skills/cse534-page-template/download_canvas_pages.py remote_pages/

  # Upload a page (and optionally a slide PDF) once it's ready:
  python3 ~/.copilot/skills/cse534-page-template/upload_to_canvas.py \
      "pages/N. Lecture Title.html" \
      "N. Lecture Title" \
      "N. Lecture Title Slides.pdf"
  ```
  Both scripts read `CANVAS_ACCESS_TOKEN`, `CANVAS_BASE_URL`, and
  `CANVAS_COURSE_ID` from the environment — set this course's own values in
  `.env` here, do not reuse CSE 534's.
- **`canvas-page-editor`** agent — revising/creating Canvas pages, checking
  links/accessibility, embedding code demos. Its examples reference CSE 534
  paths (`mathematical_foundations/pages/`); adapt to this repo's flat
  `pages/` folder and `cse620b-public` GitHub URLs instead.
- **`educational-reviewer`** agent — reviews pages/notebooks for
  accessibility to new learners (jargon, unexplained assumptions,
  pedagogical clarity, code-demo focus). Run this on any new or
  substantially revised page before uploading to Canvas.
- **`avoid-ai-writing`** skill — audits and cleans written content for
  AI-writing tells (bold-lead templates, em dashes, list-disguised-as-prose
  captions) before finalizing any page or handout.
- **`mathml-notation`** skill — writing/reviewing MathML equations
  (stretchy fences, bracket consistency, a pre-flight checklist) for any
  page needing formulas.
- **`beamer-slide-template`** skill — reference for slide-pedagogy
  constraints (no deferred-to-whiteboard content, incremental reveals) even
  though this course's decks are Marp-based rather than Beamer.

## Environment

- Canvas credentials (`CANVAS_ACCESS_TOKEN`, `CANVAS_BASE_URL`,
  `CANVAS_COURSE_ID`) belong in this repo's own `.env` (gitignored, never
  committed) since this is the repo that owns Canvas-facing page content.
- No conda/Python environment is documented yet for this repo. If demos here
  need one, record the setup in `README.md` once established.

## General workflow rules

- Never commit anything containing correct-answer flags, quiz feedback
  text, grading rubrics, or other instructor-only content, even in a draft
  or WIP state — that content lives exclusively in the private `cse620b`
  repo.
- This repo and the private repo have independent git histories on separate
  GitHub remotes — commits here never sync automatically with the private
  repo, and vice versa; content moves over deliberately (via the private
  repo's `Makefile` or manual, reviewed copy).
- When adding a new student-facing file that originated in the private repo,
  check it against the private repo's `.github/copilot-instructions.md` for
  what counts as instructor-only before copying it here.
- Keep `pages/` filenames' numbering and titles in sync with the live Canvas
  module order; treat drift between this repo and Canvas the same way
  `cse534-course-demos` treats `remote_pages/` — re-download and diff before
  assuming which version is current.
