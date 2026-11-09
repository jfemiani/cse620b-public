

# Public repo layout mirrors the private repo's Lessons/NN-Module-Name/ numbering,
# so each target's destination path lines up with its module folder here.
# Only files with a known, reproducible private-repo source get a build rule;
# see the "no rebuild rule" note below for the rest.

FILES=Lessons/01-Introduction-to-GIS-and-Python/slides/Lecture-1-GIS-Overview-and-Geospatial-Data-Types.pdf
FILES+=Lessons/01-Introduction-to-GIS-and-Python/slides/Lecture-2-Geospatial-Data-Types-and-Python-for-GIS.pdf
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc.ipynb
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-pytorch.py
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-knn.ipynb
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-kmeans.ipynb
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-naive-bayes.ipynb
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-decision-tree.ipynb
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-random-forest.ipynb
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-gradient-boosting.ipynb
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-mlp.ipynb
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-knn.slides.html
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-kmeans.slides.html
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-naive-bayes.slides.html
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-decision-tree.slides.html
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-random-forest.slides.html
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-gradient-boosting.slides.html
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-mlp.slides.html
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-deepglobe.ipynb
FILES+=Lessons/03-Optical-Radiation-and-the-EM-Spectrum/slides/IRS-Chapter-2-Electromagnetic-Radiation-beamer.pdf
FILES+=Lessons/04-Remote-Sensing-Platforms-and-Cameras/slides/IRS-Chapter-3-Remote-Sensing-Platforms-beamer.pdf
FILES+=Lessons/04-Remote-Sensing-Platforms-and-Cameras/slides/IRS-Chapter-4-Digital-Mapping-Cameras-beamer.pdf
FILES+=Lessons/01-Introduction-to-GIS-and-Python/demo-raster-data-pgd-ch3.ipynb
FILES+=Lessons/01-Introduction-to-GIS-and-Python/demo-vector-data-osm.ipynb
FILES+=Lessons/01-Introduction-to-GIS-and-Python/demo-raster-data-pgd-ch3.html
FILES+=Lessons/01-Introduction-to-GIS-and-Python/demo-vector-data-osm.html
FILES+=Lessons/09-Change-Detection-and-Accuracy-Assessment/demo-lombardia.py
FILES+=Lessons/09-Change-Detection-and-Accuracy-Assessment/demo-lombardia.ipynb
FILES+=Lessons/09-Change-Detection-and-Accuracy-Assessment/demo-lombardia.slides.html
FILES+=Lessons/09-Change-Detection-and-Accuracy-Assessment/pdf/irs-chapter-15-slides.pdf
FILES+=Lessons/10-Hyperspectral-Remote-Sensing/demo-hyperspectral-mixtures.py
FILES+=Lessons/10-Hyperspectral-Remote-Sensing/demo-hyperspectral-mixtures.ipynb
FILES+=Lessons/10-Hyperspectral-Remote-Sensing/demo-hyperspectral-mixtures.slides.html
FILES+=Lessons/10-Hyperspectral-Remote-Sensing/slides/IRS-Chapter-14-Hyperspectral-Remote-Sensing.pdf

all: ${FILES}

Lessons/01-Introduction-to-GIS-and-Python/slides/Lecture-1-GIS-Overview-and-Geospatial-Data-Types.pdf: ../cse620b/Lessons/01-Introduction-to-GIS-and-Python/slides/Lecture-1-GIS-Overview-and-Geospatial-Data-Types.pdf
	mkdir -p Lessons/01-Introduction-to-GIS-and-Python/slides
	cp $< $@

Lessons/01-Introduction-to-GIS-and-Python/slides/Lecture-2-Geospatial-Data-Types-and-Python-for-GIS.pdf: ../cse620b/Lessons/01-Introduction-to-GIS-and-Python/slides/Lecture-2-Geospatial-Data-Types-and-Python-for-GIS.pdf
	mkdir -p Lessons/01-Introduction-to-GIS-and-Python/slides
	cp $< $@

Lessons/06-Image-Classification-and-Land-Cover/demo-lulc.ipynb: ../cse620b/Lessons/IRS-Chapter-6-Slides/demo-lulc.ipynb
	mkdir -p Lessons/06-Image-Classification-and-Land-Cover
	cp $< $@

Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-%.ipynb: Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-%.py
	jupytext --to ipynb $<
	python -c 'import json; from pathlib import Path; path = Path("$@"); notebook = json.loads(path.read_text()); [cell["metadata"].__setitem__("language", "python" if cell["cell_type"] == "code" else "markdown") for cell in notebook["cells"]]; path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")'

Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-%.slides.html: Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-%.ipynb
	jupyter nbconvert --to slides --output-dir "$(dir $@)" --output "$(basename $(basename $(notdir $@)))" $<

Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-pytorch.py: ../cse620b/Lessons/IRS-Chapter-6-Slides/demo-lulc-pytorch.py
	mkdir -p Lessons/06-Image-Classification-and-Land-Cover
	cp $< $@

Lessons/06-Image-Classification-and-Land-Cover/demo-deepglobe.ipynb: ../cse620b/Lessons/IRS-Chapter-6-Slides/demo-deepglobe.ipynb
	mkdir -p Lessons/06-Image-Classification-and-Land-Cover
	cp $< $@

Lessons/03-Optical-Radiation-and-the-EM-Spectrum/slides/IRS-Chapter-2-Electromagnetic-Radiation-beamer.pdf: ../cse620b/Lessons/03-Optical-Radiation-and-the-EM-Spectrum/beamer/IRS-Chapter-2-Electromagnetic-Radiation-beamer.pdf
	mkdir -p Lessons/03-Optical-Radiation-and-the-EM-Spectrum/slides
	cp $< $@

Lessons/04-Remote-Sensing-Platforms-and-Cameras/slides/IRS-Chapter-3-Remote-Sensing-Platforms-beamer.pdf: ../cse620b/Lessons/04-Remote-Sensing-Platforms-and-Cameras/platforms/beamer/IRS-Chapter-3-Remote-Sensing-Platforms-beamer.pdf
	mkdir -p Lessons/04-Remote-Sensing-Platforms-and-Cameras/slides
	cp $< $@

Lessons/04-Remote-Sensing-Platforms-and-Cameras/slides/IRS-Chapter-4-Digital-Mapping-Cameras-beamer.pdf: ../cse620b/Lessons/04-Remote-Sensing-Platforms-and-Cameras/cameras/beamer/IRS-Chapter-4-Digital-Mapping-Cameras-beamer.pdf
	mkdir -p Lessons/04-Remote-Sensing-Platforms-and-Cameras/slides
	cp $< $@

Lessons/01-Introduction-to-GIS-and-Python/demo-raster-data-pgd-ch3.ipynb: ../cse620b/Demos/demo-raster-data-pgd-ch3.ipynb
	mkdir -p Lessons/01-Introduction-to-GIS-and-Python
	cp $< $@

Lessons/01-Introduction-to-GIS-and-Python/demo-vector-data-osm.ipynb: ../cse620b/Demos/demo-vector-data-osm.ipynb
	mkdir -p Lessons/01-Introduction-to-GIS-and-Python
	cp $< $@

# Rendered HTML for embedding via iframe on Canvas pages (nbconvert
# --embed-images produces a single self-contained file, no external assets).
Lessons/01-Introduction-to-GIS-and-Python/demo-raster-data-pgd-ch3.html: Lessons/01-Introduction-to-GIS-and-Python/demo-raster-data-pgd-ch3.ipynb
	jupyter nbconvert --to html --embed-images $<

Lessons/01-Introduction-to-GIS-and-Python/demo-vector-data-osm.html: Lessons/01-Introduction-to-GIS-and-Python/demo-vector-data-osm.ipynb
	jupyter nbconvert --to html --embed-images $<

Lessons/09-Change-Detection-and-Accuracy-Assessment/demo-lombardia.ipynb: Lessons/09-Change-Detection-and-Accuracy-Assessment/demo-lombardia.py
	jupytext --update --to ipynb $<
	python -c 'import json; from pathlib import Path; path = Path("$@"); notebook = json.loads(path.read_text()); [cell["metadata"].__setitem__("language", "python" if cell["cell_type"] == "code" else "markdown") for cell in notebook["cells"]]; path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")'

Lessons/09-Change-Detection-and-Accuracy-Assessment/demo-lombardia.slides.html: Lessons/09-Change-Detection-and-Accuracy-Assessment/demo-lombardia.ipynb
	jupyter nbconvert --to slides --execute $<

Lessons/09-Change-Detection-and-Accuracy-Assessment/pdf/irs-chapter-15-slides.pdf: ../cse620b/Lessons/IRS-Chapter-15-Change-Detection/irs-chapter-15-slides.md
	mkdir -p "$(dir $@)"
	marp --pdf --allow-local-files --output "$@" $<

Lessons/10-Hyperspectral-Remote-Sensing/demo-hyperspectral-mixtures.ipynb: Lessons/10-Hyperspectral-Remote-Sensing/demo-hyperspectral-mixtures.py
	jupytext --update --to ipynb $<
	python -c 'import json; from pathlib import Path; path = Path("$@"); notebook = json.loads(path.read_text()); [cell["metadata"].__setitem__("language", "python" if cell["cell_type"] == "code" else "markdown") for cell in notebook["cells"]]; path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")'

Lessons/10-Hyperspectral-Remote-Sensing/demo-hyperspectral-mixtures.slides.html: Lessons/10-Hyperspectral-Remote-Sensing/demo-hyperspectral-mixtures.ipynb
	jupyter nbconvert --to slides --output-dir "$(dir $@)" --output "$(basename $(basename $(notdir $@)))" $<

Lessons/10-Hyperspectral-Remote-Sensing/slides/IRS-Chapter-14-Hyperspectral-Remote-Sensing.pdf: ../cse620b/Lessons/12-Week-10-Change-Detection-and-Advanced-Hyperspectral-Analysis/advanced-hyperspectral/slides/IRS-Chapter-14-Slides/IRS-Chapter-14-Hyperspectral-Remote-Sensing.md $(wildcard ../cse620b/Lessons/12-Week-10-Change-Detection-and-Advanced-Hyperspectral-Analysis/advanced-hyperspectral/slides/IRS-Chapter-14-Slides/images/*.png)
	mkdir -p "$(dir $@)"
	marp --pdf --allow-local-files --output "$@" $<

# No rebuild rule yet (only a .md source exists privately, not a built PDF/PPTX):
#   Lessons/07-Microwave-Lidar-and-Thermal-Remote-Sensing/pdf/clouds.pdf
#   Lessons/09-Change-Detection-and-Accuracy-Assessment/pdf/irs-chapter-15-slides.pdf
#   Lessons/11-Remote-Sensing-in-Forestry-and-Vegetation/pdf/IRS-Chapter-18-Slides.pdf
#   Lessons/11-Remote-Sensing-in-Forestry-and-Vegetation/pdf/IRS-Chapter-18-Slides.pptx
# Once the corresponding Lessons/<module>/*.md deck is built (marp/soffice) in the
# private repo, add a matching rule here pointing at that built output.
