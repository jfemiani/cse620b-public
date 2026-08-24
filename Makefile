

# Public repo layout mirrors the private repo's Lessons/NN-Module-Name/ numbering,
# so each target's destination path lines up with its module folder here.
# Only files with a known, reproducible private-repo source get a build rule;
# see the "no rebuild rule" note below for the rest.

FILES=Lessons/01-Introduction-to-GIS-and-Python/slides/Lecture-1-GIS-Overview-and-Geospatial-Data-Types.pdf
FILES+=Lessons/01-Introduction-to-GIS-and-Python/slides/Lecture-2-Geospatial-Data-Types-and-Python-for-GIS.pdf
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc.ipynb
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-pytorch.py
FILES+=Lessons/06-Image-Classification-and-Land-Cover/demo-deepglobe.ipynb

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

Lessons/06-Image-Classification-and-Land-Cover/demo-lulc-pytorch.py: ../cse620b/Lessons/IRS-Chapter-6-Slides/demo-lulc-pytorch.py
	mkdir -p Lessons/06-Image-Classification-and-Land-Cover
	cp $< $@

Lessons/06-Image-Classification-and-Land-Cover/demo-deepglobe.ipynb: ../cse620b/Lessons/IRS-Chapter-6-Slides/demo-deepglobe.ipynb
	mkdir -p Lessons/06-Image-Classification-and-Land-Cover
	cp $< $@

# No rebuild rule yet (only a .md source exists privately, not a built PDF/PPTX):
#   Lessons/07-Microwave-Lidar-and-Thermal-Remote-Sensing/pdf/clouds.pdf
#   Lessons/09-Change-Detection-and-Accuracy-Assessment/pdf/irs-chapter-15-slides.pdf
#   Lessons/11-Remote-Sensing-in-Forestry-and-Vegetation/pdf/IRS-Chapter-18-Slides.pdf
#   Lessons/11-Remote-Sensing-in-Forestry-and-Vegetation/pdf/IRS-Chapter-18-Slides.pptx
# Once the corresponding Lessons/<module>/*.md deck is built (marp/soffice) in the
# private repo, add a matching rule here pointing at that built output.
