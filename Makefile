

FILES=demo-lulc-pytorch.py
FILES+=demo-lulc.ipynb
FILES+=Lecture-1-GIS-Overview-and-Geospatial-Data-Types.pdf

all: ${FILES}

demo-lulc-pytorch.py: ../cse620b/Slides/src/IRS-Chapter-6-Slides/demo-lulc-pytorch.py
	cp ../cse620b/Slides/src/IRS-Chapter-6-Slides/demo-lulc-pytorch.py  demo-lulc-pytorch.py

demo-lulc.ipynb:  ../cse620b/Slides/src/IRS-Chapter-6-Slides/demo-lulc.ipynb
	cp ../cse620b/Slides/src/IRS-Chapter-6-Slides/demo-lulc.ipynb demo-lulc.ipynb

Lecture-1-GIS-Overview-and-Geospatial-Data-Types.pdf: ../cse620b/Lessons/01-Introduction-to-GIS-and-Python/pdf/Lecture-1-GIS-Overview-and-Geospatial-Data-Types.pdf
	cp ../cse620b/Lessons/01-Introduction-to-GIS-and-Python/pdf/Lecture-1-GIS-Overview-and-Geospatial-Data-Types.pdf Lecture-1-GIS-Overview-and-Geospatial-Data-Types.pdf