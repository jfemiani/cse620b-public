

FILES=demo-lulc-pytorch.py
FILES+=demo-lulc.ipynb 

all: ${FILES}

demo-lulc-pytorch.py: ../cse620b/Slides/src/IRS-Chapter-6-Slides/demo-lulc-pytorch.py
	cp ../cse620b/Slides/src/IRS-Chapter-6-Slides/demo-lulc-pytorch.py  demo-lulc-pytorch.py

demo-lulc.ipynb:  ../cse620b/Slides/src/IRS-Chapter-6-Slides/demo-lulc.ipynb
	cp ../cse620b/Slides/src/IRS-Chapter-6-Slides/demo-lulc.ipynb demo-lulc.ipynb