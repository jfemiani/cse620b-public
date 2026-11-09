# CSE620B - Computer Vision for Remote Sensing

## Course website

The [course site](https://jfemiani.github.io/cse620b-public/) organizes materials
by topic in presentation order. Each topic has lecture notes, slides, demos,
labs, and practice-quiz tabs. Materials open in the page; notebooks also have
Colab and download links. Labs and practice quizzes are explicitly marked
unavailable until authored. The site does not accept submissions or grades.

Topic order, public titles, and resource assignments live in `course.json`.
Edit notes in `notes/`, rather than the generated topic pages in `pages/`.
The original numbered Canvas-page URLs redirect to topic-named pages, so
existing links continue to work. This changes the public website only, not
the titles of live Canvas items.

To preview changes:

```bash
npm ci
npm run build
npm test
python3 -m http.server 8000
```

Open `http://localhost:8000/_site/`. The Pages workflow runs the same build and checks
before deployment. Add only reviewed student-facing resources to `course.json`;
a referenced file that is missing will fail the build.

Lecture PDFs are displayed in Reveal.js using PDF.js, with PDF downloads kept
alongside the viewer. This preserves existing decks without pretending that
PDFs contain editable Reveal or Marp source. Existing notebook HTML/Reveal
exports are embedded directly. Marp can generate standalone HTML, but does not
natively export Reveal.js. The staged site hosts Reveal.js, PDF.js, and the
existing notebook exports' RequireJS and MathJax dependencies locally.

Four image-classification lecture PDFs remain Canvas-only; the notes label
those links as requiring course access. They are not represented as public
downloads.

### Miami University, Fall 2024

- **Class Schedule**: Tuesday & Thursday, 2:50 PM - 4:10 PM  
- **Location**: 166 McVey Data Science Building  
- **Dates**: August 26 to December 6, 2024  

This course will cover advanced topics in computer vision with a focus on applications in remote sensing. Students will learn about image processing techniques, machine learning, and how to apply these in remote sensing contexts. The course will include both theoretical foundations and hands-on exercises.



## Weekly Schedule

### Week 1: Aug 26, 2024  
-  GIS Overview and Geospatial Data Types


### Week 2: Sept 3, 2024  
- Geospatial Data Types and Python for GIS

### Week 3: Sept 10, 2024 
- Quantumn GIS for Geospatial Analysis


### Week 4: Sept 17, 2024  
- Optical Radiation ans Sensor Models


### Week 5: Sept 24, 2024  
- Remote Sensing Platforms
- Digital Mapping Cameras

### Week 6: Oct 1, 2024  
- Image Statistics and Preprocessing

### Week 7: Oct 8, 2024  
- Code Demo [Image Statistics]

### Week 8: Oct 15, 2024  
- Code Demo - Land Use and Land Cover Classification 

### Week 9: Oct 22, 2024  
- Code Demo - Land Use and Land Cover Classifiaction (Deep Learning)
  - Source images (here)
  - Notebook (here0)

### Week 10: Oct 29, 2024  

- [Change Detection: slides and Colab demo](https://jfemiani.github.io/cse620b-public/pages/Week%2010%20-%20Change%20Detection.html)
- [Advanced Hyperspectral Data Analysis: slides and Colab demo](https://jfemiani.github.io/cse620b-public/pages/Week%2010%20-%20Advanced%20Hyperspectral%20Data%20Analysis.html)
- [Small Landsat image pair (GitHub Release)](https://github.com/jfemiani/cse620b-public/releases/tag/week10-data-v1)


### Week 11: Nov 5, 2024  


### Week 12: Nov 12, 2024  


### Week 13: Nov 19, 2024  


### Week 14: Nov 26, 2024  


### Week 15: Dec 3, 2024  


## Resources

- **Textbook**: "Introduction to Remote Sensing" (6th Edition) and relevant research papers.
- **Software**: We will use QGIS, Python, Scikit-Learn, Scikit-Image, Rasterio, Geopandas, and deep learning frameworks such as  PyTorch.
  
## Projects

Students will work on team projects focused on computer vision techniques applied to remote sensing data. Each team will choose a specific problem (e.g., land use classification, environmental monitoring) and apply methods learned in class.

## Contact Information

For any questions regarding the course, please feel free to reach out during office hours or by email.
