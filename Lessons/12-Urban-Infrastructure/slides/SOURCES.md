# Image, data, and research sources

These are the sources reviewed while preparing the lecture. Figure captions are embedded in the prepared image files except where an unmodified figure is retained. This list supplies clickable source and license links. License statements for an article do not automatically establish separate rights in any third-party images or basemaps credited by the authors.

## Replacements and source links

Previously flagged Geoxy, Radiant Earth, Asoka Hospital, USGS flood, and Vancouver example images are not included in this package. The slides instead contain public-domain, CC0, CC BY, or original schematic alternatives, and link to the original examples where useful. Original source research remains listed below. Check third-party material within reused research figures if wider redistribution requires further assurance.

## Visuals

- **Chicago title:** USDA National Agriculture Imagery Program, via USGS EROS, [Chicago City Hall – NAIP](https://www.usgs.gov/media/images/chicago-city-hall-naip), 2019. USGS identifies the image as public domain.
- **Flood photograph:** Jocelyn Augustino / FEMA, [New Orleans flooding, 2005](https://www.nist.gov/image/neighborhood-new-orleans-flooded-hurricane-katrina), public domain. Original image shown after resizing and credit added.
- **Washington divider:** [USGS National Map, Washington, D.C. orthoimage](https://www.usgs.gov/media/images/orthoimage-over-washington-dc). USGS marks it public domain.
- **Guangzhou land-use example:** Shi et al., [Urban Land Use and Land Cover Classification Using Multisource Remote Sensing Images and Social Media Data](https://doi.org/10.3390/rs11222719), 2019, CC BY 4.0. The classification combines physical cover and functional classes, so it is not a clean cover-versus-use taxonomy.
- **Prediction vs reference:** Rota et al., [SegFVG dataset](https://doi.org/10.1038/s41597-025-06014-4), 2025. Article CC BY 4.0; first row and column labels extracted for teaching. The reference is not automatically error-free.
- **Morogoro imagery and land use:** Sumari et al., [A Geospatial Approach to Sustainable Urban Planning](https://doi.org/10.3390/su11226508), 2019, CC BY 4.0. The right-hand panel is a *master plan*, not necessarily observed use.
- **Scene classification, detection and segmentation:** Maxwell et al., [Accuracy Assessment, Part 1](https://doi.org/10.3390/rs13132450), 2021, CC BY 4.0. Cropped into separate images for legibility. See also their [Part 2 recommendations](https://doi.org/10.3390/rs13132591).
- **Exposure example:** Original schematic diagram, not a real flood map. The slide links to Darlington et al., [Mapping current and future flood exposure](https://doi.org/10.5194/nhess-24-699-2024), 2024, as an optional real-world example.
- **Point-cloud divider:** Zhao et al., [Robust Normal Estimation for 3D LiDAR Point Clouds in Urban Environments](https://doi.org/10.3390/s19051248), 2019, CC BY 4.0. Middle panel cropped.
- **Resolution comparison:** USDA NAIP via [USGS National Map NAIP imagery service](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer), public domain in the conterminous US. A Chicago area was requested at 800 Web Mercator metres across 1000 pixels (~0.6 ground m/pixel at latitude 41.886°); nominal 1.2 m, 3 m, and 10 m panels were created by block averaging and nearest-neighbor display. Actual native sensor resolution and effective optical resolution may differ. No ~0.3 m panel can be recovered from this image. [Radiant Earth original](https://medium.com/radiant-earth-insights/observing-the-earth-fueling-global-development-solutions-1c69fd5632bc) is linked, not reproduced.
- **Why buildings are hard:** Uses the public-domain USDA/USGS NAIP scene above. The prior example from Q. Yuan, [Building rooftop extraction](https://doi.org/10.1038/s41598-025-91206-6), is linked rather than reproduced.
- **Imagery and OSM labels:** Humanitarian OpenStreetMap Team, [VHR building segmentation dataset](https://huggingface.co/datasets/hotosm/vhr-building-segmentation). Original repository image: Open Imagery Network contributors, CC BY 4.0; labels OpenStreetMap contributors, ODbL 1.0. See dataset documentation for precise attribution.
- **Multimodal inputs:** Pan et al., [SDA-Net](https://doi.org/10.3390/s25072112), 2025, CC BY 4.0. The figure compares optical+DSM and optical+SAR examples, not raw lidar point clouds in every panel.
- **Roof offset:** Wu et al., [Utilizing Building Offset and Shadow to Retrieve Urban Building Heights with ICESat-2 Photons](https://doi.org/10.3390/rs15153786), 2023, CC BY 4.0.
- **3D models:** © 3DBAG by tudelft3d and 3DGI, [data-layer documentation](https://docs.3dbag.nl/en/schema/layers/) and [copyright](https://docs.3dbag.nl/en/copyright/), CC BY 4.0. Credit strip added. The model diagram is Dutch; the tangent separately describes [Geoxy’s Polish project](https://www.geoxy.pl/en.budynki_3d_lod2_city_gml.html) and links to the [Polish geoportal](https://www.geoportal.gov.pl/en/data/other-data/3d-models-of-building/).
- **Elevated connector:** Arlo Barnes, [Sky bridge](https://commons.wikimedia.org/wiki/File:Sky_bridge.jpg), CC0 1.0; credit strip added. The earlier [Asoka Hospital example](https://architectures.jidipi.com/j00109988/en/asoka-hospital-south-wing) is linked but not reproduced.
- **Denver / Microsoft footprint comparison:** Pourpeikari Heris et al., [A rasterized building footprint dataset for the United States](https://doi.org/10.1038/s41597-020-0542-3), 2020, CC BY 4.0. Reported recall is lower for progressively smaller buildings; neither compared dataset should automatically be treated as error-free.
- **Synthetic figures:** confusion outcomes, overlap, geographic holdout, exposure overlay, and model tree are original schematic diagrams produced for the deck; they do not depict observed data. The two-roof mask figure was already in the repository.

## Recent methods and venues

- Adimoolam et al., [Pix2Poly](https://openaccess.thecvf.com/content/WACV2025/html/Adimoolam_Pix2Poly_A_Sequence_Prediction_Method_for_End-to-End_Polygonal_Building_Footprint_WACV_2025_paper.html), WACV 2025: end-to-end polygon representation.
- Li et al., [PolyFootNet](https://doi.org/10.1109/TGRS.2025.3590054), IEEE TGRS 2025: polygonal footprints in off-nadir imagery, with roof and offset cues.
- Yin et al., [SAM-Road++](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_Towards_Satellite_Image_Road_Graph_Extraction_A_Global-Scale_Dataset_and_CVPR_2025_paper.pdf), CVPR 2025: road graph extraction; benchmark claims are dataset- and metric-specific.
- [RoadTracer](https://mapster.csail.mit.edu/roadtracer.html) and [Sat2Graph](https://arxiv.org/abs/2007.09547): graph-oriented road mapping.
- [2026 multi-dataset building-instance benchmark](https://www.mdpi.com/1424-8220/26/12/3686): compares YOLO-seg families and Mask R-CNN, and reports WHU-to-Inria domain shift. Its results do not establish a universal winner.
- [SpaceNet datasets and challenges](https://spacenet.ai/datasets/): buildings, off-nadir imagery, road networks and flood response.

Potential publication venues include ISPRS Journal of Photogrammetry and Remote Sensing; IEEE Transactions on Geoscience and Remote Sensing; IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing; Remote Sensing; and CVPR, ICCV, ECCV and WACV when the work fits their scope.
- Huang et al., [ArcPro](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_ArcPro_Architectural_Programs_for_Structured_3D_Abstraction_of_Sparse_Points_CVPR_2025_paper.html), CVPR 2025: structured architectural programs inferred from sparse points, providing a mesh representation.
