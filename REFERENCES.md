# References and Citations

## Primary Research Integration

**Farhadi Nia, M.** (2025). "Explore Cross-Codec Quality-Rate Convex Hulls Relation for Adaptive Streaming." *Department of Electrical and Computer Engineering, University of Massachusetts Lowell, Lowell, MA, USA.*

**Corresponding Author:** Masoumeh Farhadi Nia  
**Email:** Masoumeh_FarhadiNia@student.uml.edu  
**Institution:** University of Massachusetts Lowell  
**Department:** Electrical and Computer Engineering

## Implementation Mapping

### Polynomial Mathematical Models
**Source:** Farhadi Nia, M. (2025), Section 4.3 "Mathematical models of the PSNR Convex Hulls"
- **Section 4.3.1**: H.264 6th-order polynomial coefficients
- **Section 4.3.2**: H.265 5th-order polynomial coefficients  
- **Section 4.3.3**: VP9 5th-order polynomial coefficients
- **Table 3**: RMSE and R-Squared values for polynomial modeling accuracy

**Implemented in:** `quality_rate_predictor.py`

### Content-Adaptive Analysis (SI/TI Metrics)
**Source:** Farhadi Nia, M. (2025), Section 3.1.2 "Content Description by Benefitting from Spatial and Temporal Information"

**Mathematical Formulations:**
- Equation 1: `Mn(i,j) = Fn(i,j) − Fn−1(i,j)`
- Equation 2: `TIn = std(Mn)`
- Equation 3: `TI = mean(TIn)`
- Equation 4: `SIn = std(Sobel(Fn))`
- Equation 5: `SI = mean(SIn)`

**Implemented in:** `content_analyzer.py`

### Multi-Resolution Optimization
**Source:** Farhadi Nia, M. (2025), Section 3.2.2 "Compression Process Across Different Resolutions"
- 960×544 (Quarter HD) resolution profile
- 1920×1080 (Full HD) resolution profile
- 3840×2160 (Ultra HD/4K) resolution profile

**Implemented in:** `resolution_optimizer.py`

### Logarithmic Bitrate Transformation
**Source:** Farhadi Nia, M. (2025), Section 3.3.4 "Model QR Curves Mathematically"
> "It is worth mentioning that we have considered the bitrate logarithmic which means that in the following equation x is replaced with (log x)."

**Implemented in:** `quality_rate_predictor.py` - `predict_vmaf_from_bitrate()` method

## Supporting References from the Paper

**Okamoto, J., Hayashi, T., Takahashi, A., & Kurita, T.** (2006). "Proposal for an objective video quality assessment method that takes temporal and spatial information into consideration." *Electronics and Communications in Japan, Part I: Communications*, 89(12), 97-108. doi: 10.1002/ecja.20265.

**FFmpeg Development Team.** FFmpeg multimedia framework. Available: https://www.ffmpeg.org/

## Additional Academic Context

### Dataset Used in Original Research
**Zhang, F., et al.** (2019). "BVI - Video Codec Evaluation." University of Bristol. Retrieved November 4, 2019.

### VMAF Quality Assessment
**García, B., López-Fernández, L., Gortázar, F., & Gallego, M.** (2019). "Practical evaluation of VMAF perceptual video quality for webRTC applications." *Electronics*, 8(8). doi: 10.3390/electronics8080854.

## Academic Compliance Statement

This implementation strictly follows the mathematical formulations, coefficients, and methodologies published in the referenced academic paper. All polynomial coefficients, equations, and algorithmic approaches are implemented exactly as described in the original research with proper attribution to the authors.

**Key Compliance Elements:**
1. **Exact Mathematical Implementation**: All equations (1-5) implemented precisely as published
2. **Research-Validated Coefficients**: Polynomial coefficients used directly from Tables in Section 4.3
3. **Methodological Accuracy**: SI/TI analysis follows exact academic methodology
4. **Resolution Testing**: Uses identical test resolutions (960×544, 1920×1080, 3840×2160)
5. **Proper Attribution**: Author credit and section references throughout implementation

## Usage Ethics

This implementation is created for educational and research purposes, properly citing the original academic work. The mathematical models and methodologies are implemented with full attribution to the original authors and their academic institution.

**Contact for Academic Inquiries:**
- **Primary Author**: Masoumeh Farhadi Nia (Masoumeh_FarhadiNia@student.uml.edu)
- **Institution**: University of Massachusetts Lowell
- **Department**: Electrical and Computer Engineering
