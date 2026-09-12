# 🌍 Drought Impact Analysis Dataset - India (2000-2023)

## Comprehensive Multi-Source Groundwater & Agricultural Productivity Data

---

## 📋 Quick Overview

This dataset provides **23+ years of comprehensive drought-related data** for three critical drought-prone regions in India, combining satellite-based groundwater indicators with district-level agricultural productivity metrics.

### 🎯 What's Inside?

- ✅ **GRACE & GLDAS** satellite data (27 districts, 2000-2023)
- ✅ **Agricultural productivity** metrics (48 years, 75+ crop variables)
- ✅ **Climate indicators** (NDVI, SPEI, Rainfall, Soil Moisture)
- ✅ **Ready-to-use** cleaned CSV/Excel files
- ✅ **Jupyter notebooks** for data collection & EDA

---

## 📊 Dataset Description

### About This Dataset

This dataset was created as part of research on **"Modeling Drought Impacts on Agricultural Productivity: A Multi-Scale Time Series Analysis"** at Christ University. It combines multiple authoritative sources to enable comprehensive drought impact studies.

### Study Regions (27 Districts Total)

1. **Marathwada** (Maharashtra) - 8 districts

   - Aurangabad, Beed, Hingoli, Jalna, Latur, Nanded, Osmanabad, Parbhani
2. **Bundelkhand** (UP & MP) - 13 districts

   - Banda, Chitrakoot, Hamirpur, Jalaun, Jhansi, Lalitpur, Mahoba, Chhatarpur, Damoh, Datia, Panna, Sagar, Tikamgarh
3. **Eastern Tamil Nadu** - 6 districts

   - Cuddalore, Nagapattinam, Ramanathapuram, Thanjavur, Tiruvarur, Pudukkottai

---

## 📁 Files in This Dataset

### 1. Primary Data (Google Earth Engine Collected)

#### 📁 `India_Drought_Analysis_Data/`

| File                                     | Size  | Description                         | Time Period | Records |
| ---------------------------------------- | ----- | ----------------------------------- | ----------- | ------- |
| `drought_regions_grace_2003_2008.csv`  | ~2 MB | GRACE groundwater storage anomalies | 2003-2008   | ~2,000+ |
| `drought_regions_grace_2009_2017.csv`  | ~3 MB | GRACE groundwater storage anomalies | 2009-2017   | ~3,000+ |
| `drought_regions_gldas_2000_2002.xlsx` | ~1 MB | GLDAS root zone soil moisture       | 2000-2002   | ~1,000+ |
| `drought_regions_gldas_2018_2023.csv`  | ~4 MB | GLDAS root zone soil moisture       | 2018-2023   | ~4,000+ |

**Columns**: `date`, `ADM2_NAME` (district), `mean` (indicator value), `source` (GRACE/GLDAS), `state`, `region`

### 2. Agricultural Data (ICRISAT)

| File                                    | Size    | Description                   | Time Period | Records  |
| --------------------------------------- | ------- | ----------------------------- | ----------- | -------- |
| `ICRISAT-District Level Data (1).csv` | ~800 KB | Comprehensive crop statistics | 1966-2014   | ~15,000+ |

**Crops Included**: Rice, Wheat, Sorghum, Pearl Millet, Maize, Chickpea, Pigeonpea, Groundnut, Cotton, Sugarcane, and 65+ more

**Metrics**: Area (1000 ha), Production (1000 tons), Yield (kg/ha)

### 3. Climate & Vegetation Indices

| File                                                                                           | Size    | Description                    | Time Period |
| ---------------------------------------------------------------------------------------------- | ------- | ------------------------------ | ----------- |
| `NDVI_Data_1998_2013_India_Regions.xlsx`                                                     | ~500 KB | MODIS/Landsat NDVI values      | 1998-2013   |
| `Marathwada_SPEI.xlsx`                                                                       | ~100 KB | Standardized Precip-Evap Index | Multi-year  |
| `Bundelkhand_SPEI.xlsx`                                                                      | ~100 KB | Standardized Precip-Evap Index | Multi-year  |
| `CHIRPS_TimeSeries_ThreeRegions.xlsx`                                                        | ~200 KB | Rainfall time series           | Multi-year  |
| `Volumetric Soilmoisture_Monthly data for Districts for 2018-07 to 2025-06 Bundelkhand.xlsx` | ~150 KB | IMD soil moisture observations | 2018-2025   |

### 4. Code & Notebooks

| File                                  | Description                                 |
| ------------------------------------- | ------------------------------------------- |
| `project_datacollection.ipynb`      | Google Earth Engine data collection scripts |
| `eda_time_series_project.ipynb`     | Comprehensive exploratory data analysis     |
| `Eda_time_series_project (1).ipynb` | Additional EDA for specific datasets        |

---

## 🔍 Column Descriptions

### GRACE/GLDAS Files

```python
{
    'date': 'Observation date (YYYY-MM-DD)',
    'ADM2_NAME': 'District name',
    'ADM1_NAME': 'State name',  
    'mean': 'Mean value of the indicator',
    'source': 'Data source (GRACE or GLDAS)',
    'region': 'Study region (Marathwada/Bundelkhand/Tamil Nadu)'
}
```

**Units**:

- GRACE: cm (Liquid Water Equivalent thickness anomaly)
- GLDAS: kg/m² (Root zone soil moisture)

### ICRISAT Agricultural Data

```python
{
    'Dist Code': 'District identifier code',
    'Year': 'Calendar year',
    'State Code': 'State identifier',
    'State Name': 'State name',
    'Dist Name': 'District name',
    '[CROP] AREA (1000 ha)': 'Cultivated area in thousands of hectares',
    '[CROP] PRODUCTION (1000 tons)': 'Production in thousands of tons',
    '[CROP] YIELD (Kg per ha)': 'Yield in kilograms per hectare'
}
```

**75+ crop variables** including cereals, pulses, oilseeds, cash crops, fruits, and vegetables.

---

## 💡 Usage Ideas & Applications

### 1. Time Series Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load GRACE data
grace = pd.read_csv('drought_regions_grace_2003_2008.csv')
grace['date'] = pd.to_datetime(grace['date'])

# Time series plot for a specific district
district_data = grace[grace['ADM2_NAME'] == 'Aurangabad']
plt.plot(district_data['date'], district_data['mean'])
plt.title('Groundwater Storage Anomaly - Aurangabad')
plt.show()
```

### 2. Drought Impact on Crops

```python
# Load agricultural data
crops = pd.read_csv('ICRISAT-District Level Data (1).csv')

# Analyze yield trends
marathwada = crops[crops['Dist Name'].isin(['Aurangabad', 'Beed', 'Latur'])]
yield_trends = marathwada.groupby('Year')['RICE YIELD (Kg per ha)'].mean()
```

### 3. Regional Comparison

```python
# Compare groundwater trends across regions
for region in ['Marathwada', 'Bundelkhand', 'Tamil Nadu']:
    region_data = grace[grace['region'] == region]
    plt.plot(region_data['date'], region_data['mean'], label=region)
plt.legend()
plt.show()
```

### 4. Correlation Analysis

```python
# Merge groundwater and agricultural data
# Analyze correlation between drought indicators and crop yields
```

---

## 🎓 Recommended Analyses

### Beginner Level

1. **Exploratory Data Analysis**: Visualize trends, seasonality, and patterns
2. **Descriptive Statistics**: Mean, median, variance across regions
3. **Time Series Plots**: Visualize drought indicators over time

### Intermediate Level

4. **Seasonal Decomposition**: STL decomposition of time series
5. **Correlation Studies**: Relationship between groundwater and crop yields
6. **Regional Comparison**: Statistical tests (t-test, ANOVA)
7. **Drought Event Detection**: Identify drought periods using thresholds

### Advanced Level

8. **ARIMA/SARIMA Modeling**: Time series forecasting
9. **Machine Learning**: LSTM, Random Forest for prediction
10. **Causal Analysis**: Granger causality tests
11. **Spatial-Temporal Modeling**: Incorporate geographic relationships
12. **Prophet Forecasting**: Forecast future drought conditions

---

## 🛠️ Quick Start Code

### Installation

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels jupyter
```

### Load All Data

```python
import pandas as pd
import glob

# Load GRACE data
grace_files = glob.glob('India_Drought_Analysis_Data/drought_regions_grace*.csv')
grace_data = pd.concat([pd.read_csv(f) for f in grace_files])

# Load GLDAS data
gldas_2018 = pd.read_csv('India_Drought_Analysis_Data/drought_regions_gldas_2018_2023.csv')

# Load agricultural data
crops = pd.read_csv('ICRISAT-District Level Data (1).csv')

# Load NDVI
ndvi = pd.read_excel('NDVI_Data_1998_2013_India_Regions.xlsx', sheet_name='All_Data_Combined')

print(f"GRACE records: {len(grace_data)}")
print(f"GLDAS records: {len(gldas_2018)}")
print(f"Crop records: {len(crops)}")
print(f"NDVI records: {len(ndvi)}")
```

### Basic Analysis

```python
# Summary statistics
print(grace_data.groupby('region')['mean'].describe())

# Visualize by region
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
for region in grace_data['region'].unique():
    data = grace_data[grace_data['region'] == region]
    data = data.sort_values('date')
    plt.plot(data['date'], data['mean'], label=region, alpha=0.7)

plt.xlabel('Date')
plt.ylabel('GRACE LWE Thickness (cm)')
plt.title('Groundwater Storage Anomaly by Region')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## 📚 Data Sources & Methodology

### Primary Data Collection (Google Earth Engine)

**GRACE (Gravity Recovery and Climate Experiment)**

- **Source**: NASA JPL
- **Collection**: `NASA/GRACE/MASS_GRIDS/LAND`
- **Processing**: Monthly mean composites, district-level spatial aggregation
- **Quality**: Peer-reviewed, validated against in-situ measurements

**GLDAS (Global Land Data Assimilation System)**

- **Source**: NASA Goddard Space Flight Center
- **Collection**: `NASA/GLDAS/V021/NOAH/G025/T3H`
- **Processing**: 3-hourly data aggregated to monthly means
- **Model**: Noah Land Surface Model v2.1

**Data Collection Workflow**:

1. Define district boundaries using FAO GAUL 2015 Level 2
2. Extract time series using Google Earth Engine Python API
3. Apply spatial reducers (mean) for district-level aggregation
4. Export to CSV via Google Drive
5. Quality checks and validation

### Secondary Data

**ICRISAT**: International Crops Research Institute for the Semi-Arid Tropics
**IMD**: India Meteorological Department
**CHIRPS**: Climate Hazards Group InfraRed Precipitation with Station data

---

## ⚠️ Important Notes & Limitations

### Data Gaps

1. **GRACE Mission Gap**: 2017-2018 (GRACE to GRACE-FO transition)
2. **ICRISAT**: Some districts have missing values for certain years
3. **GLDAS**: Gap period 2003-2017 (covered by GRACE)

### Usage Considerations

- GRACE data represents **anomalies** (deviations from baseline), not absolute values
- GLDAS provides model-based estimates, not direct observations
- District boundaries may have changed over the 48-year period
- Agricultural data quality varies by state and time period

### Data Quality

✅ **Validated** against published research
✅ **Cleaned** for obvious errors
✅ **Standardized** formats across files
⚠️ **As-is** for historical agricultural data (minor inconsistencies may exist)

---

## 🤝 Collaboration & Citation

### How to Cite This Dataset

```bibtex
@dataset{drought_analysis_india_2024,
  author = {[Your Name/Group]},
  title = {Drought Impact Analysis Dataset - India (2000-2023)},
  year = {2024},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/YOUR_USERNAME/drought-analysis-india}
}
```

### Research Paper

**Title**: Modeling Drought Impacts on Agricultural Productivity: A Multi-Scale Time Series Analysis
**Institution**: Christ University
**Year**: 2024

---

## 🌟 Acknowledgments

- **NASA** for GRACE and GLDAS data via Google Earth Engine
- **ICRISAT** for district-level agricultural statistics
- **IMD** for climate observations
- **Google Earth Engine** for the excellent geospatial platform
- **Christ University** for research support

---

## 📊 Dataset Statistics Summary

| Metric                        | Value                                              |
| ----------------------------- | -------------------------------------------------- |
| **Total Files**         | 12+ main data files                                |
| **Total Records**       | 35,000+ observations                               |
| **Time Span**           | 1966-2023 (58 years)                               |
| **Geographic Coverage** | 27 districts across 3 regions                      |
| **Variables**           | 100+ including drought indicators and crop metrics |
| **File Formats**        | CSV, XLSX                                          |
| **Total Size**          | ~15 MB (compressed)                                |

---

## 🔗 Related Resources

### GitHub Repository

Full code, notebooks, and documentation:
**[https://github.com/YOUR_USERNAME/drought-analysis-india](https://github.com/YOUR_USERNAME/drought-analysis-india)**

### Kaggle Notebooks

- [EDA: Drought Patterns in India (2000-2023)](#) - Coming Soon
- [Time Series Forecasting: GRACE Groundwater](#) - Coming Soon
- [Crop Yield Prediction Using Drought Indicators](#) - Coming Soon

### External Links

- [Google Earth Engine](https://earthengine.google.com/)
- [GRACE Mission](https://grace.jpl.nasa.gov/)
- [GLDAS Documentation](https://ldas.gsfc.nasa.gov/gldas/)
- [ICRISAT Data Portal](http://data.icrisat.org/)

---

## 📈 Update History

| Version | Date     | Changes                               |
| ------- | -------- | ------------------------------------- |
| v1.0    | Nov 2024 | Initial release with complete dataset |
| v1.1    | TBD      | Add GRACE-FO data (2018-2023)         |
| v1.2    | TBD      | Include additional climate variables  |

---

## 🎯 Use Cases & Competitions

This dataset is ideal for:

✅ **Kaggle Competitions**: Time series forecasting, climate impact modeling
✅ **Research Projects**: Drought studies, agricultural economics
✅ **Academic Assignments**: Data science, environmental science courses
✅ **Policy Analysis**: Understanding regional vulnerability
✅ **ML Projects**: Predictive modeling, deep learning applications

---

## 💬 Discussion & Support

Have questions or insights? Let's discuss!

- **Questions**: Comment on this dataset or check the GitHub Issues
- **Notebooks**: Share your analysis notebooks with the community
- **Improvements**: Suggestions for additional data or variables welcome
- **Collaborations**: Open to research collaborations

---

## 📜 License

**License**: MIT License (permissive open-source)

You are free to:

- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use in private projects
- ✅ Use in academic research

**Requirements**:

- Provide attribution
- Include license copy

---

## 🏆 Featured Analyses (Coming Soon)

We'll showcase the best community notebooks here:

1. Best EDA Notebook: [TBD]
2. Best Forecasting Model: [TBD]
3. Most Innovative Use: [TBD]

**Submit your notebook** to be featured!

---

## 🔔 Updates & Notifications

⭐ **Star this dataset** to receive notifications about updates
📧 **Follow the author** for more datasets
💬 **Join the discussion** in the comments section

---

## 📞 Contact

**Dataset Creator**: [Kevin George]

**Email**: kmgs452003@gmail.com
**Kaggle**: https://www.kaggle.com/kevinmathewsgeorge
**LinkedIn**: www.linkedin.com/in/kevin-m-george

---

**Last Updated**: November 2024
**Version**: 1.0.0

---

🌍 **Making drought research accessible to everyone** 🌍

*If this dataset helps your research or project, please cite and give credit. Happy analyzing!* 📊✨
