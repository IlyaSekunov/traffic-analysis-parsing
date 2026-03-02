# Data Processing Pipeline for Salary Prediction

This project implements a data processing pipeline that transforms raw CSV data into NumPy arrays ready for machine learning models. The pipeline cleans the data, extracts relevant features, and prepares feature matrices (X) and target variables (y) for salary prediction tasks.

## Project Structure

- `app.py` - Main entry point that orchestrates the entire data processing pipeline
- `feature.py` - Contains feature extraction functions to transform raw text data into numerical features
- `handler.py` - Implements a chain of responsibility pattern for data cleaning and preprocessing
- `requirements.txt` - Lists project dependencies

## Features

The pipeline extracts the following features from the raw data:

| Feature | Description | Source Column |
|---------|-------------|---------------|
| `is_male` | Binary indicator if candidate is male | Пол, возраст |
| `age` | Candidate's age in years | Пол, возраст |
| `town` | Encoded town/city name | Город |
| `full_time` | Binary indicator for full-time employment | Занятость |
| `has_car` | Binary indicator if candidate has a car | Авто |
| `higher_education` | Binary indicator for higher education | Образование и ВУЗ |
| `remote_work` | Binary indicator for remote work availability | График |
| `experience` | Years of work experience | Опыт (двойное нажатие для полной версии) |

The target variable is salary (`ЗП` column).

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline by providing the path to your CSV file:

```bash
python app.py path/to/your/hh.csv
```

### Input Format

The input CSV file should contain the following columns:
- `ЗП` (Salary) - Target variable
- `Пол, возраст` (Gender, age) - Contains gender and age information
- `Город` (City) - Candidate's location
- `Занятость` (Employment type) - Full-time/part-time information
- `Авто` (Car) - Car ownership status
- `Образование и ВУЗ` (Education) - Education level and institution
- `График` (Schedule) - Work schedule preferences
- `Опыт (двойное нажатие для полной версии)` (Experience) - Years of experience

### Output

The pipeline generates two NumPy files in the same directory as the input CSV:
- `x_data.npy` - Feature matrix (samples × features)
- `y_data.npy` - Target values (salary)

## Pipeline Components

### Handler Chain

The data processing follows a chain of responsibility pattern:

1. **CleanDataHandler** - Removes Unicode special characters and normalizes text
2. **DropUselessDataHandler** - Removes rows with NA values and duplicates
3. **DropEmptySalaryHandler** - Filters out rows without salary information

### Feature Extraction

The `feature.py` module contains specialized functions to extract numerical features from text data:
- Regular expressions parse age, experience, and other numerical values
- Binary flags are created for categorical attributes
- Missing values are handled appropriately

## Requirements

- Python 3.6+
- pandas
- numpy

## Notes

- The pipeline automatically handles missing values by removing samples without salary data
- Features with missing values are filled with 0
- Town names are encoded using pandas categorical codes
- All features are converted to int64 type for consistency