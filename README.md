# NYC Green Taxi Analysis

A Streamlit application for analyzing NYC Green Taxi trip data, building predictive models, and estimating total fare amounts.

## Overview

This application provides a comprehensive platform for exploring, analyzing, and modeling NYC Green Taxi trip data. It consists of three main components:

1. **Data Analysis**: Explore and visualize trip data with interactive charts and statistical tests
2. **Model Building**: Train and evaluate regression models to predict fare amounts
3. **Prediction**: Estimate total fare amounts for new trips based on user inputs

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/nyc-green-taxi-analysis.git
   cd nyc-green-taxi-analysis
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn
   ```

4. Run the application:
   ```bash
   streamlit run nyc_green_taxi_app.py
   ```

## Usage

### Data Analysis Tab

1. Upload a CSV file containing NYC Green Taxi trip data using the file uploader
2. The application will automatically:
   - Process the data and display basic information
   - Calculate trip duration and extract temporal features
   - Impute missing values
   - Generate various visualizations and statistical analyses

### Model Building Tab

1. After uploading data in the Data Analysis tab, navigate to the Model Building tab
2. The application will:
   - Prepare data for modeling (one-hot encoding, train-test split)
   - Train four regression models: Linear Regression, Decision Tree, Random Forest, and Gradient Boosting
   - Display performance metrics and feature importance for each model
   - Save the best model for use in the Prediction tab

### Prediction Tab

1. Use the sliders and dropdown menus in the sidebar to input trip details
2. Click the "Predict Total Amount" button to generate a fare estimate
3. View the prediction result and breakdown of fare components

## Data Format

The application expects a CSV file with NYC Green Taxi trip data containing the following columns:

- `lpep_pickup_datetime`: Pickup date and time
- `lpep_dropoff_datetime`: Dropoff date and time
- `trip_distance`: Trip distance in miles
- `fare_amount`: Base fare amount
- `extra`: Extra charges
- `mta_tax`: MTA tax
- `tip_amount`: Tip amount
- `tolls_amount`: Toll charges
- `improvement_surcharge`: Improvement surcharge
- `congestion_surcharge`: Congestion surcharge
- `total_amount`: Total fare amount
- `passenger_count`: Number of passengers
- `payment_type`: Payment method code
- `trip_type`: Trip type code
- `store_and_fwd_flag`: Store and forward flag (Y/N)
- `RatecodeID`: Rate code ID

## Features

- **Interactive visualizations**: Bar charts, pie charts, heatmaps, and more
- **Statistical tests**: ANOVA, Chi-square tests for insights into trip patterns
- **Machine learning models**: Multiple regression algorithms with performance comparison
- **Feature importance analysis**: Understand which factors most influence fare amounts
- **Interactive prediction**: Real-time fare estimation based on user inputs

## Sample Data

Sample NYC Green Taxi trip data can be downloaded from the NYC Taxi & Limousine Commission website:
https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## Dependencies

- streamlit: Web application framework
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib, seaborn: Data visualization
- scipy: Statistical tests
- scikit-learn: Machine learning algorithms

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or assistance, please contact [your.email@example.com].
