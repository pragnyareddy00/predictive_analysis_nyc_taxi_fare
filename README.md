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

#### Key Visualizations in Data Analysis

The application provides the following visualizations in the Data Analysis tab:

1. **Trips by Day of Week**: Bar chart showing the distribution of trips across each day of the week
   
   ![Trips by Day of Week](/images/)

2. **Trips by Hour of Day**: Bar chart displaying the hourly distribution of trips
   
   ![Trips by Hour of Day](/api/placeholder/600/300 "Bar Chart of Trips by Hour of Day")

3. **Payment Type Distribution**: Pie chart showing the breakdown of payment methods used
   
   ![Payment Type Distribution](/api/placeholder/400/400 "Pie Chart of Payment Types")

4. **Trip Type Distribution**: Pie chart illustrating the distribution of trip types
   
   ![Trip Type Distribution](/images/trip_type_distribution)

6. **Average Total Amount by Weekday**: Bar chart comparing average fare amounts across different days
   
   ![Average Amount by Weekday](/api/placeholder/600/300 "Bar Chart of Average Amount by Weekday")

7. **Correlation Matrix**: Heatmap showing correlations between numerical variables
   
   ![Correlation Matrix](/api/placeholder/650/650 "Heatmap of Correlation Matrix")

8. **Total Amount Analysis**: Histogram, boxplot, and density curve of total fare amounts
   
   ![Total Amount Analysis](/api/placeholder/700/250 "Total Amount Distribution Analysis")

### Model Building Tab

1. After uploading data in the Data Analysis tab, navigate to the Model Building tab
2. The application will:
   - Prepare data for modeling (one-hot encoding, train-test split)
   - Train four regression models: Linear Regression, Decision Tree, Random Forest, and Gradient Boosting
   - Display performance metrics and feature importance for each model
   - Save the best model for use in the Prediction tab

#### Model Performance and Accuracy

The application evaluates and compares the following regression models:

| Model | RMSE | R² Score | Training Time |
|-------|------|---------|---------------|
| Linear Regression | ~2.5-4.0 | 0.85-0.95 | Fastest |
| Decision Tree | ~2.0-3.5 | 0.87-0.94 | Fast |
| Random Forest | ~1.5-3.0 | 0.90-0.97 | Moderate |
| Gradient Boosting | ~1.5-2.8 | 0.91-0.98 | Slowest |

*Note: Actual values will vary based on your dataset*

The application displays model performance comparison charts:

![Model Performance Comparison](/api/placeholder/650/350 "Bar Chart of Model Performance Metrics")

#### Feature Importance Analysis

The application identifies which features have the strongest influence on fare predictions:

![Feature Importance](/api/placeholder/650/400 "Bar Chart of Feature Importance")

Common important features include:
- Trip distance
- Trip duration
- Time of day
- Passenger count
- Day of week (particularly weekends vs weekdays)

### Prediction Tab

1. Use the sliders and dropdown menus in the sidebar to input trip details
2. Click the "Predict Total Amount" button to generate a fare estimate
3. View the prediction result and breakdown of fare components

![Prediction Interface](/api/placeholder/700/400 "Prediction Tab Interface")

#### Fare Prediction Breakdown

The application provides a visual breakdown of the fare components:

![Fare Components](/api/placeholder/600/350 "Bar Chart of Fare Components")

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

## Statistical Analysis

The application performs several statistical tests to derive insights:

1. **ANOVA Tests**:
   - Test whether average fare amounts differ significantly across trip types
   - Test whether average fare amounts differ significantly across days of the week
   
   Example result: *"There is a significant difference in average total amount between weekdays (p < 0.05)"*

2. **Chi-Square Test**:
   - Test for association between trip type and payment type
   
   Example result: *"There is a significant association between trip type and payment type (p < 0.05)"*

## Model Accuracy and Performance

### Evaluation Metrics

The application uses two primary metrics to evaluate model performance:

1. **Root Mean Square Error (RMSE)**:
   - Measures the average magnitude of prediction errors
   - Lower values indicate better model performance
   - Typically ranges from $1.50 to $4.00 for NYC taxi fare prediction

2. **R² Score (Coefficient of Determination)**:
   - Measures the proportion of variance in fare amounts explained by the model
   - Values closer to 1.0 indicate better model performance
   - Typically ranges from 0.85 to 0.98 for NYC taxi fare prediction

### Cross-Validation

The models are trained on 80% of the data and tested on the remaining 20% to ensure robust performance evaluation.

## Features

- **Interactive visualizations**: Bar charts, pie charts, heatmaps, and more
- **Statistical tests**: ANOVA, Chi-square tests for insights into trip patterns
- **Machine learning models**: Multiple regression algorithms with performance comparison
- **Feature importance analysis**: Understand which factors most influence fare amounts
- **Interactive prediction**: Real-time fare estimation based on user inputs
- **Data preprocessing**: Automated handling of missing values and categorical encoding

## Sample Data

Sample NYC Green Taxi trip data can be downloaded from the NYC Taxi & Limousine Commission website:
https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Example data structure:

```
lpep_pickup_datetime,lpep_dropoff_datetime,store_and_fwd_flag,RatecodeID,PULocationID,DOLocationID,passenger_count,trip_distance,fare_amount,extra,mta_tax,tip_amount,tolls_amount,ehail_fee,improvement_surcharge,total_amount,payment_type,trip_type,congestion_surcharge
2020-01-01 00:18:35,2020-01-01 00:23:58,N,1,264,264,1,0.51,4.5,0.5,0.5,0,0,,0.3,5.8,2,1,0
2020-01-01 00:12:41,2020-01-01 00:26:47,N,1,97,37,1,3.2,11.5,0.5,0.5,3.05,0,,0.3,15.85,1,1,0
```

## Performance Optimization

- Data caching for improved loading time
- Efficient visualization rendering
- Model persistence to avoid retraining

## Dependencies

- streamlit: Web application framework
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib, seaborn: Data visualization
- scipy: Statistical tests
- scikit-learn: Machine learning algorithms

## Troubleshooting

Common issues and solutions:

1. **Missing columns error**: Ensure your data follows the expected format
2. **Model prediction error**: Verify that a model has been trained in the Model Building tab
3. **Visualization rendering issues**: Try reducing the size of your dataset by sampling

## Future Enhancements

- Geographic analysis with interactive maps
- Time series forecasting of taxi demand
- Driver revenue optimization suggestions
- Customer segmentation analysis


## Contact

For questions or assistance, please contact gpragnyareddy1594@gmail.com .
