import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="NYC Green Taxi Analysis", layout="wide")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Analysis", "Model Building", "Prediction"])

with tab1:
    st.title('NYC Green Taxi Trip Data Analysis')
    
    # File uploader
    uploaded_file = st.file_uploader("Upload NYC Green Taxi Trip data (CSV or Parquet)", type=["csv", "parquet"])
    
    # Load data
    @st.cache_data
    def load_data(file):
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.parquet'):
            return pd.read_parquet(file)
        else:
            return None

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            # Display data info
            st.subheader('a) Data Information')
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)
            
            # b) Drop ehail_fee column
            st.subheader('b) Dropping ehail_fee column')
            if 'ehail_fee' in df.columns:
                df = df.drop('ehail_fee', axis=1)
                st.success('ehail_fee column dropped')
            else:
                st.info('ehail_fee column not found in dataset')
            
            # c) Calculate trip_duration
            st.subheader('c) Calculating trip duration in minutes')
            try:
                # Convert to datetime
                df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
                df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
                
                # Calculate duration in minutes
                df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
                
                st.write("Trip duration statistics (minutes):")
                st.write(df['trip_duration'].describe())
            except Exception as e:
                st.error(f"Error calculating trip duration: {e}")
        else:
            st.error("Unsupported file format!")
        
        # d) Extract weekday
        st.subheader('d) Extracting weekday')
        try:
            df['weekday'] = df['lpep_dropoff_datetime'].dt.day_name()
            st.write("Weekday distribution:")
            weekday_counts = df['weekday'].value_counts()
            st.write(weekday_counts)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            weekday_counts.plot(kind='bar', ax=ax)
            plt.title('Trips by Day of Week')
            plt.ylabel('Number of Trips')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error extracting weekday: {e}")
        
        # e) Extract hour of day
        st.subheader('e) Extracting hour of day')
        try:
            df['hourofday'] = df['lpep_dropoff_datetime'].dt.hour
            st.write("Hour of day distribution:")
            hour_counts = df['hourofday'].value_counts().sort_index()
            st.write(hour_counts)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            hour_counts.plot(kind='bar', ax=ax)
            plt.title('Trips by Hour of Day')
            plt.xlabel('Hour')
            plt.ylabel('Number of Trips')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error extracting hour of day: {e}")
        
        # f) Missing Values Imputation
        st.subheader('f) Missing Values Imputation')
        missing_values = df.isnull().sum()
        st.write("Missing values before imputation:")
        st.write(missing_values[missing_values > 0])
        
        # Impute missing values
        numeric_cols = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 
                       'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 
                       'trip_duration', 'passenger_count', 'total_amount']
        
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Categorical imputation
        cat_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'weekday', 'hourofday']
        for col in cat_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        missing_after = df.isnull().sum()
        st.write("Missing values after imputation:")
        st.write(missing_after[missing_after > 0])
        
        # g) Pie diagrams
        st.subheader('g) Pie diagrams of payment_type and trip_type')
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'payment_type' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 8))
                payment_counts = df['payment_type'].value_counts()
                ax.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%')
                ax.set_title('Payment Type Distribution')
                st.pyplot(fig)
            else:
                st.warning('payment_type column not found')
        
        with col2:
            if 'trip_type' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 8))
                trip_counts = df['trip_type'].value_counts()
                ax.pie(trip_counts, labels=trip_counts.index, autopct='%1.1f%%')
                ax.set_title('Trip Type Distribution')
                st.pyplot(fig)
            else:
                st.warning('trip_type column not found')
        
        # h) Groupby() of average total_amount & weekday
        st.subheader('h) Average total amount by weekday')
        if 'weekday' in df.columns and 'total_amount' in df.columns:
            avg_by_weekday = df.groupby('weekday')['total_amount'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            st.write(avg_by_weekday)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            avg_by_weekday.plot(kind='bar', ax=ax)
            plt.title('Average Total Amount by Weekday')
            plt.ylabel('Average Total Amount ($)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning('weekday or total_amount column not found')
        
        # i) Groupby() of average total_amount & payment_type
        st.subheader('i) Average total amount by payment type')
        if 'payment_type' in df.columns and 'total_amount' in df.columns:
            avg_by_payment = df.groupby('payment_type')['total_amount'].mean()
            st.write(avg_by_payment)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            avg_by_payment.plot(kind='bar', ax=ax)
            plt.title('Average Total Amount by Payment Type')
            plt.ylabel('Average Total Amount ($)')
            plt.xlabel('Payment Type')
            st.pyplot(fig)
        else:
            st.warning('payment_type or total_amount column not found')
        
        # j) Groupby() of average tip_amount & weekday
        st.subheader('j) Average tip amount by weekday')
        if 'weekday' in df.columns and 'tip_amount' in df.columns:
            avg_tip_by_weekday = df.groupby('weekday')['tip_amount'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            st.write(avg_tip_by_weekday)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            avg_tip_by_weekday.plot(kind='bar', ax=ax)
            plt.title('Average Tip Amount by Weekday')
            plt.ylabel('Average Tip Amount ($)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning('weekday or tip_amount column not found')
        
        # k) Groupby() of average tip_amount & payment_type
        st.subheader('k) Average tip amount by payment type')
        if 'payment_type' in df.columns and 'tip_amount' in df.columns:
            avg_tip_by_payment = df.groupby('payment_type')['tip_amount'].mean()
            st.write(avg_tip_by_payment)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            avg_tip_by_payment.plot(kind='bar', ax=ax)
            plt.title('Average Tip Amount by Payment Type')
            plt.ylabel('Average Tip Amount ($)')
            plt.xlabel('Payment Type')
            st.pyplot(fig)
        else:
            st.warning('payment_type or tip_amount column not found')
        
        # l) Test null average total_amount of different trip_type is identical
        st.subheader('l) Testing if average total amount is the same across trip types')
        if 'trip_type' in df.columns and 'total_amount' in df.columns:
            trip_types = df['trip_type'].unique()
            if len(trip_types) > 1:
                trip_groups = [df[df['trip_type'] == t]['total_amount'] for t in trip_types]
                f_stat, p_value = stats.f_oneway(*trip_groups)
                st.write(f"ANOVA Test Results: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
                
                if p_value < 0.05:
                    st.write("Conclusion: There is a significant difference in average total amount between trip types (p < 0.05)")
                else:
                    st.write("Conclusion: There is no significant difference in average total amount between trip types (p >= 0.05)")
            else:
                st.warning("Only one trip type found in the data")
        else:
            st.warning('trip_type or total_amount column not found')
        
        # m) Test null average total_amount of different weekday is identical
        st.subheader('m) Testing if average total amount is the same across weekdays')
        if 'weekday' in df.columns and 'total_amount' in df.columns:
            weekday_groups = [df[df['weekday'] == day]['total_amount'] for day in df['weekday'].unique()]
            f_stat, p_value = stats.f_oneway(*weekday_groups)
            st.write(f"ANOVA Test Results: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
            
            if p_value < 0.05:
                st.write("Conclusion: There is a significant difference in average total amount between weekdays (p < 0.05)")
            else:
                st.write("Conclusion: There is no significant difference in average total amount between weekdays (p >= 0.05)")
        else:
            st.warning('weekday or total_amount column not found')
        
        # n) Test null no association between trip_type and payment_type
        st.subheader('n) Testing association between trip_type and payment_type')
        if 'trip_type' in df.columns and 'payment_type' in df.columns:
            contingency_table = pd.crosstab(df['trip_type'], df['payment_type'])
            st.write("Contingency Table:")
            st.write(contingency_table)
            
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            st.write(f"Chi-Square Test Results: chi2 = {chi2:.4f}, p-value = {p:.4f}")
            
            if p < 0.05:
                st.write("Conclusion: There is a significant association between trip type and payment type (p < 0.05)")
            else:
                st.write("Conclusion: There is no significant association between trip type and payment type (p >= 0.05)")
        else:
            st.warning('trip_type or payment_type column not found')
        
        # q) Correlation analysis of numeric cols
        st.subheader('q) Correlation analysis of numeric columns')
        numeric_cols = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 
                       'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 
                       'trip_duration', 'passenger_count', 'total_amount']
        
        numeric_df = df[numeric_cols].copy()
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        plt.title('Correlation Matrix of Numeric Variables')
        st.pyplot(fig)
        
        # s) Dependent Variable Analysis
        st.subheader('s) Dependent Variable (total_amount) Analysis')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("Histogram")
            fig, ax = plt.subplots()
            ax.hist(df['total_amount'], bins=30)
            plt.title('Total Amount Histogram')
            plt.xlabel('Total Amount ($)')
            plt.ylabel('Frequency')
            st.pyplot(fig)
        
        with col2:
            st.write("Boxplot")
            fig, ax = plt.subplots()
            ax.boxplot(df['total_amount'])
            plt.title('Total Amount Boxplot')
            plt.ylabel('Total Amount ($)')
            st.pyplot(fig)
        
        with col3:
            st.write("Density Curve")
            fig, ax = plt.subplots()
            df['total_amount'].plot(kind='density', ax=ax)
            plt.title('Total Amount Density Curve')
            plt.xlabel('Total Amount ($)')
            st.pyplot(fig)

with tab2:
    st.title('NYC Green Taxi Model Building')
    
    if 'df' in locals():
        st.write("Preparing data for model building...")
        
        # Identify columns for model
        numeric_features = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 
                           'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 
                           'trip_duration', 'passenger_count']
        
        categorical_features = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'weekday', 'hourofday']
        
        # r) Dummy encode object columns
        st.subheader('r) One-hot encoding categorical variables')
        
        model_df = df.copy()
        
        # Handle categorical features
        for col in categorical_features:
            if col in model_df.columns:
                if model_df[col].dtype == 'object' or col == 'weekday':
                    # Create dummies for categorical variables
                    dummies = pd.get_dummies(model_df[col], prefix=col, drop_first=True)
                    # Add dummies to dataframe
                    model_df = pd.concat([model_df, dummies], axis=1)
                    # Drop original column
                    model_df = model_df.drop(col, axis=1)
            else:
                st.warning(f"{col} not found in dataset")
        
        # Get list of all features after one-hot encoding
        model_features = [col for col in model_df.columns if col != 'total_amount' and col not in ['lpep_pickup_datetime', 'lpep_dropoff_datetime']]
        
        # Split data into training and testing sets
        X = model_df[model_features]
        y = model_df['total_amount']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.write(f"Training set size: {X_train.shape[0]} rows")
        st.write(f"Testing set size: {X_test.shape[0]} rows")
        st.write(f"Number of features: {X_train.shape[1]}")
        
        # t) Build regression models
        st.subheader('t) Building Regression Models')
        
        model_results = {}
        
        # Linear Regression
        st.write("Training Linear Regression model...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_y_pred = lr_model.predict(X_test)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_y_pred))
        lr_r2 = r2_score(y_test, lr_y_pred)
        model_results['Linear Regression'] = {'RMSE': lr_rmse, 'R²': lr_r2}
        
        # Save the linear regression model
        pickle.dump(lr_model, open('linear_regression_model.pkl', 'wb'))
        st.success("Linear Regression model saved as 'linear_regression_model.pkl'")
        
        # Decision Tree
        st.write("Training Decision Tree model...")
        dt_model = DecisionTreeRegressor(random_state=42)
        dt_model.fit(X_train, y_train)
        dt_y_pred = dt_model.predict(X_test)
        dt_rmse = np.sqrt(mean_squared_error(y_test, dt_y_pred))
        dt_r2 = r2_score(y_test, dt_y_pred)
        model_results['Decision Tree'] = {'RMSE': dt_rmse, 'R²': dt_r2}
        
        # Random Forest
        st.write("Training Random Forest model (100 trees)...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_y_pred = rf_model.predict(X_test)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))
        rf_r2 = r2_score(y_test, rf_y_pred)
        model_results['Random Forest'] = {'RMSE': rf_rmse, 'R²': rf_r2}
        
        # Gradient Boosting
        st.write("Training Gradient Boosting model (100 trees)...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        gb_y_pred = gb_model.predict(X_test)
        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_y_pred))
        gb_r2 = r2_score(y_test, gb_y_pred)
        model_results['Gradient Boosting'] = {'RMSE': gb_rmse, 'R²': gb_r2}
        
        # Display model comparison
        results_df = pd.DataFrame(model_results).T
        st.subheader("Model Comparison")
        st.write(results_df)
        
        # Plot model comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        index = np.arange(len(results_df.index))
        
        bar1 = ax.bar(index, results_df['RMSE'], bar_width, label='RMSE')
        bar2 = ax.bar(index + bar_width, results_df['R²'], bar_width, label='R²')
        
        ax.set_xlabel('Model')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(results_df.index, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Feature importance for best model
        best_model_name = results_df['R²'].idxmax()
        st.subheader(f"Feature Importance for {best_model_name}")
        
        if best_model_name == 'Linear Regression':
            coefficients = pd.DataFrame({
                'Feature': X_train.columns,
                'Coefficient': lr_model.coef_
            })
            coefficients = coefficients.sort_values('Coefficient', ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            coefficients.plot(x='Feature', y='Coefficient', kind='bar', ax=ax)
            plt.title(f'Feature Coefficients for {best_model_name}')
            plt.xticks(rotation=90)
            st.pyplot(fig)
        
        elif best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
            if best_model_name == 'Decision Tree':
                model = dt_model
            elif best_model_name == 'Random Forest':
                model = rf_model
            else:
                model = gb_model
                
            importances = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': model.feature_importances_
            })
            importances = importances.sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            importances.plot(x='Feature', y='Importance', kind='bar', ax=ax)
            plt.title(f'Feature Importance for {best_model_name}')
            plt.xticks(rotation=90)
            st.pyplot(fig)
            
        # Save feature names for the prediction tab
        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(X_train.columns.tolist(), f)
        st.success("Feature names saved for prediction app")
        
    else:
        st.warning("Please upload data in the Data Analysis tab first")

with tab3:
    st.title('NYC Green Taxi Fare Prediction')
    st.write("""
    This app predicts the total fare amount for NYC Green Taxi trips based on various features.
    """)
    
    # Sidebar inputs
    st.sidebar.header('Input Features')
    
    # Create input widgets for all features
    trip_distance = st.sidebar.slider('Trip Distance (miles)', 0.1, 50.0, 2.5)
    fare_amount = st.sidebar.slider('Fare Amount ($)', 0.0, 100.0, 10.0)
    extra = st.sidebar.slider('Extra Charges ($)', 0.0, 10.0, 0.5)
    mta_tax = st.sidebar.slider('MTA Tax ($)', 0.0, 1.0, 0.5)
    tip_amount = st.sidebar.slider('Tip Amount ($)', 0.0, 50.0, 2.0)
    tolls_amount = st.sidebar.slider('Tolls Amount ($)', 0.0, 20.0, 0.0)
    improvement_surcharge = st.sidebar.slider('Improvement Surcharge ($)', 0.0, 1.0, 0.3)
    congestion_surcharge = st.sidebar.slider('Congestion Surcharge ($)', 0.0, 5.0, 2.5)
    trip_duration = st.sidebar.slider('Trip Duration (minutes)', 1, 180, 15)
    passenger_count = st.sidebar.slider('Passenger Count', 1, 6, 1)
    
    # Categorical inputs
    store_and_fwd_flag = st.sidebar.selectbox('Store and Forward Flag', ['N', 'Y'])
    RatecodeID = st.sidebar.selectbox('Rate Code ID', [1, 2, 3, 4, 5, 6])
    payment_type = st.sidebar.selectbox('Payment Type', [1, 2, 3, 4, 5, 6])
    trip_type = st.sidebar.selectbox('Trip Type', [1, 2])
    weekday = st.sidebar.selectbox('Weekday', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    hourofday = st.sidebar.slider('Hour of Day', 0, 23, 12)
    
    # Create a dataframe from inputs
    input_data = {
        'trip_distance': trip_distance,
        'fare_amount': fare_amount,
        'extra': extra,
        'mta_tax': mta_tax,
        'tip_amount': tip_amount,
        'tolls_amount': tolls_amount,
        'improvement_surcharge': improvement_surcharge,
        'congestion_surcharge': congestion_surcharge,
        'trip_duration': trip_duration,
        'passenger_count': passenger_count,
        'store_and_fwd_flag': store_and_fwd_flag,
        'RatecodeID': RatecodeID,
        'payment_type': payment_type,
        'trip_type': trip_type,
        'weekday': weekday,
        'hourofday': hourofday
    }
    features = pd.DataFrame(input_data, index=[0])
    
    # Display input features
    st.subheader('Input Features')
    st.write(features)
    
    # Check if model file exists
    model_file = 'linear_regression_model.pkl'
    
    # Make prediction
    if st.button('Predict Total Amount'):
        try:
            # Try to load saved model
            model = pickle.load(open(model_file, 'rb'))
            
            # Prepare input data (same preprocessing as during training)
            # One-hot encode categorical variables
            pred_df = features.copy()
            
            # Handle categorical features
            categorical_features = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'weekday', 'hourofday']
            for col in categorical_features:
                if col in pred_df.columns:
                    if pred_df[col].dtype == 'object' or col == 'weekday':
                        # Create dummies for categorical variables
                        dummies = pd.get_dummies(pred_df[col], prefix=col, drop_first=True)
                        # Add dummies to dataframe
                        pred_df = pd.concat([pred_df, dummies], axis=1)
                        # Drop original column
                        pred_df = pred_df.drop(col, axis=1)
            
            # Try to load feature names to ensure correct order
            try:
                with open('feature_names.pkl', 'rb') as f:
                    feature_names = pickle.load(f)
                
                # Check which features from training are in the prediction data
                missing_cols = set(feature_names) - set(pred_df.columns)
                for col in missing_cols:
                    pred_df[col] = 0
                
                # Ensure columns are in the right order
                pred_df = pred_df[feature_names]
                
            except FileNotFoundError:
                # If feature names file not found, use as is
                pass
            
            # Make prediction
            prediction = model.predict(pred_df)
            
            st.subheader('Prediction')
            st.write(f'Predicted Total Amount: ${prediction[0]:.2f}')
            
            # Add a visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            components = [
                ('Fare', fare_amount),
                ('Tip', tip_amount),
                ('Extra', extra),
                ('MTA Tax', mta_tax),
                ('Tolls', tolls_amount),
                ('Improv. Surcharge', improvement_surcharge),
                ('Cong. Surcharge', congestion_surcharge),
                ('Additional', prediction[0] - (fare_amount + tip_amount + extra + mta_tax + tolls_amount + improvement_surcharge + congestion_surcharge))
            ]
            
            labels = [c[0] for c in components]
            values = [c[1] for c in components]
            
            # Only show positive values
            pos_labels = [labels[i] for i in range(len(values)) if values[i] > 0]
            pos_values = [values[i] for i in range(len(values)) if values[i] > 0]
            
            ax.bar(pos_labels, pos_values)
            plt.title('Fare Components')
            plt.ylabel('Amount ($)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        except FileNotFoundError:
            # If model file not found, use a simple calculation
            st.info("Trained model not found. Using simple calculation for demonstration.")
            prediction = (fare_amount + extra + mta_tax + tip_amount + tolls_amount + 
                         improvement_surcharge + congestion_surcharge) * 1.1
            
            st.subheader('Prediction (Demo Mode)')
            st.write(f'Estimated Total Amount: ${prediction:.2f}')
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")