import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine


connection_string = 'postgresql+psycopg2://postgres:Shankari%40123@localhost:5432/table1'
engine = create_engine(connection_string)
query = 'SELECT * FROM table1'
df = pd.read_sql(query, engine)
df['Date'] = pd.to_datetime(df['Date']).dt.date
try:
    engine = create_engine(r'postgresql+psycopg2://postgres:Shankari%40123@localhost:5432/table2')
    query = "SELECT * FROM table2"
    data = pd.read_sql(query, engine)
    #st.write("Cleaned data retrieved successfully!")
except Exception as e:
    st.error(f"Error fetching data: {e}")
engine = create_engine('postgresql+psycopg2://postgres:Shankari%40123@localhost:5432/table3')
query = "SELECT * FROM table3"
data2= pd.read_sql(query, engine)
print(data)
min_date = df['Date'].min()
max_date = df['Date'].max()

if 'page' not in st.session_state:
    st.session_state.page = 'home'

def navigate_to(page):
    st.session_state.page = page

col1, col2, col3, col4 , col5 = st.columns(5)
with col1:
    if st.button("Home"):
        navigate_to('home')
with col2:
    if st.button("Traffic Flow"):
        navigate_to('graphs')
with col3:
    if st.button("Traffic collision"):
        navigate_to('tc')
with col4:
    if st.button("Harmful gases emission"):
        navigate_to('settings')
with col5:
    if st.button("Conclusion"):
        navigate_to("conclusion")


def home_page():
    st.title("Traffic Flow, Collisions, and Environmental Impact Analysis in Irish Cities for the Year 2022")
    
    st.subheader("Abstract")
    st.write("""The paper presents a comprehensive study on the analysis of traffic flow and road accidents, along with the resultant environmental impact in terms of vehicle emissions in Irish cities for the year 2022. In this context, the present work aims to find the patterns and correlations between traffic dynamics and air quality using data collected from different traffic monitoring systems and environmental sensors.

The analysis reflects a high positive improvement in traffic flow characterized by reduced congestion and an improvement in vehicle movement efficiency. This trend is complemented by a corresponding decline in the rate of traffic collisions, suggesting that improved measures in respect to traffic management and road safety have been effective.

Moreover, the ecological relevance of reduced congestion is such that fewer vehicles idling in such traffic congestion have resulted in lowered emissions of harmful gases like carbon monoxide (CO), nitrogen oxides (NOx), and even particulate matter (PM). This reduction in emission contributes to better air quality in the urban environment.

These findings underscore the need for continued investment in traffic infrastructure and sustainable urban planning. If Irish cities can maintain and build on these improvements, their residents will benefit from safer roads and cleaner air.
""")
    st.subheader("Dataset Preview")
    st.write(data.head())
    st.dataframe(data2)

def graphs_page():    
    with st.sidebar:
        st.title("Date controller and Stats for the Data from Postgre")
        selected_date = st.slider('Select Date:', min_value=min_date, max_value=max_date, value=max_date, format="YYYY-MM-DD")
        filtered_df = df[df['Date'] <= selected_date]

        # Flow Data
        x_series = filtered_df[:302].set_index('Date')['Y_hat_flow']
        y_series = filtered_df[302:].set_index('Date')['Y_hat_flow']
        residuals = x_series - y_series.reindex(x_series.index).fillna(0)
        fig_hist_flow = px.histogram(residuals, title='Flow Residuals Distribution')
        st.plotly_chart(fig_hist_flow)
        st.write("This is the residuals of the traffic flow from the predicted model(Prophet). Basically the reiduals are quite high in the data even after fitting the model.")
        decompose_result_flow = seasonal_decompose(filtered_df.set_index('Date')['Y_hat_flow'], model='additive', period=24)
        fig_decompose_flow = make_subplots(rows=4, cols=1, shared_xaxes=True)
        fig_decompose_flow.add_trace(go.Scatter(x=decompose_result_flow.trend.index, y=decompose_result_flow.trend, name='Trend'), row=1, col=1)
        fig_decompose_flow.add_trace(go.Scatter(x=decompose_result_flow.seasonal.index, y=decompose_result_flow.seasonal, name='Seasonality'), row=2, col=1)
        fig_decompose_flow.add_trace(go.Scatter(x=decompose_result_flow.resid.index, y=decompose_result_flow.resid, name='Residuals'), row=3, col=1)
        fig_decompose_flow.update_layout(height=400, title='Flow Time Series Decomposition')
        st.plotly_chart(fig_decompose_flow)
        st.write("Observed data is been decomposed into trend,seasonal and residuals")
        
        # Congestion Data
        x_series_cong = filtered_df[:302].set_index('Date')['Y_hat_cong']
        y_series_cong = filtered_df[302:].set_index('Date')['Y_hat_cong']
        residuals_cong = x_series_cong - y_series_cong.reindex(x_series_cong.index).fillna(0)
        fig_hist_cong = px.histogram(residuals_cong, title='Congestion Residuals Distribution')
        st.plotly_chart(fig_hist_cong)
        st.write("This is the residuals of the traffic congestion from the predicted model(Prophet). Basically the reiduals are quite high in the data even after fitting the model.")
        decompose_result_cong = seasonal_decompose(filtered_df.set_index('Date')['Y_hat_cong'], model='additive', period=24)
        fig_decompose_cong = make_subplots(rows=4, cols=1, shared_xaxes=True)
        fig_decompose_cong.add_trace(go.Scatter(x=decompose_result_cong.trend.index, y=decompose_result_cong.trend, name='Trend'), row=1, col=1)
        fig_decompose_cong.add_trace(go.Scatter(x=decompose_result_cong.seasonal.index, y=decompose_result_cong.seasonal, name='Seasonality'), row=2, col=1)
        fig_decompose_cong.add_trace(go.Scatter(x=decompose_result_cong.resid.index, y=decompose_result_cong.resid, name='Residuals'), row=3, col=1)
        fig_decompose_cong.update_layout(height=400, title='Congestion Time Series Decomposition')
        st.plotly_chart(fig_decompose_cong)
        st.write("Observed data is been decomposed into trend,seasonal and residuals")
        
        # Density Data
        x_series_dsat = filtered_df[:302].set_index('Date')['Y_hat_dsat']
        y_series_dsat = filtered_df[302:].set_index('Date')['Y_hat_dsat']
        residuals_dsat = x_series_dsat - y_series_dsat.reindex(x_series_dsat.index).fillna(0)
        fig_hist_dsat = px.histogram(residuals_dsat, title='Density Residuals Distribution')
        st.plotly_chart(fig_hist_dsat)
        st.write("This is the residuals of the traffic Density from the predicted model(Prophet). Basically the reiduals are quite high in the data even after fitting the model.")
        decompose_result_dsat = seasonal_decompose(filtered_df.set_index('Date')['Y_hat_dsat'], model='additive', period=24)
        fig_decompose_dsat = make_subplots(rows=4, cols=1, shared_xaxes=True)
        fig_decompose_dsat.add_trace(go.Scatter(x=decompose_result_dsat.trend.index, y=decompose_result_dsat.trend, name='Trend'), row=1, col=1)
        fig_decompose_dsat.add_trace(go.Scatter(x=decompose_result_dsat.seasonal.index, y=decompose_result_dsat.seasonal, name='Seasonality'), row=2, col=1)
        fig_decompose_dsat.add_trace(go.Scatter(x=decompose_result_dsat.resid.index, y=decompose_result_dsat.resid, name='Residuals'), row=3, col=1)
        fig_decompose_dsat.update_layout(height=400, title='Density Time Series Decomposition')
        st.plotly_chart(fig_decompose_dsat)
        st.write("Observed data is been decomposed into trend,seasonal and residuals")

    st.title("Traffic Flow, Congestion, and Density with Forecast from Postgre database")
    st.write("**Flow Time Series with Forecast:**")
    st.write("It is the movement of vehicles along the roadways. It is a description of the movement of vehicles through a road network. Effective flow means vehicles can move along smoothly and do not experience stops or delays. Some of the key factors that affect traffic flow include road capacity, traffic signals, driver behavior, and external conditions such as weather. Understanding the traffic flow is important in urban planning and the improvement of transportation systems.")
    fig1 = px.line(title='Flow Time Series with Forecast') 
    fig1.add_scatter(x=x_series.index, y=x_series, mode='lines', name='Past Data', line=dict(color='yellow')) 
    fig1.add_scatter(x=y_series.index, y=y_series, mode='lines', name='Predicted Data', line=dict(color='red', dash='dash'))
    st.plotly_chart(fig1)
    st.write("This chart refers to the flow of traffic from Jan 2022 to Oct 2022. The rest all datas are been forecasted through Model training.")

    st.write("**Congestion Time Series with Forecast:**")
    st.write("When demand for road space surpasses its capacity, it leads to congestion. This in turn makes the speed slow, extends trip times, and prolongs the queueing of vehicles. It is very common during peak hours when most people travel to and from work. Various reasons can cause it, such as high volume of traffic flow, construction on roads, accidents, or even events. Management of congestion would involve optimizing signals in traffic, expanding road capacity, and encouraging public transportation.")

    fig2 = px.line(title='Congestion Time Series with Forecast') 
    fig2.add_scatter(x=x_series_cong.index, y=x_series_cong, mode='lines', name='Past Data', line=dict(color='yellow')) 
    fig2.add_scatter(x=y_series_cong.index, y=y_series_cong, mode='lines', name='Predicted Data', line=dict(color='red', dash='dash'))
    st.plotly_chart(fig2)
    st.write("This chart refers to the Congestion of vechile in Ireland from Jan 2022 to Oct 2022. The rest all datas are been forecasted through Model training.")

    st.write("**Density Time Series with Forecast:**")
    st.write("It is basically the number of vehicles occupying a unit length of a roadway. The higher the density, the more vehicles are on the road; if the density exceeds the capacity of the road, it may lead to congestion. Traffic density monitoring helps in the study of road usage patterns and infrastructure improvement planning.")
    fig3 = px.line(title='Densisty Time Series with Forecast') 
    fig3.add_scatter(x=x_series_dsat.index, y=x_series_dsat, mode='lines', name='Past Data', line=dict(color='yellow')) 
    fig3.add_scatter(x=y_series_dsat.index, y=y_series_dsat, mode='lines', name='Predicted Data', line=dict(color='red', dash='dash'))
    st.plotly_chart(fig3)
    st.write("This chart refers to the Density or volume of vechile in a road at Ireland from Jan 2022 to Oct 2022. The rest all datas are been forecasted through Model training.")

def traffic_collision():
    st.title("Traffic Collision At Ireland from 2013 to 2022")
    if not data.empty:
        st.sidebar.header("App Controls")
        statistic_filter = st.sidebar.radio("Select Statistic", options=data['Statistic Label'].unique())
        year_range = st.sidebar.slider("Select Year Range", int(data['Year'].min()), int(data['Year'].max()), (int(data['Year'].min()), int(data['Year'].max())))
        county_filter = st.sidebar.multiselect("Select Counties", options=data['County'].unique(), default=data['County'].unique())

        filtered_data = data[(data['Statistic Label'] == statistic_filter) &
                             (data['Year'] >= year_range[0]) &
                             (data['Year'] <= year_range[1]) &
                             (data['County'].isin(county_filter))]

        st.subheader(f"Filtered Data for Statistic: {statistic_filter}")
        st.write(filtered_data.head())

        # Plotting Function
        def plot_data(data, plot_type, title, x_label, y_label):
            plt.figure(figsize=(14, 8))
            if plot_type == 'Bar':
                sns.barplot(data=data, x='Year', y='VALUE', hue='County')
            elif plot_type == 'Line':
                sns.lineplot(data=data, x='Year', y='VALUE', hue='County', marker='o')
            else:
                sns.scatterplot(data=data, x='Year', y='VALUE', hue='County')

            plt.title(title, fontsize=16)
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(title='County', loc='upper right', bbox_to_anchor=(1.2, 1))
            st.pyplot(plt.gcf())

        # Select plot type
        #plot_type = st.sidebar.radio("Select Plot Type", options=["Bar", "Line", "Scatter"])
        plot_data(filtered_data, "Bar", "Traffic Collision Trends", "Year", "Number of Collisions")

        # ML Model Training
        X = data[['Statistic Label', 'Year', 'County']]
        y = data['VALUE']

        label_enc = LabelEncoder()
        X['Statistic Label'] = label_enc.fit_transform(X['Statistic Label'])
        X = pd.get_dummies(X, columns=['County'], drop_first=True)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/Test Split
        test_size = st.sidebar.slider("Test Size (Ratio)", 0.1, 0.5, 0.2)
        random_state = st.sidebar.number_input("Random State", 1, 100, 42)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
        st.write(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")
        if st.sidebar.button("Train Model"):
            model = RandomForestRegressor(random_state=random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("Model Performance")
            st.write(f"R\u00b2 Score: {r2_score(y_test, y_pred)}")
            st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

            st.subheader("Residuals Distribution")
            residuals = y_test - y_pred
            plt.figure(figsize=(8, 6))
            sns.histplot(residuals, kde=True, color="blue")
            plt.title("Residuals Distribution", fontsize=14)
            plt.xlabel("Residuals", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            st.pyplot(plt.gcf())

            st.subheader("Actual vs Predicted")
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.7)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
            plt.title("Actual vs Predicted Values", fontsize=14)
            plt.xlabel("Actual Values", fontsize=12)
            plt.ylabel("Predicted Values", fontsize=12)
            st.pyplot(plt.gcf())

def settings_page():
    st.title("Air Emission Analysis by Transport Modes")
    st.write("This dashboard presents an in-depth analysis of air emissions caused by various modes of transport using data from the cleaned dataset. The visualizations offer insights into emission trends over time and highlight the contribution of different transport sectors to overall emissions.")

    # Check for statistics
    if st.sidebar.checkbox("Show Statistics"):
        st.subheader("Statistics")
        st.write(data2.describe())

    if 'year' in data2.columns:
        st.subheader("Trends in Emissions Over Time")
        emissions_trend = data2.groupby('year')['value'].sum().reset_index()
        
        fig, ax = plt.subplots()
        ax.plot(emissions_trend['year'], emissions_trend['value'], marker='o', linestyle='-', linewidth=2)
        ax.set_title("Trends in Emissions Over Time", fontsize=16)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Total Emissions", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        st.write("The **\"Trends in Emissions Over Time\"** graph demonstrates the progression of total emissions, providing a year-by-year breakdown. This helps identify whether emissions have increased, decreased, or remained constant over the analyzed period.")



    if 'year' in data2.columns and 'statistic_label' in data2.columns:
        st.subheader("Types of Air Emissions Over the Years")
        emission_data = data2.groupby(['year', 'statistic_label'])['value'].sum().unstack()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        emission_data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        ax.set_title("Types of Air Emissions Over the Years", fontsize=16)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Emission Value", fontsize=14)
        ax.legend(title="Type of Air Emission", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig)
        st.write("The **\"Types of Air Emissions by Transport Modes\"** visualization delves deeper into the data, categorizing emissions by transport types such as road, air, rail, and maritime. It uses stacked bar charts to compare annual contributions across different transport sectors, making it easier to identify key contributors.")


    if 'residence_adjustment_items' in data2.columns:
        st.subheader("Highest Emissions by Residence Type")
        residence_emissions = data2.groupby('residence_adjustment_items')['value'].sum().reset_index()
        residence_emissions = residence_emissions.sort_values(by='value', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(residence_emissions['residence_adjustment_items'], residence_emissions['value'], color='skyblue')
        ax.set_title("Highest Emissions by Residence Type", fontsize=16)
        ax.set_xlabel("Total Emissions (Thousand Tonnes)", fontsize=14)
        ax.set_ylabel("Residence Type", fontsize=14)
        ax.invert_yaxis()
        st.pyplot(fig)
        st.write("Lastly, the **\"Highest Emissions by Transport Mode\"** chart ranks transport modes based on their total emissions, providing a clear picture of which sectors require immediate attention for sustainable development.")

def conclusion():
    st.title("Traffic Flow and Collisions: ")
    st.write("In 2022, traffic flow showed a great improvement on different types of roads. According to the records, traffic congestion has reduced, and the flow of vehicles has become smooth. Similarly, the rate of traffic collisions has also shown a remarkable decline. The reasons for such improvement may include improved traffic management systems, well-developed infrastructure of roads, and awareness among the public for following the rules of the road.")
    st.title("Impact on Environment: ")
    st.write("Another most vital dimension of our analysis concerned environmental impact, particularly about harmful vehicle emissions. Reducing traffic congestion directly relates to lessened vehicle emissions. There are fewer cases of full stops and quick accelerations when there is less traffic jam; thus, it highly minimizes the amount of air contaminants being emitted by vehicles, like carbon monoxide, nitrogen oxides, particulate matter, among others.")
    st.title("Conclusion:")
    st.write("In 2022, improved traffic conditions and environmental health in Irish cities were observed. Due to the combined efforts in traffic management and public compliance, this has not only smoothened the flow of traffic but also contributed significantly towards reducing traffic collisions. More importantly, it has reduced harmful vehicle emissions, thus creating a healthier urban environment. This positive trend will therefore set a precedent for future urban planning and environmental policies in terms of sustaining and further improving these achievements.")

if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'graphs':
    graphs_page()
elif st.session_state.page == 'tc':
    traffic_collision()
elif st.session_state.page == 'settings':
    settings_page()
elif st.session_state.page == 'conclusion':
    conclusion()
else:
    home_page()
