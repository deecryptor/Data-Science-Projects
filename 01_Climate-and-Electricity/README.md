# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 1: Exploring Climate Data of Singapore

## Executive Summary

Situated near the Equator, Singapore is a peninsular island with typical tropical climate - abundant rainfall, high and uniform temperature and high humidity levels. Such weather conditions are expected to have an impact on how people in Singapore choose their activities for the day as well as how they would use electricity based on the activities chosen. This project aims to provide a data-driven analysis to understand if there is a correlation among the aforementioned factors. First, data on rainfall and other weather features as well as household electricity consumption are gathered for this project. The data is then imported and cleaned up using Pandas library primarily to ensure EDA can be carried out properly. Based on findings derived from EDA and Data Visualization steps, it is concurred that rainfall and other weather conditions do not have a strong correlation to household electricity consumption in Singapore. One key possible reason is that there are many factors affecting electricity consumption and weather could be just one of the many drivers. Finally, it is suggested that data on commercial electricity consumption can be gathered for future improvement in the analysis of this project.

## Problem Statement

Rain is plentiful in Singapore and it rains an [average of 167 days per year](http://www.weather.gov.sg/climate-climate-of-singapore/). Unlike countries in the temperate regions, temperatures in Singapore are high and relatively uniform with little variance from month to month and from day to day. Being a peninsular island surrounded by seas, Singapore is rather humid, which will influence how our bodies sense and experience warmth. These weather conditions are expected to have an impact on how people in Singapore choose their activities for the day as well as how they would use electricity based on the activities chosen. This project aims to provide a data-driven analysis to understand if there is a correlation among the aforementioned factors. If a correlation between weather and electricity consumption can be identified, relevant national agencies working on climate change issues can use these findings to plan more effective carbon emission reduction strategy.

### Datasets

The datasets that will be used in this analysis are:

* [`weather.csv`](data/weather.csv) where part of the data has been provided in the project data folder and the remaining downloaded from [singstat.gov.sg](https://tablebuilder.singstat.gov.sg/table/TS/M890081):
    - Monthly total rain recorded in mm(millimeters) from 1982 to 2022
    - Monthly number of rain days from 1982 to 2022. A day is considered to have “rained” if the total rainfall for that day is 0.2mm or more.
    - Monthly mean daily maximum air temperature from 1982 to 2022
    - Monthly mean daily minimum air temperature from 1982 to 2022
    - Monthly mean daily bright sunshine hours from 1982 to 2022
    - Monthly mean 24 hours humidity from 1982 to 2022

* [`electricity.csv`](data/electricity.csv) which came from [ema.gov.sg](https://www.ema.gov.sg/statistic.aspx?sta_sid=20140729cKpjlPbDaOxa):
    - Average Monthly Household Electricity Consumption from 2005 to 1H 2022

## Data Import and Cleaning

The following libraries are being imported for use in this project:
  - Pandas
  - Numpy
  - Matplotlib.pyplot
  - Seaborn

The two datasets are imported and the following data cleaning steps are carried out:
  - Check for missing values
  - Check for datatypes to find discrepancies
  - Convert date values to datetime and create new year column for ease of analysis
  - Check for duplicated and unique values
  - Merge the two datasets into one and drop incomplete values

### Data Dictionary

A summary of data features is as follows:

|Feature|Type|Dataset|Description|
|---|---|---|---|
|total_rainfall|float|weather|Total rainfall in that month (mm)|
|number_of_rainy_days|integer|weather|The number of rainy days in that month|
|air_temperature_means_daily_max|float|weather|Average maximum daily temperature in that month (Degree celcius)|
|air_temperature_means_daily_min|float|weather|Average miniimum daily temperature in that month (Degree celcius)|
|bright_sunshine_daily_mean_hour|float|weather|Average daily sunshine hours in that month|
|24_hours_mean_relative_humidity_percent|float|weather|Average 24 hours relative humidity in that month (%)|
|overall|float|electricity|Average monthly household electricity consumption in Singapore (kWh)|
|central_region|float|electricity|Average monthly household electricity consumption in central region of Singapore (kWh)|
|east_region|float|electricity|Average monthly household electricity consumption in East region of Singapore (kWh)|
|northeast_region|float|electricity|Average monthly household electricity consumption in Northeast region of Singapore (kWh)|
|north_region|float|electricity|Average monthly household electricity consumption in North region of Singapore (kWh)|
|west_region|float|electricity|Average monthly household electricity consumption in West region of Singapore (kWh)|

## Exploratory Data Analysis (EDA)

Investigate trends in the data and provide observations based on given questions:
  - Which month have the highest and lowest total rainfall in 2005, 2010, 2015 and 2020?
  - Which year have the highest and lowest total rainfall in the date range of analysis?
  - Which month have the highest and lowest number of rainy days in 2005, 2010, 2015 and 2020?
  - Which year have the highest and lowest number of rainy days in the date range of analysis?
  - Are there any outliers months in the dataset?

## Data Visualization

The following charts are plotted to better understand the data trends and correlations:
  - Histogram
  - Line plots
  - Heatmap
  - Scatterplots

## Conclusion and Recommendations

Contrary to what was expected, weather conditions **do not appear to be strongly correlated** to household electricity consumption. There are a couple of possible reasons for this:

* Daytime is typically warmer but many people are unlikely to be at home during this timing especially on weekdays, and hence there is lower demand to switch on the biggest energy guzzler -- air-conditioners.

* Electricity consumption can be complex and will differ based on the types electrical appliances and energy efficiency of the appliances, etc. The amount of electricity consumed by each type of usage differs and not all usages are directly driven by weather patterns. For instance, refrigerator has to be switched on regardless rain or shine.

As a result, national government agencies may not be able to make meaningful use of these inconclusive findings above.

One **recommendation for further improvement** is to gather data on commercial electricity consumption for more comprehensive analysis. Electricity consumption of buildings is understandably higher than that of households due to the much larger space area. As reported by [Channel News Asia](https://www.channelnewsasia.com/commentary/air-con-unit-electricity-energy-carbon-emissions-climate-change-1339326), cooling systems consume about 40 to 50 per cent of a building’s energy, which contributes to the urban heat island effect. Hence, the likelihood of finding more significant correlation between commercial electricity consumption and weather conditions will be probably higher.
