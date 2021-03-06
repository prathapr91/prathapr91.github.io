---
title: EDA Project
---

## **Using NYC MTA turnstile Data to Reduce Carbon Footprint**

For my first project as part of the Metis Data Science Bootcamp, I was tasked with drafting a mock proposal to solve a problem by using the [MTA turnstile dataset](http://web.mta.info/developers/turnstile.html).



**Introduction**

The NYC Department of Transportation and Department of Environmental Protection wish to reduce carbon footprint and congestion of New York City roads. One way to do this would be to reduce the need for car ownership by adding to an already significant subway infrastructure. Currently, [less than half of the city's inhabitants own cars](https://edc.nyc/article/new-yorkers-and-their-cars), but there is still room for growth as some neighborhoods/boroughs are better equipped than others.



The purpose of this exploratory data analysis is to pinpoint neighborhoods that have low ridership per capita. The NYC DOT can use this information to gauge which neighborhoods warrant more investment in subway lines, stations, and tracks. By having more New York City residents utilize the subway system in a convenient manner, more residents will benefit financially by not feeling obligated to purchase a car and the city will benefit from reduced congestion and a smaller carbon footprint.



**Design**

After extracting the MTA Turnstile Dataset, I took ridership levels by station and after cleaning up and filtering the data appropriately, I mapped these stations to zip codes and neighborhoods via Google API and GeoPandas.



Then, I used [population data by neighborhood](https://data.beta.nyc/dataset/pediacities-nyc-neighborhoods) to map to control for differences in neighborhood size and determine a ridership per capita value by neighborhood. I wanted to see which areas are better equipped to handle most of its population going car-less, and which ones have room for improvement. I then looked for neighborhoods that went below a pre-determined threshold (explained in further detail in the Algorithms section) and sought patterns and trends. Then, I built plots and charts to analyze my findings.



**Data**

As mentioned in my design, I will be using the MTA turnstile dataset as my main source of analysis, along with neighborhood population data. Each row contains entry and exit counts, split into 4-hour intervals, broken down by turnstile, time/day, and subway station. Below is a snippet of the MTA turnstile dataset:

![Screen Shot 2022-01-09 at 12.07.00 PM](https://github.com/prathapr91/MTA_EDA/blob/main/Plots/Screen%20Shot%202022-01-09%20at%2012.07.00%20PM.png?raw=true)

The following notes add more color as to the scope of data in use for this project:

*Data Filtering*

1. The intervals used are June/July 2019 and June/July 2021. 2021 was used for the purpose of taking the latest and most relevant trends. Meanwhile, 2019 was used as a sanity check. I wanted to confirm that my findings are credible and not necessarily due to COVID-related measures.
2. Not all entries are created equal. The entries for a given station could potentially be attributed to leaving the house/apartment for work, leaving the office, returning from a restaurant/bar, and tourism, to name a few. This can lead to an apples to oranges comparison between stations, as some neighborhoods can have more of these attributes than others. To control for this, and because the goal of this analysis is to ultimately have more residents using the subway and less reliant on owning cars, I will be looking at subway entries on weekdays before noon. By normalizing this way, the entries in my analysis will be most likely due to morning commuters, and less likely due to other external factors, for all neighborhoods.

*Data Cleaning*

1. The `ENTRIES` column provided is cumulative, so I created a new column, `NEW_ENTRIES`, that takes the difference between entry counts at incremental time intervals.
2. Outliers and faulty data are corrected and removed.
3. To get to the end metric of total subway entries per station, I will be summing up entries from all turnstiles for a single station. The following fields will be grouped: `C/A, UNIT, SCP`. Then, the `NEW_ENTRIES` gets summed up for total entries for the mentioned time intervals.

*Data Mapping*

1. Google API is used to map stations to zip codes and coordinates.
2. Population data is used to map zip codes of stations to neighborhoods and neighborhood populations.



**Algorithms**

1. Taking the total entries for each subway station for a given time interval
2. Mapping each subway station to a neighborhood, and dividing by neighborhood population to determine neighborhood ridership per capita, and finding neighborhoods that fail to meet threshold
3. The threshold is 4.94 entries per capita post-COVID and 12.36 pre-COVID. The calculation for the threshold is the product of the following:
   1. We ideally want NYC residents to use the subway instead of a car to commute to work daily, so we will start with the ideal threshold of 40, which is one subway entry per day for two months worth of business days.
   2. [77% adult population](https://www.baruch.cuny.edu/nycdata/population-geography/age_distribution.htm) that would commute to work
   3. [55%](https://edc.nyc/article/new-yorkers-and-their-cars) of NYC residents do not own a car. However, this varies by neighborhood so ideally, we would like to get to the 55% threshold for neighborhoods that fail to meet it.
   4. Of those car owners, [73%](https://edc.nyc/article/new-yorkers-and-their-cars) use their car for commuting, as opposed to keeping it for miscellaneous purposes such as weekend trips.
   5. For 2021 only, a COVID adjustment of 40%. I am assuming that in a post-pandemic world, hybrid or fully remote work arrangements are common.



**Results**

Taking Manhattan as an example, this chart represents the ridership per capita, broken down by neighborhood pre and post COVID. The purpose of this is to demonstrate that the pandemic did not significantly alter trends. Neighborhoods such as Lower Manhattan, Chelsea, and Clinton have the most subway usage while Central Harlem could use more investment.



![Manhattan2019](https://github.com/prathapr91/MTA_EDA/blob/main/Plots/Manhattan2019.png?raw=true)

![Manhattan2021](https://github.com/prathapr91/MTA_EDA/blob/main/Plots/Manhattan2021.png?raw=true)



Here, we can see the ridership per capita for each neighborhood vs its respective population. While Manhattan is home to neighborhoods with high ridership, as expected, the majority of neighborhoods that require attention are in Queens. In addition, this phenomenon occurs in a variety of Queens neighborhoods, regardless of size.

![ScatterAll](https://github.com/prathapr91/MTA_EDA/blob/main/Plots/ScatterAll.png?raw=true)



If we take a closer look to see where these neighborhoods are, we can see a clear pattern emerging in Queens. Each dot represents a neighborhood that misses the 4.94 threshold and the larger the dot, the smaller the ridership. This reveals a pattern where not only a large volume of neighborhoods that fail to meet the threshold are in Queens, but also neighborhoods that miss the threshold significantly.

![NYC_mapplot](https://github.com/prathapr91/MTA_EDA/blob/main/Plots/NYC_mapplot.png?raw=true)

In total, I identified 20 neighborhoods in NYC that warrant additional investment in subway infrastructure. Most of these neighborhoods are in Queens and I recommend starting in this borough. By targeting these neighborhoods, I believe that more residents will feel encouraged to use the subway and be less reliant on cars.



To enhance this project in the future, I would utilize cab data to identify neighborhoods with high cab traffic and low ridership and take a closer look at subway lines (for example, if one train line runs infrequently or if neighborhoods don???t have access to many lines).



To see my project in further detail, please visit my [GitHub Repo](https://github.com/prathapr91/MTA_EDA).



**Tools Used**

- Pandas for data manipulation
- SQL Alchemy for importing MTA data
- Google API, GeoPandas, and GeoPy for location mapping
- Matplotlib, Seaborn, and Plotly for visualization
