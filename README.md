# About rooftop sampling

The "gold standard" approach to sampling households for households surveys involves first sampling primary sampling units (PSUs), then conducting a census in each sampled PSU, and finally randomly sampling households from the census of households. The gold standard approach ensures unbiased estimates but, due to the need to conduct a census of all households in each sampled PSU, is very expensive and, in some cases, infeasible. Alternative sampling strategies like the WHO's "spin-the-pen" method and the "right-hand rule" eliminate the need to conduct a census of households in each sampled PSU but also lead to (sometimes badly) biased results (Lemeshow et al, 1985). Rooftop sampling is a new approach to household sampling which eliminates the need for a costly household census in each PSU and which leads to approximately unbiased results. With rooftop sampling, we sample households in each PSU by first sampling rooftops using the [Google-Microsoft Open Buildings dataset](https://source.coop/repositories/vida/google-microsoft-open-buildings/description) and then surveying all households that live near the selected rooftops. In a companion article ([Johnson et al, 2025](https://docs.google.com/document/d/1lKvt3flCb-Qssij8FcgPfv4TOGXF4DGW9dHRMvQxs8Y/edit?tab=t.210brtqp598y)) we show that rooftop sampling leads to very low bias. 

# Getting started with this repo

To use this code, you must have python 3.11 and poetry installed. To get started with this code, clone this repo and then install packages to a fresh poetry environment using "poetry install." 

To get the nearest point on the road, you will need to get a google maps API key, able the Road API for the given API key, and save the API key in a .env file with a line that looks something like "GOOGLE_MAPS_API_KEY=XXXXXX". If you are unsure how to create a google maps API key and enable the Road API, I found asking chatgpt how to do this very helpful. Please make sure that you don't accidentally add the .env file to your repo. 

# Overview of rooftop sampling steps

The typical workflow for implementing rooftop sampling in a new country involves 5 main steps, 3 steps to prepare the data and 2 steps to perform the sampling. We provide a basic overview of each step and approximately how long it takes below.

## Data preparation

1. **Obtain or create a dataset of PSUs with borders and population totals [3 days]** - In order to conduct rooftop sampling, we first need a dataset of PSUs with data on borders and population totals. Obtaining or creating this dataset is typically the most time intensive part of rooftop sampling since it often involves a lot of tedious data cleaning. Fortunately, this is a one time step for each country and we have already compiled cleaned datasets for two countries (India and the Philippines).
2. **Download rooftop data**  - Rooftop data can be downloaded from the [Google-Microsoft Open Buildings dataset](https://source.coop/repositories/vida/google-microsoft-open-buildings/description). In most cases, it makes sense to download all of the rooftop data for a country in one go. For a few of the most populated countries (India and Indonesia in particular) the rooftop data is too large to download at once (without waiting hours). *
3. **Merge PSU information into rooftop data** - Next, we merge the rooftop and PSU datasets using a "spatial join" so that each rooftop is assigned a single PSU.  

## Sampling

4. **Sample PSUs** - Using the PSU dataset, we randomly sample a set of PSUs using probability proportional to sample size (PPS) sampling. This step should be completed in Stata or R since, to our knowledge, python lacks a package to accurately perform PPS sampling.
5. **Sample rooftops and save outputs** - Using the sampled PSUs, we randomly select a fixed number of rooftops using the rooftop dataset with PSU information created in step 3 of data preparation. 
6. **Create Google myMaps map using outputs** - Unfortunately, there is still a tiny bit of manual processing to create the Google myMaps map that should be shared with surveyors. We describe the steps to do this below.

# 1. Obtain or create dataset of PSUs with borders and population totals

In order to conduct rooftop sampling, we need a dataset of primary sampling units for the country or area that we want to conduct sampling in along with data on the borders and population totals for those PSUs. (In theory, we could conduct rooftop sampling without having population totals by selecting PSUs using simple random sampling and then selecting a fixed proportion, rather than a fixed number, of rooftops. We haven't tested if this works though and if the ratio of people to rooftops varies by PSU this could lead to biased estimates.) As with the typical gold standard approach, the set of PSUs should be mutually exclusive and collectively exhaustive for the area we seek to perform sampling for. Fortunately, unlike with the gold standard approach, PSUs don't necessarily have to be small enough for surveyors to be able to perform a household census in.

Data on PSU population totals can usually be obtained from the most recent census.

Obtaining data on PSU borders is typically more challenging. The [humanitarian data exchange website](https://data.humdata.org/) has a lot of great data on sub-national borders and is a great place to start but in many cases you will have to search further. Regardless of where you source the border data from, we recommend performing the following checks to make sure that the data looks good:

1. **Merging the border data and population data and checking for unmerged rows -** In most cases, you will have to merge PSU border data with PSU population data. When merging, it is important to check which rows from each dataset weren't able to merge successfully. Often, merge errors happen because the border data and populatin data were created at different times and some PSUs were merge, split, or renamed between the dataset creation dates. Inspecting the merge errors can help pinpoint and resolve these errors. One useful technique for inspecting the merge errors is to create a map showing the rows from the borders dataset that couldn't be merged. 
2. **Using local experts to verify the accuracy of a sample of PSU borders** - We have found it useful to have local experts inspect a sample of PSU borders. To identify local experts, we asked survey staff which areas they were relatively familiar with.  

We have gone through this process (or started the process) for three countries: the Philippines, India, and Kenya. We provide notes on our data sources below:



| Country      | Border data source                                                                                                                       | Population data source                                                                                                                                            | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Philippines** | [PSGC shapefiles](https://github.com/altcoder/philippines-psgc-shapefiles/tree/main)                                                                                           | Barangay data file with PCWHS designation provided by Gio, who obtained this from the census website                                                                | <ul><li>HDX also has data on barangay borders [here](https://data.humdata.org/dataset/cod-ab-phl/resource/12457689-6a86-4474-8032-5ca9464d38a8), but we got more merge errors using that.</li><li>The notebook <code>clean_barangay_data.ipynb</code> merges the border and population data sources. Previously, there were merge errors for barangays in Negros Island after the region split (PSGC codes changed), but those errors are gone with the updated population file that Gio shared.</li><li>Rooftop data has been merged with barangay info (files ending in <code>w_brgys</code>). See <code>add_brgy_to_rooftop.ipynb</code> for the merge code.</li></ul> |
| **India**       | (Probably) Mapsolve's proprietary dataset of village and ward boundaries                                                              | Unclear                                                                                                                                                                                                    | <ul><li>IDinsight field managers inspected a sample of boundaries they knew well. SHRUG data was deemed very low quality, while Mapsolve data was much better. Also, SHRUG data is missing ward boundaries for many cities in India (e.g. Cuttack, Kolkata).</li></ul>                                                                                                                                                                                                                                                                                                                                         |
| **Kenya**       | HDX has data on [sublocation boundaries](https://data.humdata.org/dataset/cod-ab-ken)                                                                                       | HDX has [sublocation population totals](https://data.humdata.org/dataset/kenya-population-households-and-density-by-sublocations-2009) from 2009.<br>The [2019 census data](https://housingfinanceafrica.org/app/uploads/VOLUME-II-KPHC-2019.pdf#page=31.34) (see Table 2.4) is in PDF format | <ul><li>The population dataset doesn’t include sublocation IDs, and there are many duplicate sublocation names, making merges tedious.</li><li>A potential approach to merging is to incorporate location and province boundaries first, then merge population data by province, location, and sublocation names—likely a very manual process to resolve name mismatches across datasets.</li></ul>                                                                                                                                                                                                  |



# 2. Obtain rooftop data
The Google-Microsoft Open Buildings dataset is available at [Source Cooperative](https://beta.source.coop/repositories/vida/google-microsoft-open-buildings/description/).

To access the data you will first have to register on the website. Once you have registered, there are two ways to access the data: by manually downloading individual files from the Source Cooperative website and via the command line using aws command line tools (or programmatically using something like boto3). For most countries, the data is small enough that it makes sense to download all of the files manually in one go. For a couple of countries (India and Indonesia), it might make sense (depending on the workflow) to programmatically only download the data you need. 

To manually download individual files, login then click "browse" on the left hand nav bar and browse to the file that you want.

You can download all of the rooftop data for a country in a single file or download data by [S2 cell](http://s2geometry.io/devguide/s2cell_hierarchy.html). We recommend downloading the data by S2 cell since the code for merging PSU data with rooftop data runs very slowly on very large files. For many countries, it doesn't really matter since there is only one S2 cell file for the entire country.

We also warn users that the S2 "level" varies by country. For example, the S2 level for India is 6 and the S2 level for the Philippines is 4. 

The utils module includes a helper function called get_s2_cell_id which returns the S2 cell ID for a given point. 


# 3. Merge PSU information into rooftop data
Once you have a cleaned PSU dataset and the rooftop data, we recommend combining these two datasets to create a dataset of rooftops with PSU ID attached. After merging the two datasets, it is useful to check the number of rooftops that weren't assigned a PSU. The notebook add_brgy_to_rooftop.ipynb does this for Philippines data. For the Philippines, it took a while (5 minutes or so) to run this file but it made further sampling much easier.

Alternatively, you could first sample PSUs and then grab rooftop data for just the sampled PSUs. In practice, we have found that merging PSU and rooftop data for a small subset of sampled PSUs is not that much faster than performing the merge for the full set of PSUs. And merging all of the rooftops with the PSUs in one go makes it a lot easier to perform sampling in the future and allows you to double check that there are no (or few) rooftops that are not assigned PSUs.


# 4. Sample PSUs
PSUs should be sampled with probability proportional to size (PPS) sampling using either Stata or R. Pandas' Dataframe.sample method does not perform accurate PPS sampling without replacement. (I tested this method's functionality by trying to perform PPS sampling without replacement on a dataframe where it is impossible to perform PPS without replacement. If you do this in R or Stata, the functions will error out but pandas' sample method does not.)

To perform PPS sampling without replacement in Stata, use either samplepps or gsample. Code to sample 10 PSUs using PPS without replacement sampling using a PSU-level dataset with a column psu_pop for PSU population is provided below. Note that the entire dataset is retained and the new variable "selected" indicates whether or not the PSU was sampled.

```
samplepps selected, n(10) size(psu_pop)
gsample 10 [weight=psu_pop], wor generate(selected)
```

# 5. Sample rooftops and save outputs
The final step is to sample a fixed number of rooftops per PSU and save the outputs. The notebook demo_tfgh.ipynb demonstrates how to do this. The core logic of this notebook is pretty straightforward -- i.e. you merge the sampled PSUs with the rooftop data to generate a dataframe of rooftops in the sampled PSUs and then sample a fixed number of rooftops per sampled PSU. There are a few quirks to be aware of though.

1. In some cases, it is useful to remove very isolated rooftops before sampling. We find that most "rooftops" in the Open Buildings dataset correspond to an actual rooftop and most of these buildings are inhabited. Still, there are many false positives either because no building exists or the building is not inhabited. This is not a big deal in most areas since surveyors instead just survey inhabitants of the nearest building but can be a big deal if there are no other buildings for miles. If that is the case, surveyors have just spent a lot of time traveling for nothing. The function utils.count_neighbors_in_radius helps identify isolated rooftops.
2. In cases where the sampled rooftop is reasonably close to a road, we recommend using the nearest point on a road rather than the actual sampled rooftop. In field tests, we found that surveyors were much more likely to consistently identify the same structures if they were given a point on a road and instructed to find the nearest structure to the point and the structures to the left and right of that structure compared to just being given a point and instructed to find the nearest structures. The function utils.get_nearest_point_on_road uses the Google Maps API. We have found that the Google Maps API is quite accurate and also good for determining whether or not a sampled rooftop is reasonably close to a road. (If the rooftop is not reasonably close to a road it will return null.)

# 6. Create Google myMaps map using outputs
Unfortunately, there is still a bit of manual processing required to create the final map to share with surveyors from the notebook outputs. The notebook will save three kml files as output:

1. **points_on_road.kml** - This kml file contains a single point on a road for each sampled rooftop found to be near a road.
2. **points_on_road_lines.kml** - This kml file contains a set of lines between points on a road and the original sampled rooftop for each rooftop near a road. These lines help surveyors determine which side of the road to look for structures.
3. **points_off_road.kml** - This kml file contains all the rooftop points not near a road.

To create a single Google myMaps from these files:

1. Login to [Google My Maps](https://www.google.com/mymaps)
2. Click "create new map"
3. For each of the three kml files:
    a. Click "add layer"
    b. Within the layer spec, click "import" and upload the kml file
4. [Optional but recommended] Set the color of points_on_road and points_on_road_lines to be the same and the points_off_road to be a different color. To change the color of all elements of a layer click where it says "individual styles" in the layer spec and select "uniform style", hit escape, and click on the paint icon and select a color.

Once you have created the map, you can share with surveyors by clicking the "share" button. 