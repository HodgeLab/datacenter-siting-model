import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Get county data centroid points
county_centroid = pd.read_csv("/mnt/c/Users/maria/OneDrive/Desktop/research/dataproject/reeds/County_Centroids.csv")
county_centroid['county'] = county_centroid['county'].str.replace(' County', '', regex=False)
#print(county_centroid.tail())

county_zone = pd.read_csv("CountyMaps/county2zone.csv")
county_zone = county_zone.rename(columns={'FIPS':'cfips', 'county_name':'county'})
#print(county_zone.tail())

# merge on same county FIPS codes
county_data = pd.merge(county_zone, county_centroid, on = ["cfips"], how = 'left')
county_data = county_data.drop(columns=['state_y', 'county_y'])
county_data = county_data.rename(columns={'county_x':'county', 'state_x':'state'})

# Add wkt and / or geometry 
county_data['wkt'] = county_data.apply(lambda row: f"POINT ({row['longitude']} {row['latitude']})", axis=1)
county_data['geometry'] = county_data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
county_data= gpd.GeoDataFrame(county_data, geometry='geometry', crs="EPSG:4326")
county_data = county_data.to_crs("ESRI:102005") # better for north america and distance calculations allegedly

print(county_data.tail())
county_data.to_csv("CountyMaps/county_data.csv", index=False)

# Plot the county centroids
county_data.plot(marker='o', color='black', markersize=10, figsize=(8, 6))
plt.show()