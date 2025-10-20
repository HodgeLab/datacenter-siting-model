import numpy as np # need less than version 2 of numpy for pyomo so numpy-1.26.4
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from geopy.distance import geodesic
from cost_dict import *
import matplotlib.pyplot as plt

'''


def make_file():
    supply_data = make_supply()

    if __name__ == '__main__':
        supply_data = make_supply()
        print(supply_data['capacity_solar'].describe())
        supply_data.to_csv("supply_data.csv", index=False)
        supply_data.to_csv(str(file_path / 'supply_data.csv'), index=False)
    else:
        return supply_data
    
'''
#def make_supply():
# Loading in data

solar_supply_curve = pd.read_csv("open_access_2030_moderate_supply_curve.csv")
wind_supply_curve = pd.read_csv("open_access_2030_moderate_115hh_170rd_supply_curve.csv")
#geo_45_supply_curve = pd.read_csv("egs_4500m_supply-curve.csv") # 4500, also 5500 and 6500m
#geo_55_supply_curve = pd.read_csv("https://gdr.openei.org/files/1732/egs_5500m_supply-curve.csv")
#geo_65_supply_curve = pd.read_csv("https://gdr.openei.org/files/1732/egs_6500m_supply-curve.csv")



#nuclear_data = tifffile.imread("/mnt/c/Users/maria/githubdc/nuclear_data.tif") # makes it a numpy array
#print(f"Image shape: {nuclear_data.shape}")

'''
#def make_nuclear()
output_rows = []
with rasterio.open('/mnt/c/Users/maria/githubdc/nuclear_data.tif') as src:
    # Check ouut data
    #print("Number of bands:", src.count)
    #print("CRS:", src.crs)
    #print("Transform:", src.transform)

    band = src.read(1)  # Read the one band
    transform = src.transform
    nodata_value = src.nodata  # nodata values

    # Process row by row so it works
    for row in range(band.shape[0]):
        for col in range(band.shape[1]):
            value = band[row, col]
            
            if nodata_value is not None and (value == nodata_value): # Where there is no data
                continue
            
            x, y = rasterio.transform.xy(transform, row, col)
            output_rows.append((y, x, value))

# Convert to DataFrame
df = pd.DataFrame(output_rows, columns=['latitude', 'longitude', 'value'])
#df.to_csv('nuclear_data.csv', index=False)

print(df.head())
#return df
'''

# Transmission
'''
transmission_data = gpd.read_file("/mnt/c/Users/maria/githubdc/Electric__Power_Transmission_Lines.geojson")
state_data = gpd.read_file("/mnt/c/Users/maria/githubdc/cb_2022_us_state_20m/cb_2022_us_state_20m.shp")

transmission_data = transmission_data.to_crs(epsg=3857) # pretty standard (metric)

#print(transmission_data.columns)
#print(transmission_data.head())
'''
'''
# Combine solar and wind data into one dataset
solar_supply_curve = solar_supply_curve[['latitude', 'longitude', 'county', 'sc_point_gid', 'capacity_mw_ac', 'mean_cf_ac', 'dist_km', 'reg_mult', 'state']]
wind_supply_curve = wind_supply_curve[['latitude', 'longitude', 'county', 'sc_point_gid', 'capacity_mw', 'mean_cf', 'dist_km', 'reg_mult', 'state' ]]

def make_geo(filepath):
    geo_supply_curve = pd.read_csv(filepath)
    geo_supply_curve = geo_supply_curve[['latitude', 'longitude', 'county', 'sc_point_gid', 'capacity_ac_mw', 'capacity_factor_ac', 'dist_spur_km', 'state']]
    geo_supply_curve = geo_supply_curve.rename(columns={"capacity_ac_mw": 'capacity_geo', 'capacity_factor_ac': "cf_geo", 'dist_spur_km' : 'dist_km'})
    return geo_supply_curve

geo_45_supply_curve = make_geo("egs_4500m_supply-curve.csv")

solar_supply_curve = solar_supply_curve.rename(columns={'capacity_mw_ac': 'capacity_solar', 'mean_cf_ac': 'cf_solar'})
wind_supply_curve = wind_supply_curve.rename(columns={'capacity_mw': 'capacity_wind', 'mean_cf': 'cf_wind'})

# Left merge to prioritize solar data, keep both countydata to then prioritize solar county
#supply_data = pd.merge(solar_supply_curve, wind_supply_curve, on=['latitude', 'longitude'], how='outer', suffixes=('_solar', '_wind'))
#supply_data = pd.merge(supply_data, geo_supply_curve, on=['latitude', 'longitude'], how = 'outer', suffixes=('', '_geo'))

supply_data = pd.merge(solar_supply_curve, wind_supply_curve, on=['sc_point_gid'], how='outer', suffixes=('_solar', '_wind'))#.fillna(0)
print(supply_data.head())
supply_data = pd.merge(supply_data, geo_45_supply_curve, on=['sc_point_gid'], how = 'outer') #, suffixes=('', '_geo'))
print(supply_data.head())

supply_data['county'] = supply_data['county_solar'].fillna(supply_data['county_wind']).fillna(supply_data['county']) # fills county with county wind if no county solar
supply_data['trans_dist'] = supply_data['dist_km_solar'].fillna(supply_data['dist_km_wind']).fillna(supply_data['dist_km']) 
supply_data['latitude'] = supply_data['latitude_solar'].fillna(supply_data['latitude_wind']).fillna(supply_data['latitude']) 
supply_data['longitude'] = supply_data['longitude_solar'].fillna(supply_data['longitude_wind']).fillna(supply_data['longitude']) 
supply_data['state'] = supply_data['state_solar'].fillna(supply_data['state_wind']).fillna(supply_data['state']) 
supply_data['reg_mult'] = supply_data['reg_mult_solar'].fillna(supply_data['reg_mult_wind']) 


print(supply_data.head())
supply_data['trans_dist'] = supply_data['trans_dist']/1.609 # convert to miles 

supply_data = supply_data.fillna(0)
print(supply_data.columns)
pr
#supply_data = supply_data[:2000]
#supply_data = supply_data.sample(n=2000, replace=False) #no duplicate rows

min_lon = -80 #-130
max_lon = -50
min_lat = 20
max_lat = 30 #50
#supply_data = supply_data[(supply_data['latitude'] >= min_lat) & (supply_data['latitude'] <= max_lat) &
#                 (supply_data['longitude'] >= min_lon) & (supply_data['longitude'] <= max_lon)]

'''
'''
####### Calculate closest distance from lat/lon in supply dat to existing transmission
print(transmission_data.crs) # EPSG: 3857

supply_data["transmission_distance"] = supply_data.apply(
    lambda row: transmission_distance(row["latitude"], row["longitude"], transmission_data),
    axis=1
)

def transmission_distance(lat, lon, transmission_data, crs="EPSG:4326"): #supply dat is lat/long so its the 4326 for degrees but converted to crs of transmission data
    # Create shapely Point, reproject to match geo df
    line_geometries = transmission_data.geometry
    point = gpd.GeoSeries([Point(lon, lat)], crs=crs).to_crs(transmission_data.crs).iloc[0]
    # Return the shortest distance & convert to miles
    distance = line_geometries.distance(point).min()/1609.34
    distance = f"{distance:.7f}"
    return distance

'''
# Telecom data distance
'''
telecom_node_data = pd.read_csv("/Users/maria/Documents/Research/deloitte-proj/telecom-data/telecom_node_data.csv")

supply_data = supply_data.dropna(subset=['latitude', 'longitude'])
telecom_node_data = telecom_node_data.dropna(subset=['lat', 'lng'])

# Add tuple points for geodesic (with shape of earth not just euclidean) distance
supply_data['point'] = list(zip(supply_data.latitude, supply_data.longitude))
telecom_node_data['point'] = list(zip(telecom_node_data.lat, telecom_node_data.lng))

# For each point in supply data, find the closest point of telecom nodes and the distance
def find_min_distance(point1):
    return min(geodesic(point1, point2).kilometers for point2 in telecom_node_data['point'])

# Apply to supplydata
supply_data['telecom_dist'] = supply_data['point'].apply(find_min_distance) #/ 1.609
supply_data = supply_data.drop(columns='point')

print(supply_data.head())
supply_data.to_csv("/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data_lat_lon.csv", index=False)
'''
'''
supply_data = pd.read_csv("/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data_lat_lon.csv")
supply_data['county'] = supply_data['county'].str.replace(r'\bsaint\b', 'st.', case=False, regex=True)
supply_data = supply_data.drop(columns=['state_id', 'FIPS','ba'])
'''
'''
# Make dataset with lat/lon
supply_data = supply_data.drop(columns=['latitude_solar','latitude_wind', 'longitude_solar', 'longitude_wind', 'county_solar', 'county_wind', 'dist_km_solar', 'dist_km_wind', 'dist_km', 'state_solar','state_wind' ])

#supply_data = supply_data.dropna(subset=["capacity_solar", "cf_solar", "capacity_wind",  "cf_wind"], inplace=True)
supply_data = supply_data.fillna(0) # modify the dataset, doesn't make a new one
supply_data = supply_data.rename(columns={'sc_point_gid': 'Locations'})

# Make integer for easier multipliction
supply_data[['Locations']] = supply_data[['Locations']].apply(pd.to_numeric, downcast='integer')
supply_data[['cf_solar', 'cf_wind', 'cf_geo']] = supply_data[['cf_solar', 'cf_wind', 'cf_geo']].apply(pd.to_numeric, downcast='float')
#supply_data.to_csv("/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data.csv", index=False)


# if all data 0 drop the columns
cols_to_check = ['capacity_solar', 'cf_solar', 'capacity_wind', 'cf_wind', 'capacity_geo', 'cf_geo']
supply_data = supply_data[~(supply_data[cols_to_check] == 0).all(axis=1)]
'''
'''
# Add county FIPS code 
county2zone = pd.read_csv("CountyMaps/county2zone.csv") # has no saint's
county2zone = county2zone.rename(columns={'state': 'state_id', 'county_name':'county'})
supply_data['county'] = supply_data['county'].str.lower()

cities = pd.read_csv("/Users/maria/Documents/Research/deloitte-proj/deloitte-data/simplemaps_uscities_basicv1/uscities.csv")
cities = cities[['state_name', 'state_id']].drop_duplicates(subset='state_name')
cities = cities.rename(columns={'state_name': 'state'})

# Change counties to st. so they can merge
#cities['county'] = cities['county'].str.replace(r'^st\.', 'saint', case=False, regex=True)
#cities['county'] = cities['county'].str.replace(r'\bsaint\b', 'st.', case=False, regex=True)

supply_data = pd.merge(supply_data, cities, on=['state'], how='left')

supply_data = pd.merge(supply_data, county2zone, on=['county', 'state_id'], how='left')
#supply_data = supply_data.drop(columns=["county_name", "ba"])
#supply_data = supply_data.rename(columns={'state_x': 'state', 'state_y':'state_ab'})

# Make FIPS integer
supply_data['FIPS']= supply_data['FIPS'].astype('Int64')

supply_data.to_csv("/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data_lat_lon.csv", index=False)
supply_data = supply_data.drop(columns=["latitude", "longitude", "state", 'state_id'])

# Make everything 2 decimal point
supply_data = supply_data.round(2)

print(supply_data.head())
print(supply_data[supply_data['county']== 'st. lucie'])
supply_data.to_csv("/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data.csv", index=False)
#print(supply_data.tail())
    #return supply_data
'''
'''
import pandas as pd
import numpy as np
# Make fake load data
load_df = pd.DataFrame({
    'hour': range(0, 8760),
    'load': np.round(np.random.uniform(0, 200, size=8760), 2)
})

load_df.to_csv("fake_demand.csv", index=False)
'''

'''
supply_data_lat_lon = pd.read_csv("/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data_lat_lon.csv")
#supply_data = pd.read_csv("/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data.csv")
city_price_data = pd.read_csv("/Users/maria/Documents/Research/deloitte-proj/deloitte-data/water_prices_loc.csv")
city_price_data = city_price_data.rename(columns={'lat':'latitude', 'lng':'longitude'})

# Match water price to closest city in same state or region if not
def set_water_price(row):
    loc_coordinates = (row['latitude'], row['longitude'])
    loc_state = row['state_id']
    loc_region = water_price_region_dict.get(loc_state)

    # Try to match city to same state
    same_state = city_price_data[city_price_data['state'] == loc_state]
    if not same_state.empty:
        distances = same_state.apply(
           lambda r: geodesic(loc_coordinates, (r['latitude'], r['longitude'])).miles,
            axis=1
        ) 

        min_idx = distances.idxmin()
        if min_idx in same_state.index:
            return same_state.loc[min_idx, 'price']
    
    # Step 2: Match cities in same region
    region_states = [s for s, r in water_price_region_dict.items() if r == loc_region]
    same_region = city_price_data[city_price_data['state'].isin(region_states)]
    if not same_region.empty:
        distances = same_region.apply(
            lambda r: geodesic(loc_coordinates, (r['latitude'], r['longitude'])).miles,
            axis=1
        )

        min_idx = distances.idxmin()
        if min_idx in same_region.index:
            return same_region.loc[min_idx, 'price']
    
supply_data_lat_lon['water_price'] = supply_data_lat_lon.apply(set_water_price, axis=1)
#supply_data['water_price'] = supply_data.apply(set_water_price, axis=1)

#print(supply_data_lat_lon.head())
supply_data_lat_lon.to_csv("/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data_lat_lon_water.csv", index=False)
#supply_data.to_csv("/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data_water.csv", index=False)
'''

'''
cdd_county_monthly_data = pd.read_csv("/Users/maria/Documents/Research/deloitte-proj/raw-data/climdiv-cddccy-v1.0.0-20250806.txt", delim_whitespace=True, header=None)

cdd_county_monthly_data['FIPS'] = cdd_county_monthly_data[0].astype(str).str[:5]
cdd_county_monthly_data['year'] = cdd_county_monthly_data[0].astype(str).str[-4:]
print(cdd_county_monthly_data.head())
year_of_choice = "2022"

cdd_data = cdd_county_monthly_data[cdd_county_monthly_data['year'] == year_of_choice]
cdd_county_monthly_data = cdd_county_monthly_data.drop(columns=[0])

print(cdd_data.head())

seasons = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Fall': [9, 10, 11]
}

seasonal_averages = {}

for season, months in seasons.items():
    # Because month 12 comes before 1, 2, order doesn’t matter for averaging
    seasonal_averages[season] = cdd_data[months].mean(axis=1).round(4)

# Add seasonal averages as new columns:
for season in seasons.keys():
    cdd_data[season] = seasonal_averages[season]

print(cdd_data.head())
# supply_data is not updated but i dont use it anyway
'''

'process climate zone data into dictionary'
clim_zone_data = gpd.read_file('/Users/maria/Documents/Research/deloitte-proj/raw-data/Climate_Zones_-_DOE_Building_America_Program/Climate_Zones_-_DOE_Building_America_Program.shp')
clim_zone_data = clim_zone_data.dropna()


clim_zone_data['clim_zone'] = clim_zone_data['IECC_Clima'].astype(str) + clim_zone_data['IECC_Moist']
clim_zone_data.loc[clim_zone_data['IECC_Clima'] == 7, 'clim_zone'] = '7'
clim_zone_data.loc[clim_zone_data['IECC_Clima'] == 8, 'clim_zone'] = '8'
clim_zone_data = clim_zone_data[['geometry', 'clim_zone']]

print(clim_zone_data)
print(clim_zone_data.columns)

supply_data_lat_lon = pd.read_csv("/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data_lat_lon_water.csv")

gdf_points = gpd.GeoDataFrame(
    supply_data_lat_lon,
    geometry=gpd.points_from_xy(supply_data_lat_lon.longitude, supply_data_lat_lon.latitude),
    crs="EPSG:4326"
)

print(clim_zone_data.crs) # EPSG: 4326
fig, ax = plt.subplots(figsize=(14, 10))
clim_zone_data.plot(ax=ax, edgecolor='black', color='white', linewidth=0.8)
#plt.show()

# Spatial join
gdf_joined = gpd.sjoin(gdf_points, clim_zone_data, how="left", predicate="intersects")

# For all values near but not in a polygon do nearest polygon
unmatched = gdf_joined[gdf_joined['clim_zone'].isna()].copy() #find unmatched points
unmatched = unmatched.drop(columns=["index_right"])
nearest_match = gpd.sjoin_nearest(unmatched, clim_zone_data, how="left", distance_col="dist_to_zone") # nearest join only for unmatched points
print(nearest_match.head())

# Combine back
gdf_joined.loc[gdf_joined['clim_zone'].isna(), 'clim_zone'] = nearest_match['clim_zone_right'].values

# drop geometry column
supply_data_lat_lon = gdf_joined.drop(columns=["geometry", 'index_right'])

print(supply_data_lat_lon['clim_zone'].isna().sum())

# Save to CSV
supply_data_lat_lon.to_csv("/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data_lat_lon_water_clim.csv", index=False)

print(supply_data_lat_lon.tail())



#if __name__ == '__main__': make_file()