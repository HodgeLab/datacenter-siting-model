import pandas as pd
import geopandas as gpd 

#water_scarcity= pd.read_csv("/mnt/c/Users/maria/OneDrive/Desktop/research/dataproject/datasets/Y2019M07D12_Aqueduct30_V01/baseline/annual/csv/y2019m07d11_aqueduct30_annual_v01.csv")
'''
df = gpd.read_file("/Users/maria/Documents/Research/deloitte-proj/raw-data/Y2019M07D12_Aqueduct30_V01/baseline/annual/y2019m07d11_aqueduct30_annual_v01.gpkg",
                   columns=['geometry', 'w_awr_elp_tot_cat']) #  weighted aggregated water risk, electric poewr, total, integer value
                                                              # _score instead of cat gives decimal value

print(df.head())

print(df.crs) # EPSG:4326
df.to_file("/Users/maria/Documents/Research/deloitte-proj/deloitte-data/water_risk.gpkg", driver="GPKG")

water_risk_data = gpd.read_file("/Users/maria/Documents/Research/deloitte-proj/deloitte-data/water_risk.gpkg")
print(df.head())
'''

# Water prices
water_prices = pd.read_csv("/Users/maria/Documents/Research/deloitte-proj/raw-data/water_prices.csv")


# Load in node/line and city lat/long data
city_data = pd.read_csv("/Users/maria/Documents/Research/deloitte-proj/deloitte-data/simplemaps_uscities_basicv1/uscities.csv")
city_data = city_data[['city', 'state_id', 'state_name', 'county_name', 'lat', 'lng']]
city_data = city_data.rename(columns={'state_id':'state'})
print(city_data.head())


water_price_data = pd.merge(water_prices, city_data, on = ['city', 'state'], how = 'left')
print(water_price_data.head()) # columns : city state  price state_name county_name      lat       lng
#water_price_data.to_csv('/Users/maria/Documents/Research/deloitte-proj/deloitte-data/water_prices_loc.csv', index=False)

# Added manually a few that didn't match up originally
# waterford michigan --> pontiac MI
# got rid of warrington pa since close to Philly
# chesterfield --> chester va

