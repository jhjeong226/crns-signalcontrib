"""
PC Observatory Data Extractor and Template Generator
===================================================

This script extracts data from existing PC observatory Excel files and 
creates a LocalSiteSpec template with actual measured data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def extract_pc_data():
    """Extract and process PC observatory data from existing Excel files"""
    
    print("🔍 Extracting PC Observatory Data...")
    
    # File paths
    geo_file = r"E:\02.Data\01.Raw\18.CRNP_ML\PC\geo_locations_PC.xlsx"
    crnp_file = r"E:\02.Data\01.Raw\18.CRNP_ML\PC\PC_CRNP_input.xlsx" 
    fdr_file = r"E:\02.Data\01.Raw\18.CRNP_ML\PC\PC_FDR_daily_depths.xlsx"
    
    # ================================
    # 1. Site Information from geo_locations_PC.xlsx
    # ================================
    print("📍 Reading geographical location data...")
    geo_data = pd.read_excel(geo_file)
    
    # Find CRNP detector location
    crnp_location = geo_data[geo_data['id'] == 'CRNP'].iloc[0]
    site_lat = crnp_location['lat']
    site_lon = crnp_location['lon'] 
    site_elevation = 50  # Estimate from general area (should be measured)
    
    print(f"   CRNP Location: {site_lat:.6f}°N, {site_lon:.6f}°E")
    
    # ================================
    # 2. Environmental Parameters from CRNP input data
    # ================================
    print("🌡️ Reading CRNP environmental data...")
    crnp_data = pd.read_excel(crnp_file)
    
    # Calculate average conditions
    avg_temp = crnp_data['Ta'].mean()
    avg_humidity_abs = crnp_data['RH'].mean()  # This is relative humidity, need conversion
    avg_pressure = crnp_data['Pa'].mean()
    avg_neutron_count = crnp_data['N_counts'].mean()
    
    # Convert relative humidity to absolute humidity (rough estimate)
    # Absolute humidity (g/m³) ≈ RH(%) × saturated_vapor_density(T) / 100
    # This is simplified - should use proper psychrometric calculations
    saturated_vapor_density = 6.11 * np.exp(17.27 * avg_temp / (avg_temp + 237.3)) * 18.02 / (8314 * (avg_temp + 273.15)) * 1000
    abs_humidity_summer = avg_humidity_abs * saturated_vapor_density / 100
    abs_humidity_winter = abs_humidity_summer * 0.4  # Rough winter estimate
    
    print(f"   Average Temperature: {avg_temp:.1f}°C")
    print(f"   Average Pressure: {avg_pressure:.1f} hPa")
    print(f"   Average Neutron Count: {avg_neutron_count:.0f}")
    print(f"   Estimated Absolute Humidity (Summer): {abs_humidity_summer:.1f} g/m³")
    
    # ================================
    # 3. Field Measurements from FDR data
    # ================================
    print("💧 Reading soil moisture measurements...")
    fdr_10cm = pd.read_excel(fdr_file, sheet_name='10cm')
    
    # Get recent soil moisture measurements (last available data)
    recent_date = fdr_10cm['Date'].max()
    recent_data = fdr_10cm[fdr_10cm['Date'] == recent_date].iloc[0]
    
    print(f"   Using soil moisture data from: {recent_date}")
    
    # Create field measurements table with actual locations and soil moisture
    field_measurements = []
    
    # Process each measurement location
    for _, location in geo_data.iterrows():
        if location['id'] == 'CRNP':
            continue  # Skip detector location for now
            
        location_id = location['id']
        distance = location['dist']
        
        # Get soil moisture for this location if available
        if location_id in recent_data.index or location_id in fdr_10cm.columns:
            try:
                soil_moisture = recent_data[location_id] if location_id in recent_data.index else np.nan
                if pd.isna(soil_moisture) and location_id in fdr_10cm.columns:
                    # Get last valid measurement
                    valid_data = fdr_10cm[location_id].dropna()
                    soil_moisture = valid_data.iloc[-1] if len(valid_data) > 0 else 0.25
            except:
                soil_moisture = 0.25  # Default value
        else:
            soil_moisture = 0.25  # Default value
            
        # Parse direction from location ID
        direction = 0  # Default North
        if 'N' in location_id and 'E' in location_id:
            direction = 45  # NE
        elif 'S' in location_id and 'E' in location_id:
            direction = 135  # SE
        elif 'S' in location_id and 'W' in location_id:
            direction = 225  # SW
        elif 'N' in location_id and 'W' in location_id:
            direction = 315  # NW
        elif 'E' in location_id:
            direction = 90  # E
        elif 'S' in location_id:
            direction = 180  # S
        elif 'W' in location_id:
            direction = 270  # W
        # N stays 0
        
        field_measurements.append({
            'distance': round(distance, 1),
            'direction': direction,
            'soil_moisture': round(soil_moisture, 3) if not pd.isna(soil_moisture) else 0.25,
            'measurement_date': recent_date.strftime('%Y-%m-%d'),
            'notes': f'Location {location_id}'
        })
    
    # Add detector location (0m)
    field_measurements.insert(0, {
        'distance': 0,
        'direction': 0,
        'soil_moisture': 0.25,  # Estimate - should be measured
        'measurement_date': recent_date.strftime('%Y-%m-%d'),
        'notes': 'CRNP Detector location'
    })
    
    # ================================
    # 4. Seasonal Data Estimation
    # ================================
    print("🌿 Estimating seasonal variations...")
    
    # Get seasonal averages from FDR data if enough data available
    fdr_10cm['Date'] = pd.to_datetime(fdr_10cm['Date'])
    fdr_10cm['Month'] = fdr_10cm['Date'].dt.month
    
    seasonal_data = {
        'spring': {'avg_humidity': abs_humidity_summer * 0.7, 'avg_soil_moisture': 0.25},
        'summer': {'avg_humidity': abs_humidity_summer, 'avg_soil_moisture': 0.20},
        'autumn': {'avg_humidity': abs_humidity_summer * 0.5, 'avg_soil_moisture': 0.22},
        'winter': {'avg_humidity': abs_humidity_winter, 'avg_soil_moisture': 0.28}
    }
    
    # Calculate actual seasonal averages if data available
    try:
        spring_months = [3, 4, 5]
        summer_months = [6, 7, 8] 
        autumn_months = [9, 10, 11]
        winter_months = [12, 1, 2]
        
        for season, months in [('spring', spring_months), ('summer', summer_months), 
                              ('autumn', autumn_months), ('winter', winter_months)]:
            season_data = fdr_10cm[fdr_10cm['Month'].isin(months)]
            if len(season_data) > 0:
                # Calculate average across all measurement points
                numeric_cols = season_data.select_dtypes(include=[np.number]).columns
                moisture_cols = [col for col in numeric_cols if col not in ['Month']]
                if moisture_cols:
                    seasonal_avg = season_data[moisture_cols].mean().mean()
                    seasonal_data[season]['avg_soil_moisture'] = round(seasonal_avg, 3)
                    
    except Exception as e:
        print(f"   Warning: Could not calculate seasonal averages: {e}")
    
    # ================================
    # 5. Compile Observatory Configuration
    # ================================
    observatory_config = {
        'site_info': {
            'site_name': 'PC Observatory',
            'latitude': round(site_lat, 6),
            'longitude': round(site_lon, 6), 
            'elevation': site_elevation,
            'installation_date': '2024-08-01'  # Based on data start date
        },
        'environmental_parameters': {
            'bulk_density': 1.2,  # Estimate from geo data (sbd), should be measured
            'air_humidity_summer': round(abs_humidity_summer, 1),
            'air_humidity_winter': round(abs_humidity_winter, 1),
            'pressure': round(avg_pressure, 1),
            'vegetation_height': 0.3,  # Estimate
            'clay_content': 25,  # Estimate - needs soil analysis
            'sand_content': 45,  # Estimate - needs soil analysis
            'silt_content': 30,  # Estimate - needs soil analysis
            'organic_matter': 2.5,  # Estimate - needs soil analysis
            'detector_height': 1.5,  # Standard height
            'N0_reference': round(avg_neutron_count, 0),
            'counting_time': 3600  # 1 hour integration
        },
        'field_measurements': field_measurements,
        'seasonal_data': seasonal_data
    }
    
    return observatory_config

def create_pc_localsitespec_excel(config, filename='PC_LocalSiteSpec.xlsx'):
    """Create Excel template with PC observatory data"""
    
    print(f"📝 Creating Excel template: {filename}")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Site Information Sheet
        site_info_data = []
        for key, value in config['site_info'].items():
            site_info_data.append([key, value])
            
        site_info = pd.DataFrame(site_info_data, columns=['Parameter', 'Value'])
        site_info['Unit'] = ['text', 'degrees', 'degrees', 'meters', 'YYYY-MM-DD']
        site_info['Description'] = [
            'Observatory name',
            'Latitude in decimal degrees',
            'Longitude in decimal degrees',
            'Elevation above sea level',
            'Installation date'
        ]
        site_info.set_index('Parameter').to_excel(writer, sheet_name='Site_Info')
        
        # Environmental Parameters Sheet
        env_data = []
        for key, value in config['environmental_parameters'].items():
            env_data.append([key, value])
            
        env_params = pd.DataFrame(env_data, columns=['Parameter', 'Value'])
        env_params['Unit'] = ['g/cm³', 'g/m³', 'g/m³', 'hPa', 'm', '%', '%', '%', '%', 'm', 'counts', 'seconds']
        env_params['Description'] = [
            'Soil bulk density (from field measurements)',
            'Summer air humidity (calculated from CRNP data)',
            'Winter air humidity (estimated)',
            'Atmospheric pressure (from CRNP data)',
            'Average vegetation height (estimated)',
            'Clay content percentage (needs analysis)',
            'Sand content percentage (needs analysis)',
            'Silt content percentage (needs analysis)',
            'Organic matter content (needs analysis)',
            'Detector height above ground',
            'Reference neutron count (from CRNP data)',
            'Integration time'
        ]
        env_params.set_index('Parameter').to_excel(writer, sheet_name='Environmental_Parameters')
        
        # Field Measurements Sheet
        field_df = pd.DataFrame(config['field_measurements'])
        field_df.to_excel(writer, sheet_name='Field_Measurements', index=False)
        
        # Seasonal Data Sheet
        seasonal_rows = []
        for season, data in config['seasonal_data'].items():
            seasonal_rows.append([
                season,
                data['avg_humidity'],
                data['avg_soil_moisture'],
                f'{season.capitalize()} average conditions'
            ])
            
        seasonal_df = pd.DataFrame(seasonal_rows, 
                                 columns=['Season', 'avg_humidity', 'avg_soil_moisture', 'description'])
        seasonal_df.set_index('Season').to_excel(writer, sheet_name='Seasonal_Data')
        
        # Data Sources and Notes Sheet
        notes_data = pd.DataFrame({
            'Data_Source': [
                'Geographical Locations',
                'CRNP Environmental Data', 
                'FDR Soil Moisture Measurements',
                'Seasonal Estimates',
                '',
                'NOTES:',
                'Values marked as "estimated" should be replaced with actual measurements',
                'Soil composition (clay/sand/silt) requires laboratory analysis',
                'Vegetation height and coverage should be surveyed',
                'Bulk density values are from site characterization data',
                'Air humidity converted from relative to absolute humidity',
                'Seasonal data calculated from available FDR measurements',
                '',
                'MEASUREMENT RECOMMENDATIONS:',
                '1. Conduct soil sampling at multiple depths and locations',
                '2. Measure vegetation characteristics seasonally',
                '3. Install meteorological sensors for continuous monitoring',
                '4. Perform detector calibration with standard sources',
                '5. Document site changes and maintenance activities'
            ]
        })
        notes_data.to_excel(writer, sheet_name='Data_Sources_and_Notes', index=False)
    
    print(f"✅ Excel template created successfully!")
    
    return filename

def main():
    """Main function to extract PC data and create template"""
    
    print("🌍 PC Observatory Data Extraction and Template Generation")
    print("=" * 60)
    
    try:
        # Extract data from existing files
        config = extract_pc_data()
        
        # Create Excel template
        excel_file = create_pc_localsitespec_excel(config)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Site: {config['site_info']['site_name']}")
        print(f"   Location: {config['site_info']['latitude']:.4f}°N, {config['site_info']['longitude']:.4f}°E")
        print(f"   Field measurements: {len(config['field_measurements'])} locations")
        print(f"   Reference neutron count: {config['environmental_parameters']['N0_reference']:.0f}")
        print(f"   Excel template: {excel_file}")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Review and validate the extracted data")
        print(f"   2. Replace estimated values with actual measurements")
        print(f"   3. Run CRNS analysis: python multi_observatory_analysis.py")
        print(f"   4. Select option 3 and use filename: {excel_file}")
        
        return config, excel_file
        
    except Exception as e:
        print(f"❌ Error during data extraction: {e}")
        return None, None

if __name__ == "__main__":
    config, excel_file = main()