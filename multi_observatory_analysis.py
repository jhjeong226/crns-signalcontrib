"""
Quick PC Observatory Analysis Test
=================================

A simplified version to test PC observatory data without complex plotting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add path for CRNS modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corny.Schroen2017hess import get_footprint
from corny.Schroen2022hess import Field_at_Distance

def quick_pc_analysis():
    """Quick analysis of PC observatory data"""
    
    print("🔬 Quick PC Observatory Analysis")
    print("=" * 40)
    
    # Read the Excel file
    excel_file = 'PC_LocalSiteSpec.xlsx'
    
    try:
        # Read data
        site_info = pd.read_excel(excel_file, sheet_name='Site_Info', index_col=0)
        env_params = pd.read_excel(excel_file, sheet_name='Environmental_Parameters', index_col=0)
        field_measurements = pd.read_excel(excel_file, sheet_name='Field_Measurements')
        seasonal_data = pd.read_excel(excel_file, sheet_name='Seasonal_Data', index_col=0)
        
        print(f"✅ Successfully loaded data")
        
        # Extract key parameters
        site_name = site_info.loc['site_name', 'Value']
        latitude = float(site_info.loc['latitude', 'Value'])
        longitude = float(site_info.loc['longitude', 'Value'])
        
        bulk_density = float(env_params.loc['bulk_density', 'Value'])
        pressure = float(env_params.loc['pressure', 'Value'])
        N0 = float(env_params.loc['N0_reference', 'Value'])
        
        print(f"📍 Site: {site_name}")
        print(f"   Location: {latitude:.4f}°N, {longitude:.4f}°E")
        print(f"   Bulk Density: {bulk_density} g/cm³")
        print(f"   Reference N0: {N0:.0f}")
        
        # Analyze field measurements
        print(f"\n💧 Field Measurements:")
        print(f"   Number of locations: {len(field_measurements)}")
        
        # Convert columns to proper types
        field_measurements['distance'] = field_measurements['distance'].astype(float)
        field_measurements['direction'] = field_measurements['direction'].astype(float)
        field_measurements['soil_moisture'] = field_measurements['soil_moisture'].astype(float)
        
        mean_sm = field_measurements['soil_moisture'].mean()
        std_sm = field_measurements['soil_moisture'].std()
        
        print(f"   Soil moisture mean: {mean_sm:.3f}")
        print(f"   Soil moisture std: {std_sm:.3f}")
        print(f"   Coefficient of variation: {std_sm/mean_sm*100:.1f}%")
        
        # Calculate seasonal footprints
        print(f"\n🌿 Seasonal Footprint Analysis:")
        
        for season in seasonal_data.index:
            humidity = float(seasonal_data.loc[season, 'avg_humidity'])
            soil_moisture = float(seasonal_data.loc[season, 'avg_soil_moisture'])
            
            # Calculate R86 footprint
            R86 = get_footprint(soil_moisture, humidity, pressure)
            area_ha = np.pi * R86**2 / 10000
            
            print(f"   {season.capitalize():8}: R86={R86:6.1f}m, Area={area_ha:5.2f}ha")
        
        # Test field at distance calculation
        print(f"\n🎯 Signal Contribution Test:")
        
        # Test with a few distances
        test_distances = [25, 50, 100]
        
        for dist in test_distances:
            # Find measurements near this distance
            nearby = field_measurements[
                (field_measurements['distance'] >= dist-15) & 
                (field_measurements['distance'] <= dist+15)
            ]
            
            if len(nearby) > 0:
                zone_sm = nearby['soil_moisture'].mean()
                
                try:
                    N1, N2, Neff, thetaeff, c1, c2 = Field_at_Distance(
                        float(dist), 
                        theta1=float(mean_sm), 
                        theta2=float(zone_sm),
                        hum=5.0,  # Default humidity
                        N0=float(N0),
                        bd=float(bulk_density),
                        verbose=False, max_radius=500
                    )
                    
                    print(f"   {dist}m zone: SM={zone_sm:.3f}, Contribution={c2:.3f}")
                    
                except Exception as e:
                    print(f"   {dist}m zone: Error in calculation - {e}")
            else:
                print(f"   {dist}m zone: No measurements available")
        
        # Create simple plot
        print(f"\n📊 Creating simple plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Measurement locations
        for _, row in field_measurements.iterrows():
            if row['distance'] > 0:
                angle_rad = np.radians(row['direction'])
                x = row['distance'] * np.sin(angle_rad)
                y = row['distance'] * np.cos(angle_rad)
                ax1.scatter(x, y, c=row['soil_moisture'], s=100, cmap='Blues', vmin=0.15, vmax=0.4)
                ax1.annotate(f'{row["soil_moisture"]:.2f}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        ax1.scatter(0, 0, c='red', s=200, marker='s', label='CRNP Detector')
        ax1.set_xlabel('Distance East (m)')
        ax1.set_ylabel('Distance North (m)')
        ax1.set_title(f'{site_name}\nMeasurement Locations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: Seasonal footprints
        seasons = seasonal_data.index
        R86_values = []
        
        for season in seasons:
            humidity = float(seasonal_data.loc[season, 'avg_humidity'])
            soil_moisture = float(seasonal_data.loc[season, 'avg_soil_moisture'])
            R86 = get_footprint(soil_moisture, humidity, pressure)
            R86_values.append(R86)
        
        bars = ax2.bar(seasons, R86_values, color=['lightgreen', 'orange', 'brown', 'lightblue'])
        ax2.set_title('Seasonal Footprint Variation')
        ax2.set_ylabel('R86 Footprint (m)')
        ax2.set_ylim(0, max(R86_values) * 1.1)
        
        # Add value labels
        for bar, value in zip(bars, R86_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{value:.1f}m', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('pc_quick_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Plot saved as: pc_quick_analysis.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_pc_analysis()