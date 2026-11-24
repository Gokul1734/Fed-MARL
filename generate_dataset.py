"""
Generate 20,000 sample records for Meri Pechaan Dataset
Matches the structure of the original dataset with realistic agricultural data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Indian states and major agricultural districts
states_districts = {
    'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Trichy', 'Salem', 'Tirunelveli', 'Erode', 'Vellore'],
    'Karnataka': ['Bangalore', 'Mysore', 'Hubli', 'Belgaum', 'Mangalore', 'Gulbarga', 'Bellary', 'Shimoga'],
    'Andhra Pradesh': ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore', 'Tirupati', 'Rajahmundry', 'Kakinada', 'Anantapur'],
    'Telangana': ['Hyderabad', 'Warangal', 'Nizamabad', 'Karimnagar', 'Khammam', 'Mahbubnagar', 'Rangareddy', 'Medak'],
    'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Nashik', 'Aurangabad', 'Solapur', 'Kolhapur', 'Ahmednagar'],
    'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Bhavnagar', 'Jamnagar', 'Junagadh', 'Anand'],
    'Punjab': ['Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala', 'Bathinda', 'Mohali', 'Hoshiarpur', 'Moga'],
    'Haryana': ['Faridabad', 'Gurgaon', 'Hisar', 'Rohtak', 'Panipat', 'Karnal', 'Sonipat', 'Ambala'],
    'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Agra', 'Varanasi', 'Meerut', 'Allahabad', 'Bareilly', 'Ghaziabad'],
    'Madhya Pradesh': ['Indore', 'Bhopal', 'Jabalpur', 'Gwalior', 'Ujjain', 'Sagar', 'Dewas', 'Ratlam'],
    'Rajasthan': ['Jaipur', 'Jodhpur', 'Kota', 'Bikaner', 'Udaipur', 'Ajmer', 'Bhilwara', 'Alwar'],
    'West Bengal': ['Kolkata', 'Howrah', 'Durgapur', 'Siliguri', 'Asansol', 'Bardhaman', 'Malda', 'Murshidabad'],
    'Bihar': ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur', 'Purnia', 'Darbhanga', 'Arrah', 'Bihar Sharif'],
    'Odisha': ['Bhubaneswar', 'Cuttack', 'Rourkela', 'Brahmapur', 'Puri', 'Sambalpur', 'Balasore', 'Baripada'],
    'Kerala': ['Thiruvananthapuram', 'Kochi', 'Kozhikode', 'Thrissur', 'Kollam', 'Palakkad', 'Alappuzha', 'Malappuram'],
    'Assam': ['Guwahati', 'Silchar', 'Dibrugarh', 'Jorhat', 'Nagaon', 'Tinsukia', 'Tezpur', 'Bongaigaon']
}

# Agency name (constant)
agency_name = "NRSC VIC MODEL"

# Generate date range (3 years of data)
start_date = datetime(2023, 1, 1)
end_date = datetime(2026, 12, 31)

print("Generating 20,000 sample records for Meri Pechaan Dataset...")
print("="*70)

# Generate records
records = []
total_records = 20000

# Create date pool (dates from 2023-2026)
date_pool = []
current_date = start_date
while current_date <= end_date:
    date_pool.append(current_date)
    current_date += timedelta(days=1)

print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"States covered: {len(states_districts)}")
print(f"Total districts: {sum(len(districts) for districts in states_districts.values())}")
print("="*70)

for i in range(total_records):
    # Randomly select state and district
    state = random.choice(list(states_districts.keys()))
    district = random.choice(states_districts[state])
    
    # Randomly select date
    date_obj = random.choice(date_pool)
    date_str = date_obj.strftime('%d-%m-%Y')
    year = date_obj.year
    month = date_obj.month
    
    # Generate realistic soil moisture level (2-6 range with seasonal variation)
    # Higher moisture in monsoon months (June-September), lower in summer (March-May)
    if month in [6, 7, 8, 9]:  # Monsoon season
        base_moisture = np.random.uniform(4.5, 6.0)
    elif month in [3, 4, 5]:  # Summer season
        base_moisture = np.random.uniform(2.0, 3.5)
    elif month in [12, 1, 2]:  # Winter season
        base_moisture = np.random.uniform(2.5, 4.5)
    else:  # Other months
        base_moisture = np.random.uniform(3.0, 5.0)
    
    # Add some random variation
    soil_moisture = base_moisture + np.random.uniform(-0.5, 0.5)
    soil_moisture = max(1.5, min(6.5, soil_moisture))  # Clamp between 1.5 and 6.5
    
    # Create record
    record = {
        'State': state,
        'District': district,
        'Date': date_str,
        'Year': year,
        'Month': month,
        'Avg_smlvl_at15cm': round(soil_moisture, 9),
        'Agency_name': agency_name
    }
    
    records.append(record)
    
    # Progress indicator
    if (i + 1) % 2000 == 0:
        print(f"Generated {i + 1:,} records...")

# Create DataFrame
df = pd.DataFrame(records)

# Sort by State, District, Year, Month, Date for better organization
df = df.sort_values(['State', 'District', 'Year', 'Month', 'Date']).reset_index(drop=True)

# Save to CSV
output_file = 'Meri Pechaan Official Dataset.csv'
df.to_csv(output_file, index=False)

print("="*70)
print(f"Successfully generated {len(df):,} records!")
print(f"Saved to: {output_file}")
print("="*70)

# Display summary statistics
print("\nDATASET SUMMARY:")
print("="*70)
print(f"Total Records: {len(df):,}")
print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
print(f"States: {df['State'].nunique()}")
print(f"Districts: {df['District'].nunique()}")
print(f"Years: {sorted(df['Year'].unique())}")
print(f"\nSoil Moisture Statistics:")
print(f"  Mean: {df['Avg_smlvl_at15cm'].mean():.3f}")
print(f"  Min: {df['Avg_smlvl_at15cm'].min():.3f}")
print(f"  Max: {df['Avg_smlvl_at15cm'].max():.3f}")
print(f"  Std Dev: {df['Avg_smlvl_at15cm'].std():.3f}")
print("="*70)

# Display sample data
print("\nSAMPLE RECORDS (First 10):")
print("="*70)
print(df.head(10).to_string(index=False))
print("="*70)

print("\nDataset generation complete!")
print("Ready for use in Smart Agriculture System")
