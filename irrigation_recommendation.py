"""
Irrigation Recommendation Module
Rule-based and ML-assisted irrigation scheduling system
Aims for 25-40% water savings as per project requirements
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class IrrigationRecommendationSystem:
    """
    Smart Irrigation Recommendation System
    Uses rule-based logic combined with environmental parameters
    """
    
    def __init__(self):
        self.crop_water_requirements = self._get_crop_water_requirements()
        self.irrigation_history = []
        
    def _get_crop_water_requirements(self):
        """
        Water requirements for different crops (mm/day)
        Based on FAO guidelines
        """
        return {
            'rice': {'base': 6.5, 'critical_stage': 8.0, 'min_moisture': 75},
            'maize': {'base': 5.0, 'critical_stage': 6.5, 'min_moisture': 65},
            'wheat': {'base': 4.5, 'critical_stage': 6.0, 'min_moisture': 60},
            'cotton': {'base': 5.5, 'critical_stage': 7.0, 'min_moisture': 65},
            'sugarcane': {'base': 7.0, 'critical_stage': 8.5, 'min_moisture': 70},
            'jute': {'base': 5.5, 'critical_stage': 7.0, 'min_moisture': 70},
            'coconut': {'base': 6.0, 'critical_stage': 7.5, 'min_moisture': 70},
            'papaya': {'base': 5.0, 'critical_stage': 6.5, 'min_moisture': 65},
            'orange': {'base': 4.5, 'critical_stage': 6.0, 'min_moisture': 60},
            'apple': {'base': 4.0, 'critical_stage': 5.5, 'min_moisture': 60},
            'muskmelon': {'base': 5.5, 'critical_stage': 7.0, 'min_moisture': 70},
            'watermelon': {'base': 5.5, 'critical_stage': 7.0, 'min_moisture': 70},
            'grapes': {'base': 4.0, 'critical_stage': 5.5, 'min_moisture': 60},
            'mango': {'base': 5.0, 'critical_stage': 6.5, 'min_moisture': 65},
            'banana': {'base': 6.5, 'critical_stage': 8.0, 'min_moisture': 75},
            'pomegranate': {'base': 4.5, 'critical_stage': 6.0, 'min_moisture': 60},
            'lentil': {'base': 3.5, 'critical_stage': 5.0, 'min_moisture': 55},
            'blackgram': {'base': 4.0, 'critical_stage': 5.5, 'min_moisture': 60},
            'mungbean': {'base': 4.0, 'critical_stage': 5.5, 'min_moisture': 60},
            'mothbeans': {'base': 3.5, 'critical_stage': 5.0, 'min_moisture': 55},
            'pigeonpeas': {'base': 4.0, 'critical_stage': 5.5, 'min_moisture': 60},
            'kidneybeans': {'base': 4.5, 'critical_stage': 6.0, 'min_moisture': 65},
            'chickpea': {'base': 3.5, 'critical_stage': 5.0, 'min_moisture': 55},
            'coffee': {'base': 5.5, 'critical_stage': 7.0, 'min_moisture': 70},
        }
    
    def calculate_evapotranspiration(self, temperature, humidity, wind_speed=2.0):
        """
        Calculate reference evapotranspiration (ET0) using simplified Penman-Monteith
        """
        # Simplified ET0 calculation
        # Full formula requires more parameters, this is an approximation
        
        # Saturation vapor pressure (kPa)
        es = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))
        
        # Actual vapor pressure (kPa)
        ea = es * (humidity / 100)
        
        # Vapor pressure deficit (kPa)
        vpd = es - ea
        
        # Simplified ET0 (mm/day)
        ET0 = 0.0023 * (temperature + 17.8) * np.sqrt(vpd) * wind_speed
        
        return max(ET0, 0.5)  # Minimum ET0
    
    def calculate_soil_water_balance(self, current_moisture, rainfall, evapotranspiration, 
                                    irrigation=0, runoff_coef=0.1):
        """
        Calculate soil water balance
        """
        effective_rainfall = rainfall * (1 - runoff_coef)
        
        # Water balance equation
        new_moisture = current_moisture + effective_rainfall + irrigation - evapotranspiration
        
        # Constrain between 0-100%
        new_moisture = max(0, min(100, new_moisture))
        
        return new_moisture
    
    def recommend_irrigation(self, crop, temperature, humidity, soil_moisture, 
                           rainfall_last_7days, growth_stage='vegetative', 
                           soil_type='loam', season='normal'):
        """
        Recommend irrigation based on multiple factors
        
        Args:
            crop: Crop name
            temperature: Current temperature ( C)
            humidity: Current humidity (%)
            soil_moisture: Current soil moisture (%)
            rainfall_last_7days: Rainfall in last 7 days (mm)
            growth_stage: 'seedling', 'vegetative', 'flowering', 'maturity'
            soil_type: 'sandy', 'loam', 'clay'
            season: 'normal', 'summer', 'winter'
        
        Returns:
            dict with irrigation recommendation
        """
        print(f"\n Analyzing irrigation needs for {crop}...")
        
        # Get crop water requirements
        crop_lower = crop.lower()
        if crop_lower not in self.crop_water_requirements:
            # Use average values for unknown crops
            water_req = {'base': 5.0, 'critical_stage': 6.5, 'min_moisture': 65}
        else:
            water_req = self.crop_water_requirements[crop_lower]
        
        # Adjust for growth stage
        if growth_stage in ['flowering', 'critical']:
            daily_water_need = water_req['critical_stage']
        else:
            daily_water_need = water_req['base']
        
        # Adjust for season
        season_factor = {
            'summer': 1.3,
            'winter': 0.7,
            'normal': 1.0
        }.get(season, 1.0)
        
        daily_water_need *= season_factor
        
        # Calculate ET0
        ET0 = self.calculate_evapotranspiration(temperature, humidity)
        
        # Soil type factor (water retention)
        soil_factor = {
            'sandy': 0.8,   # Low retention, needs more frequent irrigation
            'loam': 1.0,    # Balanced
            'clay': 1.2     # High retention, less frequent irrigation
        }.get(soil_type, 1.0)
        
        # Critical moisture threshold
        critical_moisture = water_req['min_moisture']
        
        # Decision logic
        irrigation_needed = False
        urgency = 'None'
        amount = 0
        frequency = 'Not needed'
        reasoning = []
        
        # Check soil moisture
        if soil_moisture < critical_moisture:
            irrigation_needed = True
            urgency = 'High' if soil_moisture < (critical_moisture - 10) else 'Medium'
            reasoning.append(f"Soil moisture ({soil_moisture}%) below critical level ({critical_moisture}%)")
        
        # Check rainfall
        avg_daily_rainfall = rainfall_last_7days / 7
        if avg_daily_rainfall < (daily_water_need * 0.5):
            irrigation_needed = True
            if urgency == 'None':
                urgency = 'Low'
            reasoning.append(f"Insufficient rainfall (avg {avg_daily_rainfall:.1f} mm/day, need {daily_water_need:.1f} mm/day)")
        
        # Check temperature stress
        if temperature > 35:
            if urgency == 'Medium':
                urgency = 'High'
            elif urgency == 'Low':
                urgency = 'Medium'
            reasoning.append(f"High temperature ({temperature} C) increases water stress")
        
        # Check humidity
        if humidity < 40:
            if urgency == 'Low':
                urgency = 'Medium'
            reasoning.append(f"Low humidity ({humidity}%) increases evaporation")
        
        # Calculate irrigation amount if needed
        if irrigation_needed:
            # Water deficit
            moisture_deficit = critical_moisture - soil_moisture
            deficit_mm = moisture_deficit * soil_factor
            
            # Add buffer for ET
            buffer = ET0 * 2
            
            amount = max(10, deficit_mm + buffer)  # Minimum 10mm
            amount = round(amount, 1)
            
            # Determine frequency
            if urgency == 'High':
                frequency = 'Immediate (within 24 hours)'
            elif urgency == 'Medium':
                frequency = 'Within 2-3 days'
            else:
                frequency = 'Within 4-5 days'
        else:
            reasoning.append("Soil moisture and rainfall levels are adequate")
        
        # Water savings calculation
        # Compare with traditional fixed-schedule irrigation
        traditional_irrigation = 50  # mm per week (typical fixed schedule)
        smart_irrigation = amount if irrigation_needed else 0
        
        if traditional_irrigation > 0:
            water_savings = ((traditional_irrigation - smart_irrigation) / traditional_irrigation) * 100
            water_savings = max(0, min(100, water_savings))
        else:
            water_savings = 0
        
        recommendation = {
            'irrigation_needed': irrigation_needed,
            'urgency': urgency,
            'amount_mm': amount,
            'frequency': frequency,
            'reasoning': reasoning,
            'daily_water_need': round(daily_water_need, 2),
            'current_moisture': soil_moisture,
            'critical_moisture': critical_moisture,
            'ET0': round(ET0, 2),
            'water_savings_percent': round(water_savings, 1)
        }
        
        # Print summary
        if irrigation_needed:
            print(f" Irrigation NEEDED - Urgency: {urgency}")
            print(f"  Amount: {amount} mm")
            print(f"  Frequency: {frequency}")
        else:
            print(f" Irrigation NOT needed - Soil moisture adequate")
        
        print(f"  Estimated water savings: {water_savings:.1f}% vs traditional irrigation")
        
        return recommendation
    
    def create_irrigation_schedule(self, crop, field_data, forecast_days=7):
        """
        Create irrigation schedule for next N days
        
        Args:
            crop: Crop name
            field_data: Dictionary with current field conditions
            forecast_days: Number of days to schedule
        
        Returns:
            List of daily irrigation recommendations
        """
        print(f"\n Creating {forecast_days}-day irrigation schedule for {crop}...")
        
        schedule = []
        current_moisture = field_data.get('soil_moisture', 70)
        
        for day in range(forecast_days):
            date = datetime.now() + timedelta(days=day)
            
            # Simulate weather variation
            temp_variation = np.random.uniform(-2, 2)
            humidity_variation = np.random.uniform(-5, 5)
            
            temp = field_data.get('temperature', 25) + temp_variation
            humidity = field_data.get('humidity', 70) + humidity_variation
            rainfall = field_data.get('rainfall_forecast', [0]*forecast_days)[day] if day < len(field_data.get('rainfall_forecast', [])) else 0
            
            # Get recommendation
            rec = self.recommend_irrigation(
                crop, temp, humidity, current_moisture,
                rainfall_last_7days=field_data.get('rainfall_last_7days', 0),
                growth_stage=field_data.get('growth_stage', 'vegetative'),
                soil_type=field_data.get('soil_type', 'loam')
            )
            
            # Update moisture for next day
            ET0 = rec['ET0']
            irrigation_amount = rec['amount_mm'] if rec['irrigation_needed'] else 0
            current_moisture = self.calculate_soil_water_balance(
                current_moisture, rainfall, ET0, irrigation_amount
            )
            
            schedule.append({
                'date': date.strftime('%Y-%m-%d'),
                'day': day + 1,
                'recommendation': rec,
                'projected_moisture': round(current_moisture, 1)
            })
        
        print(f" Schedule created for {forecast_days} days")
        
        return schedule
    
    def estimate_water_savings(self, smart_schedule, traditional_schedule_mm_per_week=50):
        """
        Estimate water savings compared to traditional irrigation
        """
        total_smart_irrigation = sum([
            day['recommendation']['amount_mm'] 
            for day in smart_schedule 
            if day['recommendation']['irrigation_needed']
        ])
        
        weeks = len(smart_schedule) / 7
        traditional_total = traditional_schedule_mm_per_week * weeks
        
        savings_mm = traditional_total - total_smart_irrigation
        savings_percent = (savings_mm / traditional_total) * 100 if traditional_total > 0 else 0
        
        print(f"\n Water Savings Analysis:")
        print(f"  Traditional irrigation: {traditional_total:.1f} mm")
        print(f"  Smart irrigation: {total_smart_irrigation:.1f} mm")
        print(f"  Savings: {savings_mm:.1f} mm ({savings_percent:.1f}%)")
        
        if savings_percent >= 25:
            print("   Meets project target (25-40% savings)")
        
        return {
            'traditional_mm': traditional_total,
            'smart_mm': total_smart_irrigation,
            'savings_mm': savings_mm,
            'savings_percent': savings_percent
        }
    
    def get_irrigation_tips(self, crop, season):
        """Get best practices for irrigation"""
        tips = [
            " Best time to irrigate: Early morning (6-8 AM) or late evening (6-8 PM)",
            " Use drip irrigation or sprinklers for better water efficiency",
            " Mulch around plants to reduce evaporation",
            " Monitor soil moisture regularly with sensors",
            " Avoid over-irrigation to prevent waterlogging and diseases",
            " Consider rainfall predictions before scheduling irrigation",
            " Adjust irrigation based on crop growth stage",
        ]
        
        # Crop-specific tips
        crop_lower = crop.lower()
        if crop_lower in ['rice', 'banana']:
            tips.append(f" {crop} requires consistent moisture - maintain higher soil moisture levels")
        elif crop_lower in ['chickpea', 'lentil', 'mothbeans']:
            tips.append(f" {crop} is drought-tolerant - avoid over-irrigation")
        
        # Season-specific tips
        if season == 'summer':
            tips.append(" Summer: Increase irrigation frequency, irrigate during cooler hours")
        elif season == 'winter':
            tips.append(" Winter: Reduce irrigation frequency, avoid waterlogging")
        
        return tips


if __name__ == "__main__":
    print("="*60)
    print("IRRIGATION RECOMMENDATION MODULE")
    print("="*60)
    print("\n Module ready for integration")
    
    # Demo
    system = IrrigationRecommendationSystem()
    
    # Test recommendation
    field_data = {
        'temperature': 32,
        'humidity': 55,
        'soil_moisture': 50,
        'rainfall_last_7days': 10,
        'growth_stage': 'flowering',
        'soil_type': 'loam'
    }
    
    rec = system.recommend_irrigation('rice', **field_data)
    print(f"\n Sample Recommendation: {rec}")
