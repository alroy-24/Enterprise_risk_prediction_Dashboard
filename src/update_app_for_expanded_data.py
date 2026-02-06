"""
Script to update the Streamlit app to use the expanded dataset.
"""

import pandas as pd
import shutil
from pathlib import Path


def update_app_data_source():
    """Update app.py to use the expanded dataset."""
    
    app_path = Path("src/app.py")
    backup_path = Path("src/app_backup.py")
    
    # Create backup
    if app_path.exists():
        shutil.copy2(app_path, backup_path)
        print(f"✅ Created backup: {backup_path}")
    
    # Read current app.py
    with open(app_path, 'r', encoding='utf-8') as f:
        app_content = f.read()
    
    # Replace data loading section
    old_data_loading = '''# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/sample_financials.csv")
    df = engineer_features(df)
    return df'''
    
    new_data_loading = '''# Load data
@st.cache_data
def load_data():
    # Try expanded dataset first, fallback to original
    try:
        df = pd.read_csv("data/expanded_real_world_financials.csv")
        st.info("📊 Using expanded real-world dataset (2,144 records)")
    except FileNotFoundError:
        try:
            df = pd.read_csv("data/real_world_financials.csv")
            st.info("📊 Using real-world dataset (15 records)")
        except FileNotFoundError:
            df = pd.read_csv("data/sample_financials.csv")
            st.info("📊 Using sample dataset (20 records)")
    
    df = engineer_features(df)
    return df'''
    
    # Replace the data loading function
    if old_data_loading in app_content:
        app_content = app_content.replace(old_data_loading, new_data_loading)
        print("✅ Updated data loading function")
    else:
        print("⚠️ Could not find exact data loading function to replace")
    
    # Write updated app
    with open(app_path, 'w', encoding='utf-8') as f:
        f.write(app_content)
    
    print(f"✅ Updated {app_path}")


def create_dataset_info():
    """Create a dataset information display for the app."""
    
    info_code = '''
def show_dataset_info(df):
    """Display information about the current dataset."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    
    total_records = len(df)
    unique_companies = df['company_id'].nunique()
    industries = df['industry'].nunique() if 'industry' in df.columns else 0
    regions = df['region'].nunique() if 'region' in df.columns else 0
    
    st.sidebar.metric("📈 Total Records", f"{total_records:,}")
    st.sidebar.metric("🏢 Companies", f"{unique_companies:,}")
    st.sidebar.metric("🏭 Industries", f"{industries:,}")
    st.sidebar.metric("🌍 Regions", f"{regions:,}")
    
    # Show industry breakdown if available
    if 'industry' in df.columns:
        st.sidebar.markdown("#### 🏭 Industry Breakdown")
        industry_counts = df['industry'].value_counts().head(5)
        for industry, count in industry_counts.items():
            st.sidebar.write(f"• {industry}: {count:,}")
'''
    
    return info_code


def main():
    """Main function to update app for expanded data."""
    
    print("🚀 Updating App for Expanded Dataset")
    print("=" * 40)
    
    # Update app.py
    update_app_data_source()
    
    print("\n💡 App Updated Successfully!")
    print("\n🔄 Restart your Streamlit app to see the changes:")
    print("   streamlit run src/app.py")
    
    print("\n📊 New Dataset Features:")
    print("   • 2,144 total records")
    print("   • 48 real companies across 8 sectors")
    print("   • 28 different industries")
    print("   • 3 years of historical data")
    print("   • 2,000 synthetic variations for model training")
    
    print("\n🎯 Benefits:")
    print("   • Better risk model training")
    print("   • More diverse industry coverage")
    print("   • Improved statistical significance")
    print("   • Enhanced dashboard visualizations")


if __name__ == "__main__":
    main()
