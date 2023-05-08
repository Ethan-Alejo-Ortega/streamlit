import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
expand_e = st.expander('Documentation')
expand_e.write('Name: Ethan Ortega CS230: Section 1'
                    'Data: Mass Shooting Data Sets'
                    'URL:https://ethan-alejo-ortega-streamlit-streamlit-finalproject-ggs13m.streamlit.app/'
                    'Description:Through the use of pandas, streamlit, matplotlib.pyplot, seaborn, and folium this project takes a data' 
                    'analytics approach toward the investigation of mass shootings in the United States. Throughout this project'
                    'there are functions, dictionaries, and pivot tables that all help to create the visualizations. These '
                    'visualizations include interactive charts and maps through the integration of streamlit. This streamlit' 
                    'integration allows the user to determine the dependent variables of the charts allowing for personalized '
                    'analysis')


st.title('Investigation of United States Mass Shootings')
expander= st.expander('Project Description')
expander.write('This project investigates the unfortunate advent of Mass Shootings through the use of two datasets.'
               ' The primary areas this project investigates is the state in which it occurred, the year it occurred, '
               'the number of victims, and the gender of the perpetrator. This page includes many interactive elements'
               ' allowing the user to determine the dependent variables for the charts, creating an abundance of '
               'potential comparisons.')
df = pd.read_excel("data_csv.xlsx")
def get_state_data(df):
    def extract_state(location):
        return location.split(',')[1].strip()

    df['state'] = df['Location'].apply(extract_state)

# Converting state abbreviations to their full state name.
    state_abbr_dict = {'TX': 'Texas', 'NV': 'Nevada', 'MD': 'Maryland', 'LA': 'Louisiana',
                   'CO': 'Colorado','CA': 'California','PA': 'Pennsylvania','WA': 'Washington'}
    df['state'] = df['state'].map(lambda x: state_abbr_dict[x] if x in state_abbr_dict else x)

    # Count the number of mass shootings per state
    state_counts = df.groupby('state')['Event'].count()

    state_df = pd.DataFrame(state_counts)

    state_df = state_df.rename(columns={'Event': 'Count of Mass Shootings'})

    state_df = state_df.sort_values('Count of Mass Shootings', ascending=False)
    return state_df, state_counts

# Creating Bar Chart
state_df, state_counts = get_state_data(df)
fig, ax = plt.subplots(figsize=(12,6))
ax.set_facecolor('lightgray')
state_df.plot(kind='bar', y='Count of Mass Shootings', ax=ax, color ='red')
plt.title('Number of Mass Shootings by State', fontweight='bold', fontsize=22)
plt.xlabel('State', fontweight='bold', fontsize=16)
plt.ylabel('Count', fontweight='bold',fontsize=16)
#Streamlit
st.header('Mass Shootings by State')
'''The data used to create the following graphs has been gathered from 1982 to 2017.'''
state_options = list(state_df.index)
selected_e = st.expander('Select States')
with selected_e:
    selected_states = st.multiselect('Select States', state_options, default=state_options)
    filtered_state_df = state_df.loc[selected_states]
'''Using the boxes above decide which states to display allowing for any comparison the user decides.'''
fig, ax = plt.subplots(figsize=(12,6))
ax.set_facecolor('lightgray')
filtered_state_df.plot(kind='bar', y='Count of Mass Shootings', ax=ax, color ='red')
plt.title('Number of Mass Shootings by State', fontweight='bold', fontsize=22)
plt.xlabel('State', fontweight='bold', fontsize=16)
plt.ylabel('Count', fontweight='bold',fontsize=16)
for i, count in enumerate(filtered_state_df['Count of Mass Shootings']):
    if filtered_state_df.index[i] in selected_states:
        ax.text(i, count, str(count), ha='center', fontsize=10)

st.pyplot(fig)


# Data Frame for Subject B
df['Year'] = pd.DatetimeIndex(df['Date']).year
year_counts = df.groupby('Year')['Event'].count()

year_df = pd.DataFrame(year_counts)
year_df = year_df.rename(columns={'Event': 'Count of Mass Shootings'})
year_df = year_df.sort_index()



st.header('Investigation by Year')
year_options = list(year_df.index)
selected_year_range = st.slider('Select Year Range',
                                min_value=min(year_options),
                                max_value=max(year_options),
                                value=(min(year_options), max(year_options)),
                                step=-1)
'''Using the double slider above the user is able to investigate anytime period between 1982 and 2017.'''
start_year, end_year = selected_year_range
filtered_year_df = year_df.loc[start_year:end_year]

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_facecolor('lightgray')
sns.lineplot(data=filtered_year_df, x=filtered_year_df.index, y='Count of Mass Shootings', color='red')
ax.set_xlabel('Year', fontweight='bold', fontsize=16)
ax.set_ylabel('Count', fontweight='bold',fontsize=16)
ax.set_title('Number of Mass Shootings by Year', fontweight='bold', fontsize=22)
#streamlit
st.pyplot(fig)

st.caption('Code for double slider was created based on code from ChatGPT. See Section 1 of accompanying document.')
# Read the data from the Excel file

df = pd.read_excel("data_csv.xlsx")
df['Month'] = pd.DatetimeIndex(df['Date']).month_name()
df = df.drop(df.columns[4], axis=1)

month_counts = df.groupby('Month')['Event'].count()
st.header('Investigation by Month')
# create dataframe from counts
month_df = pd.DataFrame(month_counts)

# rename columns
month_df = month_df.rename(columns={'Event': 'Count of Mass Shootings'})

# Sorting
month_df = month_df.reindex(['January', 'February', 'March', 'April', 'May', 'June',
                             'July', 'August', 'September', 'October', 'November', 'December'])
default_months = month_df.index.tolist()
months = st.multiselect('Select months', month_df.index, default=default_months)
month_df = month_df.loc[months]
'''Using the select boxes above determine which months are shown on the barchart.'''
filtered_df = df[df['Month'].isin(months)]
filtered_month_counts = filtered_df.groupby('Month')['Event'].count()
filtered_month_df = pd.DataFrame(filtered_month_counts)
filtered_month_df = filtered_month_df.rename(columns={'Event': 'Count of Mass Shootings'})
filtered_month_df = filtered_month_df.reindex(['January', 'February', 'March', 'April', 'May', 'June',
                                               'July', 'August', 'September', 'October', 'November', 'December'])


fig, ax = plt.subplots()
ax.set_facecolor('lightgray')
sns.barplot(x=month_df.index, y="Count of Mass Shootings", data=month_df, color='red', ax=ax)


ax.set_xlabel('Month',fontweight='bold', fontsize=14)
ax.set_ylabel('Number of Mass Shootings',fontweight='bold', fontsize=14)
ax.set_title('Mass Shootings by Month',fontweight='bold', fontsize= 18)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
# Streamlit
st.pyplot(fig)

#CREATING MAP
df = pd.read_excel("data_csv.xlsx")

df = df[['Date', 'Event', 'Location', 'Lat', 'Lng', 'location_2']]
df.columns = ['Date', 'Event', 'Location', 'Lat', 'Lng', 'location_2']

# extract latitude and longitude from location_2 column
df['Latitude'] = df['location_2'].apply(lambda x: float(x.split('(')[1].split(' ')[0]))
df['Longitude'] = df['location_2'].apply(lambda x: float(x.split(' ')[1].split(')')[0]))

# drop location_2 column
df.drop('location_2', axis=1, inplace=True)
#MAP
us_center = [37.0902, -95.7129]
m = folium.Map(location=us_center, zoom_start=3)


# Add a red dot for each event
for index, row in df.iterrows():
    folium.Marker([row['Lat'], row['Lng']], popup=row['Event'], icon=folium.Icon(color='red')).add_to(m)

# Display the map in Streamlit
st.header('Mass Shooting Events in the United States')
folium_static(m)



# Pivot Table
st.header('Pivot Tables')
st.caption('Pivot Table that illustrates the Gender, State, and Year of Mass Shootings up to 2015.')
df = pd.read_excel("USMassShootings.xlsx")
pivot_table = df.pivot_table(values='TOTALVICTIMS', index=['GENDER', 'STATE', 'YEAR'], aggfunc='sum')
pivot_table = df.loc[df['SHOOTINGTYPE'].isin(['Mass', 'Spree']) & df['LOCATIONTYPE'].isin(['School', 'Workplace','Military','Religious','Other'])].pivot_table(values='TOTALVICTIMS', index=['GENDER', 'STATE', 'YEAR'], aggfunc='sum')
pivot_table = pivot_table.rename(columns={'TOTALVICTIMS': 'Victims'})

st.write(pivot_table)
#Pivot Table: Filter using and/or

st.caption('Pivot Table that only considers Mass Shootings by Males with the location being at a School.')
df_male_school = df.loc[(df['GENDER'] == 'Male') & (df['LOCATIONTYPE'] == 'School')].pivot_table(values='TOTALVICTIMS', index=['STATE', 'YEAR'], aggfunc='sum')
df_male_school = df_male_school.rename(columns={'TOTALVICTIMS': 'Victims'})

st.write(df_male_school)

#Piechart
df = pd.read_excel("USMassShootings.xlsx")

gender_counts = df['GENDER'].value_counts()

# Set up the plot
fig, ax = plt.subplots(figsize=(4, 4), facecolor='black')
ax.set_title('Gender Distribution', color='white')

# Define the colors
colors = ['red', 'white']

# Create the pie chart
ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=colors, textprops={'color': 'white'})
# Set legend font color
legend = ax.legend(title="Gender", loc="upper right", fontsize=12)
plt.setp(legend.get_texts(), color='black')

plt.setp(ax.patches, linewidth=0.5, edgecolor='black')


plt.setp(ax.patches, linewidth=0.5, edgecolor='black')
plt.setp(ax.texts, fontsize=8, color='black')
# Display the plot
st.pyplot(fig)

st.caption('This Streamlit page was created by Ethan Ortega.')
