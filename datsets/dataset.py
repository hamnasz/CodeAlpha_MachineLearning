import requests

# Replace this with your CSV URL
url = 'https://archive.ics.uci.edu/static/public/27/data.csv'

# File name you want to save as
filename = 'data3.csv'

# Download the file
response = requests.get(url)

# Save the content
with open(filename, 'wb') as file:
    file.write(response.content)

print(f"CSV file downloaded and saved as '{filename}'")
