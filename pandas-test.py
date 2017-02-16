import pandas as pd

data = {
    'Name': ['John', 'Milad', 'Katy'],
    'Location': ['New York', 'Irvine', 'Irvine'],
    'Age': [23, 24, 24]
}

data_pandas = pd.DataFrame(data)
print(data_pandas)