import pandas as pd
import numpy as np
#
# data_frame = pd.DataFrame([1,2], columns=['a', 'b'])
# my_data = pd.concat(data_frame)
# writer = pd.ExcelWriter('data/test.xlsx', engine='xlsxwriter')
# my_data.to_excel(writer, sheet_name='Sheet1')
# writer.save()
my_list = list(np.array([[1, 2, 3]]))
df = pd.DataFrame(data=my_list, columns=[48, 49, 50])
print(df)
# Pass `2` to `loc`

print('its done ...')