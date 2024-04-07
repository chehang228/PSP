
from fml import read_data
from fml.preprocessing import Preprocessing
from fml.sampling import HpSplit, random_split
from fml.validates import Validate
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

dataset = read_data("filled_descriptors.csv", df=False)

dataset = Preprocessing(corr_criterion=0.95).fit_transform(dataset)

dataset.to_df().to_excel("preprocessed_data.xlsx")
