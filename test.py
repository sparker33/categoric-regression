import pandas as pd
import numpy as np
import datetime

from regressor import regressor
from arrange_sim import arrange_sim
from category_mapper import category_mapper

data = pd.read_csv("raw_polls.csv")
significant_polls = data.groupby('pollster_rating_id', as_index=False)['poll_id'].count().rename(columns={'poll_id':'polls_count'})
significant_polls = significant_polls[significant_polls['polls_count']>500]
print(significant_polls)
data = pd.merge(data, significant_polls, how='inner', on='pollster_rating_id')
data['poll_date_relative'] = data.apply(lambda row: int((datetime.datetime.strptime(row['electiondate'], "%m/%d/%Y") - datetime.datetime.strptime(row['polldate'], "%m/%d/%Y")).days), axis=1)

data_model = data[['location', 'type_detail', 'pollster_rating_id', 'year', 'poll_date_relative', 'cand1_pct', 'cand2_pct', 'cand1_actual', 'cand2_actual']].copy()

cardinal_dims_indexes = [3,4,5,6]
result_dim_index = [7,8]
base_regression = regressor(data_model, cardinal_dims_indexes, result_dim_index)

base_results = base_regression.get_Y_pred_Series()
data_model['Cand1_Base_Pred'] = base_regression.get_Y_pred_Series().apply(lambda val: val[0])
data_model['Cand2_Base_Pred'] = base_regression.get_Y_pred_Series().apply(lambda val: val[1])
data_model['Cand1_Base_Error'] = (data_model['Cand1_Base_Pred'] - data_model['cand1_actual']) / 100.0
data_model['Cand2_Base_Error'] = (data_model['Cand2_Base_Pred'] - data_model['cand2_actual']) / 100.0

category_dims = ['location', 'type_detail', 'pollster_rating_id']
vals_dims = ['Cand1_Base_Error', 'Cand2_Base_Error']
sim = arrange_sim(data_model, category_dims, vals_dims, values_force_const=0.012, categories_force_const=0.1, ambient_force_const=-0.2,
                    values_order_const=1.0, categories_order_const=0.0, ambient_order_const=0.0, damping_const=0.33, dist_thresh=10**(-8))
                    
sim.run_sim(duration=3000, increment=0.1, cutoff_mean_kinetic_energy=10.0**(-5), print_prog=True)

point_positions = sim.get_positions_df()
arranged_points_norm = category_mapper.orient_points(point_positions)
mapper = category_mapper.get_mapper(data, arranged_points_norm, category_dims)

print(mapper)