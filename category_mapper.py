from heapq import merge
from turtle import position
import numpy as np
import pandas as pd
from scipy import optimize

class category_mapper:
    def orient_points(positions_df:pd.DataFrame) -> np.array:
        positions_array = np.array(positions_df)
        w,v = np.linalg.eig(np.cov(positions_array.T))
        v_ordered = v[np.argsort(w)][:-1]
        positions_norm = (v_ordered @ positions_array.T).T
        positions_center = np.mean(positions_norm, 0)
        positions_norm = positions_norm + (0.5*np.ones(positions_norm.shape) - positions_center)
        return positions_norm

    def get_mappers(data:pd.DataFrame, positions_norm:np.array, category_dims:list) -> dict:
        positions_df = pd.DataFrame(positions_norm)
        positions_dims = list(positions_df.columns)
        merged_data = pd.merge(data, positions_df, left_index=True, right_index=True)
        var_operations = {}
        sum_operations = {}
        for dim in positions_df.columns:
            var_operations[dim] = 'var'
            sum_operations[dim] = 'sum'
        sum_vars = []
        for cat in category_dims:
            cat_vars = merged_data.groupby(cat, as_index=False).agg(var_operations)
            sum_vars.append(list(cat_vars.agg(sum_operations)))
        _,org_cols = optimize.linear_sum_assignment(np.array(sum_vars))
        mapper = {}
        for i,cat in enumerate(category_dims):
            cat_means = merged_data.groupby(cat).agg({org_cols[i]: 'mean'})
            mapper[cat] = {}
            for l in cat_means.iterrows():
                mapper[cat][l[0]] = l[1].iloc[0]
        return mapper