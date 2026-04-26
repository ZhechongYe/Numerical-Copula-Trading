import numpy as np
import pandas as pd
from scipy.stats import t, norm


def compute_mispricing_index(uv_df, best_copula_info, u_col="u", v_col="v", eps=1e-6):
   
    if isinstance(best_copula_info, pd.Series):
        best_copula_info = best_copula_info.to_dict()

    copula_name = best_copula_info["best copula"]
    params = best_copula_info["params"]

    u = np.clip(uv_df[u_col].astype(float).to_numpy(), eps, 1 - eps)
    v = np.clip(uv_df[v_col].astype(float).to_numpy(), eps, 1 - eps)


    if copula_name in ["student_t", "student-t", "t"]:
        rho = params["rho"]
        df = params["df"]

        x = t.ppf(u, df)
        y = t.ppf(v, df)

        scale_x_given_y = np.sqrt((df + y**2) * (1 - rho**2) / (df + 1))
        h_1_given_2 = t.cdf(
            (x - rho * y) / scale_x_given_y,
            df=df + 1
        )


        scale_y_given_x = np.sqrt((df + x**2) * (1 - rho**2) / (df + 1))
        h_2_given_1 = t.cdf(
            (y - rho * x) / scale_y_given_x,
            df=df + 1
        )

    elif copula_name == "gaussian":
        rho = params["rho"]

        x = norm.ppf(u)
        y = norm.ppf(v)

        denom = np.sqrt(1 - rho**2)

        h_1_given_2 = norm.cdf((x - rho * y) / denom)
        h_2_given_1 = norm.cdf((y - rho * x) / denom)

    else:
        raise ValueError(f"Copula type {copula_name} is not implemented yet.")

    result = uv_df.copy()

    result["h_1_given_2"] = np.clip(h_1_given_2, eps, 1 - eps)
    result["h_2_given_1"] = np.clip(h_2_given_1, eps, 1 - eps)

    result["MI_1_given_2"] = result["h_1_given_2"] - 0.5
    result["MI_2_given_1"] = result["h_2_given_1"] - 0.5

    return result