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
        h_1_given_2 = t.cdf((x - rho * y) / scale_x_given_y, df=df + 1)

        scale_y_given_x = np.sqrt((df + x**2) * (1 - rho**2) / (df + 1))
        h_2_given_1 = t.cdf((y - rho * x) / scale_y_given_x, df=df + 1)

    elif copula_name == "gaussian":
        rho = params["rho"]

        x = norm.ppf(u)
        y = norm.ppf(v)

        denom = np.sqrt(1 - rho**2)
        h_1_given_2 = norm.cdf((x - rho * y) / denom)
        h_2_given_1 = norm.cdf((y - rho * x) / denom)

    elif copula_name == "clayton":
        theta = params["theta"]

        a = u ** (-theta) + v ** (-theta) - 1

        h_1_given_2 = (a ** (-1 / theta - 1)) * (v ** (-theta - 1))
        h_2_given_1 = (a ** (-1 / theta - 1)) * (u ** (-theta - 1))

    elif copula_name == "gumbel":
        theta = params["theta"]

        lu = -np.log(u)
        lv = -np.log(v)

        a = lu ** theta + lv ** theta
        C = np.exp(-(a ** (1 / theta)))

        h_1_given_2 = C * (a ** (1 / theta - 1)) * (lv ** (theta - 1)) / v
        h_2_given_1 = C * (a ** (1 / theta - 1)) * (lu ** (theta - 1)) / u

    elif copula_name == "frank":
        theta = params["theta"]

        if abs(theta) < 1e-8:
            h_1_given_2 = u.copy()
            h_2_given_1 = v.copy()
        else:
            e_theta = np.exp(-theta)
            e_tu = np.exp(-theta * u)
            e_tv = np.exp(-theta * v)

            denom = (e_theta - 1) + (e_tu - 1) * (e_tv - 1)

            h_1_given_2 = e_tv * (e_tu - 1) / denom
            h_2_given_1 = e_tu * (e_tv - 1) / denom

    else:
        raise ValueError(f"Copula type '{copula_name}' is not implemented.")

    result = uv_df.copy()

    result["h_1_given_2"] = np.clip(h_1_given_2, eps, 1 - eps)
    result["h_2_given_1"] = np.clip(h_2_given_1, eps, 1 - eps)

    result["MI_1_given_2"] = result["h_1_given_2"] - 0.5
    result["MI_2_given_1"] = result["h_2_given_1"] - 0.5

    return result