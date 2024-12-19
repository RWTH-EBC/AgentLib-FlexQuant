import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tikzplotlib import get_tikz_code
from FlexibilityQuantification.flex_offer import OfferStatus


def set_mean_values(arr):
    def count_false_after_true(lst):
        count = 0
        found_true = False
        for item in lst:
            if item:
                if found_true:
                    break
                found_true = True
            elif found_true:
                count += 1
        return count

    missing_indices = np.isnan(arr)
    m = count_false_after_true(missing_indices)
    result = []
    values = arr.values[:-1]

    for i in range(0, len(values), m + 1):
        if np.isnan(values[i]):
            data = values[i:i + m + 1]
            non_nan_values = np.nan_to_num(data, nan=0)
            mean_value = np.sum(non_nan_values) / m
            result.append(mean_value)
            result.extend(data[1:])
        else:
            result.extend(arr[i:i + m + 1])

    return result


_CONVERSION_MAP = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}


def get_series_from_predictions(series, convert_to="seconds", fname=None, return_first=False, index_of_return=0,
                                handle_col_vals=False):
    actual_values: dict[float, float] = {}
    if fname is not None:
        f = open(fname, "w+")
    for i, (time, prediction) in enumerate(series.groupby(level=0)):
        time = time / _CONVERSION_MAP[convert_to]
        if handle_col_vals:
            try:
                prediction = pd.Series(set_mean_values(prediction), index=prediction[time].index[:-1] + time)
            except ValueError:
                prediction = pd.Series(set_mean_values(prediction), index=prediction[time].index + time)

        else:
            prediction: pd.Series = prediction.dropna().droplevel(0)
            prediction.index += time
        if return_first:
            if i == index_of_return:
                return prediction
            else:
                continue
        if fname is not None:
            f.write(f"{time} {prediction} \n")
        actual_values[time] = prediction.iloc[0]
        prediction.index = (prediction.index + time) / _CONVERSION_MAP[convert_to]
    if fname is not None:
        f.close()
    return pd.Series(actual_values)


def plot_temperature_simulation(results, fmu=False, size=(6.4, 4.8), convert_to="seconds", tikz=False):
    fig, ax = plt.subplots(figsize=size)
    t_upper = get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['parameter']["T_upper"])
    t_lower = get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['parameter']["T_lower"])
    if fmu:
        arr = results["SimAgent"]["SimTestHall"]["TAirRoom"]
    else:
        arr = results["SimAgent"]["SimTestHall"]["T_out"]
    arr.index = arr.index / _CONVERSION_MAP[convert_to]
    t_lower.index = t_lower.index / _CONVERSION_MAP[convert_to]
    t_upper.index = t_upper.index / _CONVERSION_MAP[convert_to]

    ax.plot(arr, color="black")
    ax.plot(get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['variable']["T_Air"]), color="red")
    ax.plot(t_upper, linestyle="dashed", color="red")
    ax.plot(t_lower, linestyle="dashed", color="blue")
    ax.set_ylabel("Room Temperature [K]")
    ax.legend(["Simulation", "Model"])
    if not tikz:
        return fig
    else:
        return get_tikz_code(fig)


def plot_power_simulation(results, fmu=False, detailed=False, size=(6.4, 4.8), convert_to="seconds", tikz=False):
    if detailed:
        fig, axes = plt.subplots(4, figsize=size)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=size)
    if fmu:
        arr = 1000 * (results["SimAgent"]["SimTestHall"]["Q_Ahu"] + results["SimAgent"]["SimTestHall"]["Q_Tabs"]) / \
              get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['parameter']['COP']).iloc[0]
    else:
        arr = results["SimAgent"]["SimTestHall"]["P_el_c"]
    arr_mod = get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['variable']['P_el_c'])
    # preds = [get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['variable']['P_el_c'], handle_col_vals=True, return_first=True, index_of_return=i) for i in range(0, 15)]
    # for pred in preds:
    #     pred.index = pred.index / _CONVERSION_MAP[convert_to]
    #     ax.plot(pred)
    arr.index = arr.index / _CONVERSION_MAP[convert_to]
    arr_mod.index = arr_mod.index / _CONVERSION_MAP[convert_to]

    ax.plot(arr, color="black")
    ax.plot(arr_mod, color="red", drawstyle="steps-post")

    for time, flex_res in results["FlexibilityMarket"]["FlexibilityMarket"].groupby(level=0):
        missing_indices = results["NegFlexMPC"]['NegFlexMPC']['variable']['P_el_c'][time][
            np.isnan(results["NegFlexMPC"]['NegFlexMPC']['variable']['P_el_c'][time])].index
        time_step = missing_indices[1] - missing_indices[0]

        if flex_res["Status"].iloc[0] == OfferStatus.accepted_positive.value:
            profile = flex_res["Base Profile"][time] - flex_res["Positive Profile"][time] * \
                      flex_res["Power Multiplier"][time].iloc[0]
        elif flex_res["Status"].iloc[0] == OfferStatus.accepted_negative.value:
            profile = flex_res["Base Profile"][time] + flex_res["Negative Profile"][time] * \
                      flex_res["Power Multiplier"][time].iloc[0]

        elif flex_res["Status"].iloc[0] == OfferStatus.not_accepted.value:
            continue

        profile[profile.index[-1] + time_step] = profile[profile.index[-1]]
        profile.index += time
        profile.index = profile.index / _CONVERSION_MAP[convert_to]

        ax.plot(profile, color="blue", drawstyle="steps-post")
        ax.set_ylabel("Leistung [W]")
    if detailed:
        preds = [
            get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['variable']['P_el_c'], handle_col_vals=True,
                                        return_first=True, index_of_return=i) for i in range(0, 24, 6)]
        for pred in preds:
            pred.index = pred.index / _CONVERSION_MAP[convert_to]
            axes[0].plot(pred)
        arr = 1000 * results["SimAgent"]["SimTestHall"]["Q_Tabs"]
        arr.index = arr.index / _CONVERSION_MAP[convert_to]

        axes[1].plot(arr, color="black")
        axes[1].plot(get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['variable']['Q_Tabs_set_del'],
                                                 convert_to=convert_to, handle_col_vals=True), color="red",
                     drawstyle="steps-post")
        axes[1].set_ylabel("Q_Tabs")
        preds = [get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['variable']['Q_Tabs_set_del'],
                                             handle_col_vals=True, return_first=True, index_of_return=i) for i in
                 range(0, 24, 6)]
        for pred in preds:
            pred.index = pred.index / _CONVERSION_MAP[convert_to]
            axes[1].plot(pred)

        arr = 1000 * results["SimAgent"]["SimTestHall"]["Q_Ahu"]
        arr.index = arr.index / _CONVERSION_MAP[convert_to]
        axes[2].plot(arr, color="black")
        axes[2].plot(
            get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['variable']['Q_Ahu'], convert_to=convert_to,
                                        handle_col_vals=True), color="red", drawstyle="steps-post")
        axes[2].set_ylabel("Q_Ahu")
        preds = [
            get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['variable']['Q_Ahu'], handle_col_vals=True,
                                        return_first=True, index_of_return=i) for i in range(0, 15)]
        for pred in preds:
            pred.index = pred.index / _CONVERSION_MAP[convert_to]
            axes[2].plot(pred)

        axes[3].plot(
            get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['parameter']['r_pel'], convert_to=convert_to),
            color="black", drawstyle="steps-post")
        axes[3].set_ylabel("K_el")
    ax.legend(["Simulation", "Model", "Abrufsprofil"])
    if not tikz:
        return fig
    else:
        return get_tikz_code(fig)


def plot_prediction(results, times=[], size=(6.4, 4.8), detailed=False, tikz=False):
    figs = []
    if times == []:
        times = results["PosFlexMPC"]['PosFlexMPC']['variable']['P_el_c'].index.levels[0]
    for time in times:
        if detailed:
            f, a = plt.subplots(4, figsize=size)
        else:
            f, a = plt.subplots(2, figsize=size)

        f.suptitle(f"Power Prediction on t={time}s")
        t_upper = results["myMPCAgent"]['FlexMPC']['parameter']['T_upper'][time].dropna()
        t_lower = results["myMPCAgent"]['FlexMPC']['parameter']['T_lower'][time].dropna()
        a[1].plot(t_upper, linestyle="dashed", color="red")
        a[1].plot(t_lower, linestyle="dashed", color="blue")
        a[0].plot(results["myMPCAgent"]['FlexMPC']['variable']['P_el_c'][time].dropna(), color="black",
                  drawstyle="steps-post")
        a[0].plot(results["PosFlexMPC"]['PosFlexMPC']['variable']['P_el_c'][time].dropna(), color="red",
                  drawstyle="steps-post")
        a[0].plot(results["NegFlexMPC"]['NegFlexMPC']['variable']['P_el_c'][time].dropna(), color="blue",
                  drawstyle="steps-post")
        a[0].legend(["Baseline", "Positiv", "Negativ"])
        a[0].set_ylabel("Leistung [W]")

        missing_indices = np.isnan(results["NegFlexMPC"]['NegFlexMPC']['variable']['P_el_c'][time])
        a[1].plot(results["myMPCAgent"]['FlexMPC']['variable']['T_Air'][time][missing_indices], color="black")
        a[1].plot(results["PosFlexMPC"]['PosFlexMPC']['variable']['T_Air'][time][missing_indices], color="red")
        a[1].plot(results["NegFlexMPC"]['NegFlexMPC']['variable']['T_Air'][time][missing_indices], color="blue")
        a[1].set_ylabel("Lufttemperatur [K]")
        if detailed:
            a[2].plot(results["myMPCAgent"]['FlexMPC']['variable']['Q_Tabs_set'][time][missing_indices], color="black")
            a[2].plot(results["PosFlexMPC"]['PosFlexMPC']['variable']['Q_Tabs_set'][time][missing_indices], color="red")
            a[2].plot(results["NegFlexMPC"]['NegFlexMPC']['variable']['Q_Tabs_set'][time][missing_indices], color="blue")
            a[2].set_ylabel("Q TABS")
            a[3].plot(results["myMPCAgent"]['FlexMPC']['variable']['T_ahu_set'][time][missing_indices], color="black")
            a[3].plot(results["PosFlexMPC"]['PosFlexMPC']['variable']['T_ahu_set'][time][missing_indices], color="red")
            a[3].plot(results["NegFlexMPC"]['NegFlexMPC']['variable']['T_ahu_set'][time][missing_indices], color="blue")
            a[3].set_ylabel("Q AHU")
        figs.append(f)

    return figs


### set xticklabels
### fix axis names
def plot_flex_price_graph(result_wo_flex, flex_results_list, flex_call_time, electricity_prices, size=(6.4, 4.8),
                          tikz=False, fmu=False):
    # since we dont care about the accepted profile, its not important here which flex_result we use
    offer = flex_results_list[0]['FlexibilityMarket']['FlexibilityMarket'].loc[flex_call_time]
    if offer["Status"].iloc[0] == OfferStatus.accepted_positive.value:
        flex_type = "Positive"
    elif offer["Status"].iloc[0] == OfferStatus.accepted_negative.value:
        flex_type = "Negative"
    else:
        raise ValueError(f"No offer is accepted on {flex_call_time}")
    time_step = offer.index[1] - offer.index[0]
    full_price = offer[f"{flex_type} Price"].iloc[0]
    en_flex_base_kWh = sum(offer[f"{flex_type} Profile"]) * time_step / 3.6e6 * (1 if flex_type == "Positive" else -1)
    fig, ax = plt.subplots(figsize=size)

    base_profile = None
    scatter_list = []
    for res in flex_results_list:
        if not fmu:
            p_el_with = res["SimAgent"]["SimTestHall"]["P_el_c"]
        else:
            p_el_with = 1000 * (res["SimAgent"]["SimTestHall"]["Q_Ahu"] + res["SimAgent"]["SimTestHall"]["Q_Tabs"]) / \
                        get_series_from_predictions(res["myMPCAgent"]['FlexMPC']['parameter']['COP']).iloc[0]

        if base_profile is None:
            base_profile = offer["Base Profile"]
            base_profile.index += flex_call_time
            end_of_flex = base_profile.index[-1] + time_step
            base_profile = base_profile.reindex(p_el_with.index).fillna(method="ffill")
            base_profile = base_profile.loc[:end_of_flex].dropna()
            sample_time = base_profile.index[1] - base_profile.index[0]
        if fmu:
            p_el_wo = 1000 * (
                        result_wo_flex["SimAgent"]["SimTestHall"]["Q_Ahu"] + result_wo_flex["SimAgent"]["SimTestHall"][
                    "Q_Tabs"]) / \
                      get_series_from_predictions(result_wo_flex["myMPCAgent"]['FlexMPC']['parameter']['COP']).iloc[0]
        else:
            p_el_wo = result_wo_flex["SimAgent"]["SimTestHall"]["P_el_c"]

        price_wo_flex = sum((electricity_prices * p_el_wo).dropna()) * sample_time / 3.6e6
        price_with_flex = sum((electricity_prices * p_el_with).dropna()) * sample_time / 3.6e6
        if flex_type == "Negative":
            en_flex = -sum((p_el_with - base_profile).dropna()) * sample_time / 3.6e6
        else:
            en_flex = sum((base_profile - p_el_with).dropna()) * sample_time / 3.6e6

        scatter_list.append([en_flex, price_with_flex - price_wo_flex - 50])
    if not tikz:
        ax.plot([0, en_flex_base_kWh], [0, full_price], color="black")
        for scatter in scatter_list:
            ax.scatter(*scatter, color="red", marker="x")
        ax.set_xlabel("Energieflexibilität [kWh]")
        ax.set_ylabel("Kosten von Abruf [ct]")
        ax.legend(["Vorhersage", "Abruf"])
        return fig

    else:
        tikz_str = r"""
\begin{tikzpicture}

\begin{axis}[
legend cell align={left},
legend style={fill opacity=0.8, draw opacity=1, text opacity=1, draw=white!80!black},
tick align=outside,
tick pos=left,
x grid style={white!69.0196078431373!black},
xlabel={Energieflexibilität [kWh]},
"""
    tikz_str += f"xmin={min(scatter_list, key=lambda x: x[0])[0] * 1.1}, xmax={max(scatter_list, key=lambda x: x[0])[0] * 1.1},"
    tikz_str += """
xtick style={color=black},
y grid style={white!69.0196078431373!black},
ylabel={Kosten von Abruf [ct]},
"""

    tikz_str += f"ymin={min(scatter_list, key=lambda x: x[1])[1] * 1.05}, ymax={max(scatter_list, key=lambda x: x[1])[1] * 1.05},\n"
    tikz_str += "ytick style={color=black},\n]\n"
    tikz_str += r"\addplot [semithick, black]"
    tikz_str += "\ntable {%\n0 0\n"
    tikz_str += f"{en_flex_base_kWh} {full_price}\n"
    tikz_str += "};\n"
    tikz_str += r"\addlegendentry{Vorhersage}"
    tikz_str += "\n"
    tikz_str += r"\addplot [draw=none, color=red, mark=star, mark size=4, very thick, only marks]"
    tikz_str += "\ntable {%\nx y\n"
    for scatter in scatter_list:
        tikz_str += f"{scatter[0]} {scatter[1]}\n"
    tikz_str += "};\n"
    tikz_str += r"\addlegendentry{Abruf}"
    tikz_str += "\n"
    tikz_str += r"\end{axis}"
    tikz_str += "\n"
    tikz_str += r"\end{tikzpicture}"
    return tikz_str


def plot_flex_amount(results, size=(6.4, 4.8), convert_to="seconds", tikz=False, fmu=False):
    x_vals = results["PosFlexMPC"]['PosFlexMPC']['variable']['P_el_c'].index.levels[0]
    data = {x / _CONVERSION_MAP[convert_to]: [0, 0] for x in x_vals}
    fig, ax = plt.subplots(figsize=size)

    for time, offer in results['FlexibilityMarket']['FlexibilityMarket'].groupby(level=0):
        time_step = offer.loc[time].index[1] - offer.loc[time].index[0]

        if offer["Status"].iloc[0] == OfferStatus.accepted_positive.value:
            flex_type = "Positive"
            en_flex_offered = sum(offer.loc[time]["Positive Profile"]) * time_step / 3.6e6
        elif offer["Status"].iloc[0] == OfferStatus.accepted_negative.value:
            flex_type = "Negative"
            en_flex_offered = -sum(offer.loc[time]["Negative Profile"]) * time_step / 3.6e6

        else:
            continue
        try:
            base_profile = offer.loc[time]["Base Profile"]
            base_profile.index += time
            end_of_flex = base_profile.index[-1] + time_step
            if not fmu:
                profile = results["SimAgent"]["SimTestHall"]["P_el_c"]
            else:
                profile = (results["SimAgent"]["SimTestHall"]["Q_Ahu"] + results["SimAgent"]["SimTestHall"]["Q_Tabs"]) / \
                          get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['parameter']['COP']).iloc[0]

            base_profile = base_profile.reindex(profile.index).fillna(method="ffill")
            base_profile = base_profile.loc[:end_of_flex].dropna()
            sample_time = base_profile.index[1] - base_profile.index[0]
            if flex_type == "Negative":
                en_flex_real = sum((profile - base_profile).dropna()) * sample_time / 3.6e6
            else:
                en_flex_real = sum((base_profile - profile).dropna()) * sample_time / 3.6e6
            time = time / _CONVERSION_MAP[convert_to]
            data[time] = (en_flex_offered, en_flex_real)
        except IndexError:
            continue
    positive_flex = get_series_from_predictions(
        results["FlexibilityIndicator"]["FlexibilityIndicator"]["energyflex_pos"], convert_to=convert_to)
    negative_flex = - get_series_from_predictions(
        results["FlexibilityIndicator"]["FlexibilityIndicator"]["energyflex_neg"], convert_to=convert_to)
    if not tikz:
        for time, val in data.items():
            ax.bar(time + 2000 / _CONVERSION_MAP[convert_to], val[0], width=4000 / _CONVERSION_MAP[convert_to],
                   color="blue")
            ax.bar(time - 2000 / _CONVERSION_MAP[convert_to], val[1], width=4000 / _CONVERSION_MAP[convert_to],
                   color="red")

        ax.plot(positive_flex, color="rosybrown", linewidth=0.5)
        ax.plot(negative_flex, color="cornflowerblue", linewidth=0.5)
        ax.legend(
            ["Energieflexibilität Positiv", "Energieflexibilität Negative", "Angebotsmenge", "Abgerufene Flexibilität"])
        ax.set_xlabel('Zeit')
        ax.set_ylabel('Energieflexibilität [kWh]')
        ax.axhline(y=0, color="black", linewidth=0.5)

        return fig
    else:
        tikz_str = r"""
\begin{tikzpicture}

\definecolor{cornflowerblue}{RGB}{100,149,237}
\definecolor{darkgray176}{RGB}{176,176,176}
\definecolor{lightgray204}{RGB}{204,204,204}
\definecolor{rosybrown}{RGB}{188,143,143}

\begin{axis}[
%new
width=0.8\textwidth, height=200,
legend cell align={bottom},
%new
legend style={
fill opacity=0,
draw opacity=1,
text opacity=1,
at={(0.5,-0.25)},
anchor=north
},
%new
legend columns = 2,
tick align=outside,
tick pos=left,
x grid style={darkgray176},
xlabel={Zeit},"""
        tikz_str += f"\nxmin={min(negative_flex.index) * 0.9}, xmax={max(negative_flex.index)},"
        tikz_str += "\nxtick style={color=black},"
        tikz_str += "\ny grid style={darkgray176},"
        tikz_str += "\nylabel={Energieflexibilität [kWh]},"

        tikz_str += f"\nymin={min(negative_flex) * 1.1}, ymax={max(positive_flex) * 1.1},"
        tikz_str += "\nytick style={color=black}]"
        for time, val in data.items():
            tikz_str += f"\n\draw[draw=none,fill=blue] (axis cs:{time},0) rectangle (axis cs:{time + 4000 / _CONVERSION_MAP[convert_to]},{val[0]});"
            tikz_str += f"\n\draw[draw=none,fill=red] (axis cs:{time},0) rectangle (axis cs:{time - 4000 / _CONVERSION_MAP[convert_to]},{val[1]});"
        tikz_str += "\n"
        tikz_str += r"\addplot [very thin, rosybrown]"
        tikz_str += "\ntable {"

        for ind, val in positive_flex.items():
            tikz_str += f"\n {ind} {val}"
        tikz_str += "\n};\n"
        tikz_str += r"\addlegendentry{Vorhersage Positiv}"
        tikz_str += "\n"

        tikz_str += r"\addplot [very thin, cornflowerblue]"
        tikz_str += "\ntable {"

        for ind, val in negative_flex.items():
            tikz_str += f"\n {ind} {val}"
        tikz_str += "\n};\n"
        tikz_str += r"\addlegendentry{Vorhersage Negativ}"
        tikz_str += "\n"

        tikz_str += r"\addplot [very thin, black, forget plot]"
        tikz_str += "\ntable {%\n"
        tikz_str += f"{min(negative_flex.index) * 0.9} 0\n"
        tikz_str += f"{max(negative_flex.index)} 0\n"

        tikz_str += "};\n"
        tikz_str += r"""
\addlegendimage{blue,fill=blue,line width=1mm}
\addlegendentry{Angebotmenge}
\addlegendimage{red,fill=red,line width=1mm}
\addlegendentry{Abrufmenge}

\end{axis}

\end{tikzpicture}
"""
        return tikz_str


def save_tikz(fig, fpath):
    with open(fpath, "w+") as f:
        f.write(get_tikz_code(fig))


def debug_print(results):
    ist_wert = 1000 * results["SimAgent"]["SimTestHall"]["Q_Tabs"]
    ist_wert_del = results["SimAgent"]["SimTestHall"]["Q_tabs_del"]
    soll_wert = get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['variable']["Q_Tabs_set"])
    delay_wert = get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['variable']["Q_Tabs_set_del"])
    f, a = plt.subplots(3)
    a[0].plot(ist_wert, color="black")
    a[0].plot(ist_wert_del, color="green")
    a[0].plot(soll_wert, color="red", drawstyle="steps-post")
    a[0].plot(delay_wert, color="blue")
    a[0].set_ylabel("Q Tabs")
    a[0].legend(["SIM", "SIM_DEL", "SOLL", "DEL"])

    ist_wert = 1000 * results["SimAgent"]["SimTestHall"]["Q_Ahu"]
    soll_wert = get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['variable']["Q_Ahu"])
    a[1].plot(ist_wert, color="black")
    a[1].plot(soll_wert, color="red", drawstyle="steps-post")
    a[1].legend(["SIM", "OPT"])
    a[1].set_ylabel("Q ahu")

    ist_wert = get_series_from_predictions(results["myMPCAgent"]['FlexMPC']['variable']["T_Tabs"])
    soll_wert = results["SimAgent"]["SimTestHall"]["T_Tabs"]
    a[2].plot(ist_wert, color="black")
    a[2].plot(soll_wert, color="red", drawstyle="steps-post")
    a[2].legend(["SIM", "OPT"])
    a[2].set_ylabel("T Tabs")

    return f


def debug_print_mult_sim(results_fmu, results, constants_list):
    ist_wert_del = results_fmu["SimAgent"]["SimTestHall"]["Q_tabs_del"]
    soll_wert = results_fmu["SimAgent"]["SimTestHall"]["QFlowTabsSet"]
    f, a = plt.subplots()
    a.plot(soll_wert, color="red", drawstyle="steps-post")
    a.plot(ist_wert_del, color="black")
    legend = ["input", "FMU"]
    for res, const in zip(results, constants_list):
        ist_wert_del = res["SimAgent"]["SimTestHall"]["q_tabs_del_out"]
        a.plot(ist_wert_del)
        legend.append(const)

    a.legend(legend)
    return f
