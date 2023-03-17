import time
import datetime
import yfinance as yf
import pandas as pd
import polars as pol
import numpy as np
import time
import platform
from numba import njit
print("Intérprete usado para la ejecución: ", platform.python_implementation())

print("""Bienvenido al modelo "FULL ROB-DC".""")

ticker = input("""Introduzca la empresa elegida (Ticker): """)
begin_date = input("Introduce la fecha de inicio (dd/mm/yyyy): ")
end_date = input("Introduce la fecha de finalización (dd/mm/yyyy): ")
begin_date = int(time.mktime(time.strptime(begin_date, "%d/%m/%Y")))
end_date = int(time.mktime(time.strptime(end_date, "%d/%m/%Y")))
data_query = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={begin_date}&period2={end_date}&interval=1d&events=history&includeAdjustedClose=true'
df = pd.read_csv(data_query)

print("·Mostrando vista previa del Data-Frame·")
print(df.head(10))
print(df.tail(10))

print("·Reseteando índice·")
df = df.reset_index()

print("·Filtrando las columnas útiles·")
df = df[["Date","Open","High","Low","Close"]]

print("·Transformando las columnas en listas numéricas para optimizar código·")
Date = df['Date'].to_numpy()
Open = df['Open'].to_numpy()
High = df['High'].to_numpy()
Low = df['Low'].to_numpy()
Close = df['Close'].to_numpy()

print("·Creando las listas necesarias para revisar todos los escenarios posibles·")
d_before = np.arange(1,16,1)
p_high1 = np.array([0.03,0.04,0.05,0.06,0.07,0.08,0.1,0.01,0.02,0.03,0.01,0.02,0.03,0.05,0.1,0.15,0.02,0.05,0.1,0.15,0.2,0.0,0.0,0.0])
p_high2 = np.array([0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.03,0.04,0.05,0.05,0.06,0.07,0.1,0.15,0.2,0.1,0.15,0.2,0.25,0.3,0.01,0.05,0.1])
p_low1 = np.array([-0.03,-0.04,-0.05,-0.06,-0.07,-0.08,-0.1,-0.01,-0.02,-0.03,-0.01,-0.02,-0.03,-0.05,-0.1,-0.15,-0.02,-0.05,-0.1,-0.15,-0.2,0.0,0.0,0.0])
p_low2 = np.array([-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.03,-0.04,-0.05,-0.05,-0.06,-0.07,-0.1,-0.15,-0.2,-0.1,-0.15,-0.2,-0.25,-0.3,-0.01,-0.05,-0.1])
d_next = np.arange(1,16,1)
t_profit_pos = np.arange(0.01,0.105,0.005)
t_profit_neg = np.arange(-0.01,-0.105,-0.005)
t_profit = list(t_profit_pos)+ list(t_profit_neg)

print("·Definiendo funciones de: máximos y mínimos móviles, creación de listas vacías ajustadas al tamaño del Data-Frame original, tendencia de resultados (R2), error cuadrático y bucles 'for' optimizados·")
def max_rolling(a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        a_new = np.max(rolling,axis=axis)
        for i in range(len(a)-len(a_new)):
            a_new = np.insert(a_new,0,0)
        return a_new

def min_rolling(a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        a_new = np.min(rolling,axis=axis)
        for i in range(len(a)-len(a_new)):
            a_new = np.insert(a_new,0,0)
        return a_new

def create_array(x):
    a = np.zeros(len(x))
    return a

def r2score(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    results['polynomial'] = coeffs.tolist()
    correlation = np.corrcoef(x, y)[0,1]
    return correlation**2  

def mse(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.sqrt(np.square(np.subtract(actual, pred)).mean())

@njit
def for1(d1, d2, c,list,array):
    for x in range(d1, len(c)):
        list[x] = array[x-d2]
    return list

@njit
def for2(d1,c,O,H,L,ph1,ph2,pl1,pl2,s,h,l,S):
    for z in range(d1, len(c)):
        if H[z]/O[z]-1 >= ph1 and H[z]/O[z]-1 < ph2 and L[z]/O[z]-1 <= pl1 and L[z]/O[z]-1 > pl2 and s > l[z]/O[z]-1 and s < h[z]/O[z]-1:
            S[z] = 1
        else:
            S[z] = 0
    return S     

@njit
def for3(d1,d2,c,N,L):
    for y in range(d1, len(c)-d2):
        N[y] = L[y+d2]
    return N

@njit
def for4(d1,d2,c,S,R,t,s,NH,NL,NC,O):
    for r in range(d1, len(c)-d2):
        if S[r] == 0:
            R[r] = -100
        elif S[r] == 1 and t > 0 and NH[r]/(O[r]*(1+s))-1 > t:
            R[r] = t
        elif S[r] == 1 and t > 0 and NH[r]/(O[r]*(1+s))-1 < t:
            R[r] = NC[r]/(O[r]*(1+s))-1
        elif S[r] == 1 and t < 0 and NL[r]/(O[r]*(1+s))-1 < t:
            R[r] = t
        elif S[r] == 1 and t < 0 and NL[r]/(O[r]*(1+s))-1 > t:
            R[r] = (NC[r]/(O[r]*(1+s))-1)*-1    
    return R

@njit
def for5(d1,d2,c,R,r):
    for c in range(d1, len(c)-d2):
        if R[c] != -100:
            r = np.append(r, R[c])
    return r

@njit
def for6(c,m,d):
    for f in range(len(c)):
        if c[f] > m:
            m = c[f]
        d = np.append(d,c[f]-m)    
    return d

@njit
def if1(n1,n_up):
    if n1 == 0.01:
        n_up = 0.05
    elif n1 > 0.01 and n1 <= 0.05:
        n_up = 0.1   
    elif n1 > 0.05 and n1 <= 0.1:
        n_up = 0.15
    elif n1 > 0.1 and n1 <= 0.15:
        n_up = 0.2  
    elif n1 > 0.15 and n1 <= 0.2:
        n_up = 0.25
    elif n1 > 0.2:
        n_up = 0.3
    return n_up

@njit
def if2(n2,n_down):
    if n2 == -0.01:
        n_down = -0.05
    elif n2 < -0.01 and n2 >= -0.05:
        n_down = -0.1
    elif n2 < -0.05 and n2 >= -0.1:
        n_down = -0.15
    elif n2 < -0.1 and n2 >= -0.15:
        n_down = -0.2
    elif n2 < -0.15 and n2 >= -0.2:
        n_down = -0.25
    elif n2 < -0.2:
        n_down = -0.3
    return n_down                         

print("·Definiendo la función principal del cálculo (bucle)·")
def main_function():
    start = time.time()
    n_iterations = 0
    n_iterations_bs = 0
    max_f = 0
    r2_opt = 0
    error_opt = 0
    avg_trade_opt = 0
    profit_sum_opt = 0
    max_drawdown_opt = 0
    d_before_opt = 0
    p_high1_opt = 0
    p_high2_opt = 0
    p_low1_opt = 0
    p_low2_opt = 0
    signal_on_opt = 0
    d_next_opt = 0
    t_profit_opt = 0

    global best_strategies_df, start_date_list, end_date_list, d_before_opt_list, p_high1_opt_list, p_high2_opt_list, p_low1_opt_list, p_low2_opt_list, signal_on_opt_list, d_next_opt_list, t_profit_opt_list, max_f_list, r2_opt_list, error_opt_list, avg_trade_opt_list, profit_sum_opt_list, max_drawdown_opt_list
    best_strategies_df = pol.DataFrame({})
    start_date_list = np.array([])
    end_date_list = np.array([])
    d_before_opt_list = np.array([])
    p_high1_opt_list = np.array([])
    p_high2_opt_list = np.array([])
    p_low1_opt_list = np.array([])
    p_low2_opt_list = np.array([])
    signal_on_opt_list = np.array([])
    d_next_opt_list = np.array([])
    t_profit_opt_list = np.array([])
    max_f_list = np.array([])
    r2_opt_list = np.array([])
    error_opt_list = np.array([])
    avg_trade_opt_list = np.array([])
    profit_sum_opt_list = np.array([])
    max_drawdown_opt_list = np.array([])

    for d in range(len(d_before)):
        Open_Period = create_array(Close)
        High_Period_Calc = max_rolling(High, d_before[d])
        Low_Period_Calc = min_rolling(Low, d_before[d])
        High_Period = create_array(Close)
        Low_Period = create_array(Close)
        Open_Period = for1(d_before[d], d_before[d], Close, Open_Period, Open)
        High_Period = for1(d_before[d], 1, Close, High_Period, High_Period_Calc)
        Low_Period = for1(d_before[d], 1, Close, Low_Period, Low_Period_Calc)
        for ph in range(0, len(p_high1)):
            for pl in range(0, len(p_low1)):
                number_up = 0.0
                number_down = 0.0
                number_up = if1(p_high2[ph], number_up)
                number_down = if2(p_low2[pl], number_down)
                signal_on = np.arange(number_down, number_up+0.01, 0.010)
                for s in range(len(signal_on)):
                    Signal_On = create_array(Close)
                    Signal_On = for2(d_before[d], Close, Open_Period, High_Period, Low_Period, p_high1[ph], p_high2[ph], p_low1[pl], p_low2[pl], signal_on[s], High, Low, Signal_On)
                    for n in range(len(d_next)):
                        High_Lookback = max_rolling(High, d_next[n])
                        Low_Lookback = min_rolling(Low, d_next[n])
                        Next_High = create_array(Close)
                        Next_Low = create_array(Close)
                        Next_Close = create_array(Close)
                        Next_High = for3(d_before[d],d_next[n], Close, Next_High, High_Lookback)
                        Next_Low = for3(d_before[d],d_next[n], Close, Next_Low, Low_Lookback)
                        Next_Close = for3(d_before[d],d_next[n], Close, Next_Close, Close)
                        for t in range(len(t_profit)):
                            Result = create_array(Close)
                            Result = for4(d_before[d], d_next[n], Close, Signal_On, Result, t_profit[t], signal_on[s], Next_High, Next_Low, Next_Close, Open_Period)
                            result_list = np.array([])
                            result_list = for5(d_before[d],d_next[n], Close, Result, result_list)
                            if len(result_list) > 1:
                                list_cumsum = np.cumsum(result_list)
                                index_list = np.arange(1,len(list_cumsum)+1,1)
                                profit_sum = list_cumsum[len(list_cumsum)-1]
                                drawdown = np.array([])
                                max_draw = 0
                                drawdown = for6(list_cumsum, max_draw, drawdown)
                                max_drawdown = drawdown.min() 
                                avg_trade = result_list.mean()
                                x_data = np.vstack((np.ones(len(index_list)), index_list)).T
                                beta = np.linalg.inv(x_data.T.dot(x_data)).dot(x_data.T).dot(list_cumsum)
                                predict = x_data.dot(beta)
                                try:
                                    r2 = r2score(predict, list_cumsum, 1)
                                except:
                                    r2 = 0.0
                                error = mse(list_cumsum, predict)
                                max_now = r2 * profit_sum * (1-error)
                                if max_now > max_f and r2 > 0.8 and avg_trade > 0.007 and max_drawdown > -0.33:
                                    max_f = max_now
                                    r2_opt = r2
                                    error_opt = error
                                    avg_trade_opt = avg_trade
                                    max_drawdown_opt = max_drawdown
                                    profit_sum_opt = profit_sum
                                    d_before_opt = d_before[d]
                                    p_high1_opt = p_high1[ph]
                                    p_high2_opt = p_high2[ph]
                                    p_low1_opt = p_low1[pl]
                                    p_low2_opt = p_low2[pl]
                                    signal_on_opt = signal_on[s]
                                    d_next_opt = d_next[n]
                                    t_profit_opt = t_profit[t]
                                if max_now > 0 and r2 > 0.8 and avg_trade > 0.007 and max_drawdown > -0.33:    
                                    start_date_list = np.append(start_date_list, begin_date)
                                    end_date_list = np.append(end_date_list, end_date)
                                    d_before_opt_list = np.append(d_before_opt_list, d_before[d])
                                    p_high1_opt_list = np.append(p_high1_opt_list, p_high1[ph])
                                    p_high2_opt_list = np.append(p_high2_opt_list, p_high2[ph])
                                    p_low1_opt_list = np.append(p_low1_opt_list, p_low1[pl])
                                    p_low2_opt_list = np.append(p_low2_opt_list, p_low2[pl])
                                    signal_on_opt_list = np.append(signal_on_opt_list, signal_on[s])
                                    d_next_opt_list = np.append(d_next_opt_list, d_next[n])
                                    t_profit_opt_list = np.append(t_profit_opt_list, t_profit[t])
                                    max_f_list = np.append(max_f_list, max_now)
                                    r2_opt_list = np.append(r2_opt_list, r2)
                                    error_opt_list = np.append(error_opt_list, error)
                                    avg_trade_opt_list = np.append(avg_trade_opt_list, avg_trade)
                                    profit_sum_opt_list = np.append(profit_sum_opt_list, profit_sum)
                                    max_drawdown_opt_list = np.append(max_drawdown_opt_list, max_drawdown)
                                n_iterations += 1
                                et = time.time()
                                time_now = et - start
                                if n_iterations % 20000 == 0:
                                    print(f"""································································································································
                                    Tiempo transcurrido: {round(time_now/60,2)} minutos | {round(time_now,2)} segundos
                                    Número de iteraciones: {n_iterations}
                                    ································································································································
                                    --------------------------------VALORES INPUTS ACTUALES-------------------------------- 
                                    Días hacia atrás: {d_before[d]}
                                    Valor Sky 1: {round(p_high1[ph],3)}
                                    Valor Sky 2: {round(p_high2[ph],3)}
                                    Valor Abyss 1: {round(p_low1[pl],3)}
                                    Valor Abyss 2: {round(p_low2[pl],3)}
                                    Signal On: {round(signal_on[s],3)}
                                    Días hacia delante: {d_next[n]}
                                    Take Profit: {round(t_profit[t],3)}
                                    --------------------------------RESULTADOS ACTUALES--------------------------------
                                    Valor máximo: {round(max_now,3)}
                                    R2: {round(r2,3)}
                                    Error: {round(error,4)}
                                    Media por operación: {round(avg_trade,4)}
                                    Acumulado total: {round(profit_sum,3)}
                                    Drawdown máximo: {round(max_drawdown,3)}
                                    ································································································································
                                    --------------------------------MEJORES INPUTS-------------------------------- 
                                    Días hacia atrás: {d_before_opt}
                                    Valor Sky 1: {round(p_high1_opt,3)}
                                    Valor Sky 2: {round(p_high2_opt,3)}
                                    Valor Abyss 1: {round(p_low1_opt,3)}
                                    Valor Abyss 2: {round(p_low2_opt,3)}
                                    Signal On: {round(signal_on_opt,3)}
                                    Días hacia delante: {d_next_opt}
                                    Take Profit: {round(t_profit_opt,3)}
                                    --------------------------------MEJORES RESULTADOS--------------------------------
                                    Valor máximo: {round(max_f,3)}
                                    R2: {round(r2_opt,3)}
                                    Error: {round(error_opt,4)}
                                    Media por operación: {round(avg_trade_opt,4)}
                                    Acumulado total: {round(profit_sum_opt,3)}
                                    Drawdown máximo: {round(max_drawdown_opt,3)}
                                    ································································································································
                                    Tamaño del Data-Frame con las mejores estrategias guardadas: {len(max_f_list)}
                                    ································································································································
                                    """)
                            else:
                                n_iterations += 1
                                et = time.time()
                                time_now = et - start
                                if n_iterations % 20000 == 0:
                                    print(f"""································································································································
                                    Tiempo transcurrido: {round(time_now/60,2)} minutos | {round(time_now,2)} segundos
                                    Número de iteraciones: {n_iterations}
                                    ································································································································
                                    --------------------------------VALORES INPUTS ACTUALES-------------------------------- 
                                    Días hacia atrás: {d_before[d]}
                                    Valor Sky 1: {round(p_high1[ph],3)}
                                    Valor Sky 2: {round(p_high2[ph],3)}
                                    Valor Abyss 1: {round(p_low1[pl],3)}
                                    Valor Abyss 2: {round(p_low2[pl],3)}
                                    Signal On: {round(signal_on[s],3)}
                                    Días hacia delante: {d_next[n]}
                                    Take Profit: {round(t_profit[t],3)}
                                    ································································································································
                                    --------------------------------MEJORES INPUTS-------------------------------- 
                                    Días hacia atrás: {d_before_opt}
                                    Valor Sky 1: {round(p_high1_opt,3)}
                                    Valor Sky 2: {round(p_high2_opt,3)}
                                    Valor Abyss 1: {round(p_low1_opt,3)}
                                    Valor Abyss 2: {round(p_low2_opt,3)}
                                    Signal On: {round(signal_on_opt,3)}
                                    Días hacia delante: {d_next_opt}
                                    Take Profit: {round(t_profit_opt,3)}
                                    --------------------------------MEJORES RESULTADOS--------------------------------
                                    Valor máximo: {round(max_f,3)}
                                    R2: {round(r2_opt,3)}
                                    Error: {round(error_opt,4)}
                                    Media por operación: {round(avg_trade_opt,4)}
                                    Acumulado total: {round(profit_sum_opt,3)}
                                    Drawdown máximo: {round(max_drawdown_opt,3)}
                                    ································································································································
                                    Tamaño del Data-Frame con las mejores estrategias guardadas: {len(max_f_list)}
                                    ································································································································
                                    """)
                            if len(max_f_list) > 100000 and n_iterations_bs == 0:
                                n_iterations_bs += 1
                                best_strategies_df = pol.DataFrame({"Begin Date":start_date_list, "End Date":end_date_list, "Days Before":d_before_opt_list, "Sky Value 1":p_high1_opt_list, "Sky Value 2":p_high2_opt_list, "Abyss Value 1": p_low1_opt_list, "Abyss Value 2": p_low2_opt_list,"Signal On":signal_on_opt_list, "Days Next":d_next_opt_list, "Take Profit":t_profit_opt_list, "Max Value":max_f_list, "R2 Trend":r2_opt_list, "Error":error_opt_list, "AVG Trade":avg_trade_opt_list, "Profit Sum":profit_sum_opt_list, "Max Drawdown":max_drawdown_opt_list})
                                start_date_list = np.array([])
                                end_date_list = np.array([])
                                d_before_opt_list = np.array([])
                                p_high1_opt_list = np.array([])
                                p_high2_opt_list = np.array([])
                                p_low1_opt_list = np.array([])
                                p_low2_opt_list = np.array([])
                                signal_on_opt_list = np.array([])
                                d_next_opt_list = np.array([])
                                t_profit_opt_list = np.array([])
                                max_f_list = np.array([])
                                r2_opt_list = np.array([])
                                error_opt_list = np.array([])
                                avg_trade_opt_list = np.array([])
                                profit_sum_opt_list = np.array([])
                                max_drawdown_opt_list = np.array([])    
                            elif len(max_f_list) > 100000 and n_iterations_bs > 0:
                                n_iterations_bs += 1
                                best_strategies = pol.DataFrame({"Begin Date":start_date_list, "End Date":end_date_list, "Days Before":d_before_opt_list, "Sky Value 1":p_high1_opt_list, "Sky Value 2":p_high2_opt_list, "Abyss Value 1": p_low1_opt_list, "Abyss Value 2": p_low2_opt_list,"Signal On":signal_on_opt_list, "Days Next":d_next_opt_list, "Take Profit":t_profit_opt_list, "Max Value":max_f_list, "R2 Trend":r2_opt_list, "Error":error_opt_list, "AVG Trade":avg_trade_opt_list, "Profit Sum":profit_sum_opt_list, "Max Drawdown":max_drawdown_opt_list})
                                best_strategies_df = pol.concat([best_strategies_df,best_strategies],how="vertical")
                                start_date_list = np.array([])
                                end_date_list = np.array([])
                                d_before_opt_list = np.array([])
                                p_high1_opt_list = np.array([])
                                p_high2_opt_list = np.array([])
                                p_low1_opt_list = np.array([])
                                p_low2_opt_list = np.array([])
                                signal_on_opt_list = np.array([])
                                d_next_opt_list = np.array([])
                                t_profit_opt_list = np.array([])
                                max_f_list = np.array([])
                                r2_opt_list = np.array([])
                                error_opt_list = np.array([])
                                avg_trade_opt_list = np.array([])
                                profit_sum_opt_list = np.array([])
                                max_drawdown_opt_list = np.array([])     
    print("""·Fin del proceso de cálculo del modelo "ROB-DC"·""")                                

print("·Definiendo ruta del archivo csv·")
path = r'D:\ROB-DC\FULL ROB-DC\DATA.csv'

print("·Ejecutando la función principal·")
main_function()

print("·Transformando las listas numéricas en columnas de Data-Frame y guardando datos de las últimas estrategias rentables del modelo en el archivo 'DATA'·")
if len(max_f_list) > 0:
    best_strategies = pol.DataFrame({"Begin Date":start_date_list, "End Date":end_date_list, "Days Before":d_before_opt_list, "Sky Value 1":p_high1_opt_list, "Sky Value 2":p_high2_opt_list, "Abyss Value 1": p_low1_opt_list, "Abyss Value 2": p_low2_opt_list,"Signal On":signal_on_opt_list, "Days Next":d_next_opt_list, "Take Profit":t_profit_opt_list, "Max Value":max_f_list, "R2 Trend":r2_opt_list, "Error":error_opt_list, "AVG Trade":avg_trade_opt_list, "Profit Sum":profit_sum_opt_list, "Max Drawdown":max_drawdown_opt_list})
    best_strategies_df = pol.concat([best_strategies_df,best_strategies],how="vertical")
best_strategies_df.write_csv(path, sep=",")
print("·Data-frame cargado correctamente en el archivo csv·")

print("·Fin del código·")