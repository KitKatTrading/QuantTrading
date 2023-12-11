DEBUG_PRINT = False
def debug_logging(message, debug_print=DEBUG_PRINT):
    if debug_print:
        print(message)

def propagate_values(df, len_prop):
    i = 0
    while i < len(df):
        if df.iloc[i]['value'] != 0:
            value = df.iloc[i]['value']
            j = 1
            while j < len_prop + 1 and i + j < len(df) and df.iloc[i + j]['value'] == 0:
                df.at[df.index[i + j], 'value'] = value
                j += 1
            i += j
        else:
            i += 1
