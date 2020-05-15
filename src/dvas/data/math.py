"""
Module containing classes and function for specialized math operation on
dataframes and series

"""


def crosscorr(datax, datay, lag=0, wrap=False, method='kendall'):
    """Lag-N cross correlation.
    If wrap is False, shifted data are filled with NaNs.

    Args:
        datax, datay (pandas.Series): Must be of equal length
        lag(`obj`:int, optional):  Default is 0
        wrap (`obj`:bool, optional:  (Default value = False)
        method (`obj`:str, optional): kendall, pearson, spearman.
            Default is 'kendall'

    Returns:
        float

    """

    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        out = datax.corr(shiftedy, method=method)
    else:
        out = datax.corr(datay.shift(lag), method=method)

    return out
