"""Analysis algorithms"""

from psrqpy import QueryATNF

def query_profiles_metadata(df, save=True, filename='atnfquery.pkl'):
    """Query ATNF database for profiles metadata

    Parameters
    ----------
    df: pandas.DataFrame

    save:Boolean (optional)
        Save query to pickle file (filename)
    filename:str
        Output filename (optional, default: atnfquery.pkl)
    """
    return QueryATNF(
        params=['JNAME', 'RAJ', 'DECJ', 'P0', 'P1', 'ASSOC', 'BINARY', 'TYPE', 'P1_I'],
        psrs=[row['name'] for index, row in df.iterrows()]
    )

    if save:
        query.save(filename)

def load_query_result(filename='atnfquery.pkl'):
    """Load ATNF database Query results

    Parameters
    ----------
    filename:str
        Input filename (default: atnfquery.pkl)
    """
    return QueryATNF(loadquery=filename)
