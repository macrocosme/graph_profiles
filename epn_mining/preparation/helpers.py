METRICS = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean',
                      'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
                      'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
                      'wminkowski', 'yule']

def download_epn_database():
    import os
    print ('Downloading fits files from EPN database. This will take a moment. Please be patient.')
    os.system('wget http://www.epta.eu.org/epndb/psrfits/ --recursive -A .fits -l 10 --no-parent --directory-prefix=../')
    print ('Done.')

def scrape_new_epn(url = 'http://www.epta.eu.org/epndb/list.php'):
    import requests
    import urllib.request
    import time
    from bs4 import BeautifulSoup
    import os.path

    def fetch(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        return response, soup

    url = url
    base_path = os.path.split(url)[0] + '/'

    response, soup = fetch(url)

    epn_db = {}

    for a in soup.findAll('a'):
        current_pulsar = a.contents[0]
        print (current_pulsar)
        epn_db[current_pulsar] = {}
        new_url = base_path + a['href']
        epn_db[current_pulsar]['url'] = new_url
        pulsar_response, pulsar_soup = fetch(new_url)
        time.sleep(1)

        for aa in pulsar_soup.findAll('a'):
            if '.sum.' in aa:
                ref = aa['href'].split('/')[-1].split('.')[0]
                epn_db[current_pulsar][ref] = {}
                epn_db[current_pulsar][ref]['url'] = base_path + current_pulsar + '/' + aa['href']

                summary_response, summary_soup = fetch(new_url)
                time.sleep(1)

                epn_db[current_pulsar][ref]['html'] = summary_soup

    return epn_db

def scrape_old_epn(url='http://www.jb.man.ac.uk/research/pulsar/Resources/epn/psrlist.html'):
    import requests
    import urllib.request
    import time
    from bs4 import BeautifulSoup
    import os.path

    def fetch(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        return response, soup

    url = url
    base_path = os.path.split(url)[0] + '/'

    response, soup = fetch(url)

    epn_db = {}

    for a in soup.findAll('a'):
        current_pulsar = a.contents[0]
        print (current_pulsar)
        epn_db[current_pulsar] = {}
        new_url = base_path + a['href']
        epn_db[current_pulsar]['url'] = new_url
        pulsar_response, pulsar_soup = fetch(new_url)
        time.sleep(1)

        for aa in pulsar_soup.findAll('a'):
            if '.sum.' in aa:
                ref = aa['href'].split('/')[-1].split('.')[0]
                epn_db[current_pulsar][ref] = {}
                epn_db[current_pulsar][ref]['url'] = base_path + current_pulsar + '/' + aa['href']

                summary_response, summary_soup = fetch(new_url)
                time.sleep(1)

                epn_db[current_pulsar][ref]['html'] = summary_soup

    return epn_db


class Evaluation:
    def __init__(self, tp, tn, fp, fn):
        """
        tp: true positive
        tn: true negative
        fn: false positive
        fn: false negative
        """
        self.accuracy = lambda tp, tn, fp, fn: (tp+tn)/(tp+tn+fp+fn)
        self.recall = lambda tp, fn: tp/(tp+fn)
        self.precision = lambda tp, fp: tp/(tp+fp)
        self.f_measure = (2*self.recall*self.presision)/(self.recall+self.presition)
