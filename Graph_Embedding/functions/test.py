import warnings

warnings.filterwarnings('ignore')
from data_process import *
#  features: X (n × d); adjacency: similarity matrix; labels: Y
#  Parameters that need to be entered manual

path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
path = path + "/Datasets/Graph_Datasets/"
print(path)
