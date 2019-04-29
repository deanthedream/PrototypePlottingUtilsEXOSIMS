
#tunnel atuin port 3306 to local 3306 using on local machine:
#ssh -g -L 3306:localhost:3306 -f -N drk94@atuin.coecis.cornell.edu
#

#pip install sqlalchemy
#pip install pymysql
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://drk94@127.0.0.1/dsavrans_plandb',echo=False)

result = engine.execute('SELECT * from Aliases')
data = result.fetchall()     


import numpy as np
arrayData = np.vstack(data)

import pickle
pklPath = './alias_4_11_2019.pkl'
with open(pklPath, 'wb') as f: pickle.dump(np.vstack(data),f)

#Boom array of aliases saved to computer
#col0: index, col1: Names, col2: actual index of alias (stars with same names have the same index here)