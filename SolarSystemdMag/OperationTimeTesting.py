#Operation Time Testing


import time
import numpy as np
x = np.random.rand(10**8)
start =time.time()
ltp5 = np.where(x<0.5)[0]
stop=time.time()
print(stop-start)


import time
import numpy as np
x = np.random.rand(10**8)
start =time.time()
ltp5 = (x<0.5)
stop=time.time()
print(stop-start)

import time
import numpy as np
x = np.random.rand(10**8)
start =time.time()
ltp5 = (x<0.5)*(x<0.3)
stop=time.time()
print(stop-start)

import time
import numpy as np
x = np.random.rand(10**8)
start =time.time()
ltp5 = (x<0.5)*(x<0.3)*(x<0.1)
stop=time.time()
print(stop-start)

import time
import numpy as np
x = np.random.rand(10**8)
start =time.time()
ltp5 = (x<0.5)*(x<0.3)*(x<0.1)*(x<0.05)
stop=time.time()
print(stop-start)
