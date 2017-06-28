'''
    prc.py
    usage: python prc.py
    plot the precision-recall curve, from the `precision` and `recall` arrays provided by the user.
'''

import matplotlib.pyplot as plt
import numpy

recall = [0.801533406353, 0.795618838992, 0.789558232932, 0.783935742972,
          0.778386272362, 0.772033588901, 0.766995253742, 0.761591821833,
          0.755239138372]
precision = [0.9801768015, 0.987493202828, 0.995305596465, 0.999255398362,
             1, 1, 1, 1, 1]

fig = plt.figure()
plt.clf()
plt.plot(recall, precision, color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.01])
plt.xlim([0.0, 1.0])

ax = fig.gca()
ax.set_xticks(numpy.arange(0, 1, 0.1))
ax.set_yticks(numpy.arange(0, 1.01, 0.1))
plt.grid()

plt.show()
