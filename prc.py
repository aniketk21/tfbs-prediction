recall = [1.0, 0.671875, 0.6078125, 0.5375, 0.5374968812375249, 0.5373939620758483, 0.5183289670658683, 0.5106817614770459]
precision = [0.3132182925578419, 0.75331496349344, 0.7941363432622958, 0.837259464737688, 0.8387622704686261, 0.8407826680979799, 0.8726954804900205, 0.8747061965811965]
import matplotlib.pyplot as plt
import numpy
fig = plt.figure()
plt.clf()
plt.plot(recall, precision, color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.5, 1.0])
()
ax = fig.gca()
ax.set_xticks(numpy.arange(0, 1, 0.1))
ax.set_yticks(numpy.arange(0, 1.1, 0.1))
plt.grid()

plt.show()
