import matplotlib.pyplot as plt

x = list(range(1,11))
y1 = [93.38,88.61,85.61,88.05,88.73,91.02,89.9,89.09,90.58,91.26]
#y2 = [92.56,88.45,86.6,87.69,88.17,86.56,87.24,84.06,88.65,90.1]
y2 = [97.96,95.56,94.32,96.48,97.28,95.2,96.56,96.28,96.72,95.08]
#y2 = [88.52,94.18,93.26,95.1,95.38,94.62,95.26,94.94,94.82,94.42]
#y1 = [0.73,0.36,0.31,0.37,0.49,0.55,0.52,0.47,0.51,0.53]
#y2 = [0.71,0.32,0.30,0.35,0.46,0.36,0.43,0.28,0.42,0.46]
#y1 = [0.93,0.76,0.66,0.86,0.89,0.73,0.82,0.85,0.85,0.74]
#y2 = [0.82,0.70,0.67,0.64,0.72,0.66,0.74,0.72,0.67,0.68]
#y1 = [0.58,0.31,0.42,0.29,0.47,0.45,0.75,0.38,0.62,0.52]
#y2 = [0.47,0.22,0.32,0.26,0.40,0.45,0.69,0.32,0.46,0.39]
#y2 = [0.87,0.77,0.78,0.77,0.86,0.82,0.85,0.79,0.81,0.73]
#y2 = [0.82,0.70,0.67,0.64,0.72,0.66,0.74,0.72,0.67,0.68]
plt.bar(x,y2,label = 'MFCC',alpha = 0.5)
plt.bar(x,y1,label = 'Spectrogram',alpha = 0.5)
for i in range(1,11):
    plt.text(i,y1[i-1],str(y1[i-1]))
    plt.text(i,y2[i-1]-0.04,str(y2[i-1]))
#plt.plot(x,y1)
#plt.plot(x,y2)
plt.legend(loc = 'upper right')
plt.xlabel('Class Label')
plt.ylabel('Accuracy%')
plt.axis([0,11,80,100])
plt.xticks(x)
plt.title('Accuracy Comparison for MFCC & Spectrogram Features')
plt.show()