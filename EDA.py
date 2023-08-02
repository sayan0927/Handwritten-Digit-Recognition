



from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np



(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("\n\n")




fig, axs = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axs.flatten()):
    ax.hist(y_train[y_train==i])
    
    ax.set_title(f'Digit {i}')
    

   # print(y_train[i],end=' ')


# set the overall title for the figure
fig.suptitle('First 10 digits', fontsize=16)

# adjust spacing between subplots
plt.tight_layout()

# show the figure
plt.show()

print(X_train[0].shape)

for i in range(0,27):
    for j in range(0,27):
        print(X_train[1][i][j],end=" ")
    print("\n")
    
    
    


# Set the positions and labels of the ticks on the x-axis
   
plt.hist(y_train[(y_train==0) | (y_train==1) | (y_train==2)| (y_train==3)| (y_train==4) ],align='mid')
plt.xlabel('Digit')
plt.ylabel('Frequency')
plt.show()
plt.hist(y_train[(y_train==5) | (y_train==6) | (y_train==7)| (y_train==8)| (y_train==9) ],align='mid')
plt.xlabel('Digit')
plt.ylabel('Frequency')
plt.show()