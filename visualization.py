

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


"""
This module visualizes the thermodynamic quantities associated with the learned information content in the context of Energy-Based Models (EBM) and Joint Energy-Based Models (JEM). 

 The thermodynamic quantities visualized include:
- $W$: work on the marginal model.
- $Q$: Heat flow from the marginal model.
- $W_J$: Work on the joint model.
- $Q_J$: Heat flow from the joint model.
- $F_J$: Joint free energy.
- $F_M$: Marginal free energy.

"""


def plot_thermo(phase=False, cc=0 ,t=-1):

  w   = np.asanyarray(trainer.thermo)[:,0]
  Q   = np.asanyarray(trainer.thermo)[:,1]
  j_w = np.asanyarray(trainer.thermo)[:,2]
  j_Q = np.asanyarray(trainer.thermo)[:,3]


  F_j  =  np.asanyarray(trainer.thermo)[:,7]
  F_m  =  np.asanyarray(trainer.thermo)[:,9]
  fig_size = (4.77376504773765, 2.9503490538081323)
  full_fig_size = (2* 4.77376504773765,2* 2.9503490538081323)
  x_1, x_2 = np.arange(0,(epoch+1)*1000,5000) , np.arange(0,epoch+1,5)

  fig, ax = plt.subplots(figsize=fig_size)  # Set figure size

  # Save the outputs of plot functions to variables
  line1, = plt.plot(w[:t], label='$W_M$', color='#1f77b4' , linewidth=2)
  line2, = plt.plot(Q[:t], label='$Q_M$', color='orange', linewidth=2)

  line3, = plt.plot(j_w[:t], label='$W_J$ ', color='#1f77b4', linewidth=4)
  line4, = plt.plot(j_Q[:t], label='$Q_J$', color='orange', linewidth=4 )

  plt.title('Energy flows')
  plt.xlabel('time | epoch')
  plt.ylabel('energy | info ')

  if phase== True : plt.axvline(x=cc, color='g', linestyle=':', alpha=0.6)

  plt.xticks(x_1, x_2 )

  # Create two separate legends
  legend1 = plt.legend([line1, line2], ['$W_M$', '$Q_M$'], loc='upper left', fontsize="10")
  legend2 = plt.legend([line3, line4], ['$W_J$', '$Q_J$'], loc='lower left', fontsize="10")

  plt.gca().add_artist(legend1)  # Add legend1 back after it's removed by creating legend2
  plt.tight_layout()
  plt.show()



  fig, ax = plt.subplots(figsize=fig_size)  # Set figure size
  # joint reversiblity
  plt.plot(( F_j )[:t] , label = '$\Delta F_J$' , color = 'orange' , linewidth=4)

  plt.plot((j_w)[:t] , label = '$W_J$' , color = '#1f77b4' , linewidth=4)

  # marginal reversiblity
  plt.plot(( F_m )[:t] , label = '$\Delta F_M$' , color = 'orange', linewidth=2)

  plt.plot((w)[:t] , label = '$W_M$' , color = '#1f77b4',  linewidth=2)


  plt.title('Reversibility panel')
  plt.xlabel('time | epoch  ')
  plt.ylabel('energy |  info ')
  if phase== True : plt.axvline(x=cc, color='g', linestyle=':', alpha = 0.6)
  plt.legend(fontsize="10")
  # plt.xticks(x_1, x_2)

  plt.tight_layout()
  plt.show()


  cl_test = np.asanyarray(trainer.ll)[:,0]
  acc =100- np.asanyarray(trainer.val)[:,0]

  fig, ax1 = plt.subplots(figsize=fig_size)


  # Plot cl_train and cl_test on the first y-axis
  ax1.plot(cl_test[:t]  , label = 'test loss')
  if phase== True : ax1.axvline(x=cc, color='g', linestyle='--')

  ax1.legend(loc="upper left")

  # Create a second y-axis and plot acc on it
  ax2 = ax1.twinx()

  # Scale the x values of acc
  x_values = np.linspace(0, len(cl_test[:t]), len(acc))
  ax2.plot(x_values, acc, label = 'accuracy', color='r')

  ax2.legend(loc="upper right")

  plt.tight_layout()
  plt.show()



  ##### more visualizations

  mport copy
last_step = {
          'epoch': epoch,
          'model_state_dict': copy.deepcopy(model.state_dict()),          # Make a deep copy
          'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),  # Make a deep copy
   }


save_mode = True

if save_mode == True :
    os.mkdir("/content/Data4")
    os.chdir(r"/content/Data4")

    losi = np.asanyarray(trainer.ll)
    val = np.asanyarray(trainer.val)
    ther  = np.asanyarray(trainer.thermo)

    np.save('ll',losi)
    np.save('val',val)
    np.save('thermo', ther)

    torch.save(last_step, 'last_step.pth')

else:
  ll = np.load('ll.npy')
  val = np.load('val.npy')
  thermo  =np.load('thermo.npy')

  losi = np.asanyarray(ll)
  val = np.asanyarray(val)
  ther  = np.asanyarray(thermo)
  last_step  = torch.load('last_step.pth')

cl_test  =  losi[:,0]
main_loss = losi[:,1]
m_reg =  losi[:,2]
j_reg =  losi[:,3]

epoch =30

cross_ge = val[:,1]
miss_ge  = val[:,0]
miss_test  = val[:,2]


w = ther[:,0]
Q = ther[:,1]
j_w = ther[:,2]
j_Q = ther[:,3]
m_E= ther[:,4]
j_E= ther[:,5]

j_en =  -1* ther[:,6]
F_j =  ther[:,7]
m_en  =  -1* ther[:,8]
F_m  =  ther[:,9]

M_info  = - Q - (w- F_m)
J_info  = - j_Q - (j_w- F_j)

fig_size = (4.77376504773765, 2.9503490538081323)
full_fig_size = (2* 4.77376504773765,2* 2.9503490538081323)
x_1, x_2 = np.arange(0,(epoch+1)*100,100) , np.arange(0,epoch+1,2)

t=-1
cc = 750
phase = False


fig, ax = plt.subplots(figsize=fig_size)  # Set figure size
plt.plot(w[:t], label='$w_x$')
plt.plot(Q[:t],label=  '$Q_x$', color='orange')
plt.plot(w[:t]+Q[:t], label = '$\Delta E_x$', color='green', alpha = 0.6)

if phase== True : plt.axvline(x=cc, color='g', linestyle='--')
plt.xlabel('time | epoch  ')
plt.ylabel('energy |  info ')
plt.legend(fontsize="10")
# plt.xticks(x_1, x_2)
plt.grid(True)
plt.tight_layout()
plt.title('Marginal energy flow')
plt.savefig('marginal en flow', dpi=300)  # Save with DPI of 300


plt.show()

fig, ax = plt.subplots(figsize=fig_size)  # Set figure size
plt.plot(j_w[:t], label='$w_{X,Y}$')
plt.plot(j_Q[:t],label=  '$Q_{X,Y}$', color='orange')
plt.plot(j_w[:t]+j_Q[:t], label = '$\Delta E_{X,Y}$', color='green', alpha = 0.6)
if phase== True : plt.axvline(x=cc, color='g', linestyle='--')

plt.xlabel('time | epoch  ')
plt.ylabel('energy |  info ')
plt.legend(fontsize="10")
# # plt.xticks(x_1, x_2)

plt.title('Joint energy flow')
plt.grid(True)
plt.tight_layout()
plt.savefig('Joint en flow', dpi=300)  # Save with DPI of 300

plt.show()

fig, ax = plt.subplots(figsize=fig_size)  # Set figure size
    # marginal reversiblity
    plt.plot(( F_m )[:t] , label = '$\Delta F_x$' , color = 'orange' )
    plt.plot((w)[:t] , label = '$w_x$' , color = 'blue' )
    # plt.xticks(x_1, x_2)
    plt.title('Marginal Reversibility')
    plt.xlabel('time | epoch  ')
    plt.ylabel('energy |  info ')
    plt.grid(True)
    if phase== True : plt.axvline(x=cc, color='g', linestyle='--')
    plt.legend(fontsize="10")
    plt.tight_layout()
    plt.savefig('marginal reversibility.png', dpi=300)  # Save with DPI of 300

fig, ax = plt.subplots(figsize=fig_size)  # Set figure size
    # marginal reversiblity
    plt.plot(( F_j )[:t] , label = '$\Delta F_{X,Y}$' , color = 'orange' )
    plt.plot((j_w)[:t] , label = '$w_{X,Y}$' , color = 'blue' )

    # plt.xticks(x_1, x_2)
    plt.title('Joint Reversibility')
    plt.xlabel('time | epoch  ')
    plt.ylabel('energy |  info ')
    plt.grid(True)
    if phase== True : plt.axvline(x=cc, color='g', linestyle='--')
    plt.legend(fontsize="10")
    plt.tight_layout()
    plt.savefig('joint reversibility.png', dpi=300)  # Save with DPI of 300

fig, ax = plt.subplots(figsize=fig_size)  # Set figure size
  # marginal reversiblity
  plt.plot(( F_j )[:t] , label = '$\Delta F_{X,Y}$' , color = 'orange' )
  plt.plot((j_w)[:t] , label = '$w_{X,Y}$' , color = 'blue' )

  plt.plot(( F_m )[:t] , label = '$\Delta F_x$' , color = 'orange', linestyle='--', dashes=(1, 1))
  plt.plot((w)[:t] , label = '$w_x$' , color = 'blue'  ,linestyle='--',  dashes=(1, 1))
  # plt.xticks(x_1, x_2)
  plt.title(' Reversibility')
  plt.xlabel('time | epoch  ')
  plt.ylabel('energy |  info ')
  plt.grid(True)
  if phase== True : plt.axvline(x=cc, color='g', linestyle='--')
  plt.legend(fontsize="10")
  plt.tight_layout()
  plt.savefig('reversibility.png', dpi=300)  # Save with DPI of 300

# phase = True
# cc = 21000-200
fig, ax = plt.subplots(figsize=fig_size)  # Set figure size

cmap = plt.cm.get_cmap('RdBu')
scatter = plt.scatter(M_info[:t],J_info[:t], c=np.arange(Q[:t].shape[0]), cmap=cmap)
plt.xlabel(r'$ I_{X;\Theta}$')
if phase== True :
   plt.scatter(M_info[cc], J_info[cc], color='green', s=100, alpha = 0.6, marker='*')  # 's' is size of the point

plt.ylabel(r'$I_{Y,X;\Theta} $')
plt.title('L-info')
# create colorbar
cbar = plt.colorbar(scatter, label = 'time')

# set colorbar ticks
# cbar.set_ticks(x_1)
# cbar.set_ticklabels(x_2)

plt.tight_layout()
plt.savefig('info.png', dpi=300)  # Save with DPI of 300
plt.show()

fig, ax = plt.subplots(figsize=fig_size)  # Set figure size

cmap = plt.cm.get_cmap('RdBu')
scatter = plt.scatter(-Q[:t],-j_Q[:t], c=np.arange(Q[:t].shape[0]), cmap=cmap)
plt.xlabel(r'$ - Q_X$')
if phase== True :
   plt.scatter(M_info[cc], J_info[cc], color='green', s=100, alpha = 0.6, marker='*')  # 's' is size of the point
plt.ylabel(r'$- Q_{Y,X}  $')

# create colorbar
cbar = plt.colorbar(scatter, label = 'time')
plt.title('M-info')
# # set colorbar ticks
# cbar.set_ticks(x_1)
# cbar.set_ticklabels(x_2)

plt.tight_layout()
plt.savefig('QQ.png', dpi=300)  # Save with DPI of 300
plt.show()

# Sample every 100th point
sampling_rate = 100

fig, ax = plt.subplots(figsize=fig_size)

# Plot lines for M-info vs J-info
plt.plot(M_info[:t:sampling_rate], J_info[:t:sampling_rate], label='M-info vs J-info', alpha=0.6)

# Plot lines for -Q vs -j_Q
plt.plot(-Q[:t:sampling_rate], -j_Q[:t:sampling_rate], label='-Q vs -j_Q', linestyle='--', alpha=0.6)

if phase == True:
    plt.scatter(M_info[cc], J_info[cc], color='green', s=100, alpha=0.6, marker='*')
    plt.axhline(y=-j_Q[cc], color='g', linestyle=':', alpha=0.5)

plt.xlabel(r'$ - Q_M / M-info$')
plt.ylabel(r'$- Q_J / J-info $')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('combined_plot.png', dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=fig_size)  # Set figure size
# joint reversiblity
plt.plot( j_w[:t] - F_j [:t] , label = '$\Sigma_{Y,X|\Theta}$' , color = 'red' )
plt.plot( w[:t] - F_m [:t] , label = '$\Sigma_{X|\Theta}$' , color = 'blue')


plt.title('Reversibility ')
plt.xlabel('time | epoch  ')
plt.ylabel('energy |  info ')
if phase== True : plt.axvline(x=cc, color='g', linestyle=':', alpha = 0.6)
plt.legend(fontsize="10")
# plt.xticks(x_1, x_2)
plt.grid(True)
plt.tight_layout()

plt.savefig('reversibility diff')
plt.show()

acc =100- miss_ge[:t]
acc_t = 100 - miss_test[:t]

fig, ax1 = plt.subplots(figsize=fig_size)

reg  = m_reg[:t]+j_reg[:t] - (m_reg[:t]+j_reg[:t]).mean()
# Plot cl_train and cl_test on the first y-axis
# ax1.plot(cl_test[:t]  , label = 'obj')
ax1.plot(main_loss[:t]  , label = 'obj')
ax1.plot( reg  , label = 'reg', alpha=0.4)
if phase== True : ax1.axvline(x=cc, color='g', linestyle='--')

ax1.legend(loc="upper left")

# Create a second y-axis and plot acc on it
ax2 = ax1.twinx()
plt.title('Loss function and accuracy')
# Scale the x values of acc
x_values = np.linspace(0, len(main_loss[:t]), len(acc))
ax2.plot(x_values, acc, label = 'test acc', color='r', linestyle = '--')
ax2.plot(x_values, acc_t, label = 'train acc', color='b', linestyle = '--')

plt.grid(True)
ax2.legend(loc="lower right")
# plt.xticks(x_1, x_2)
plt.tight_layout()
plt.savefig('loss_validate', dpi=300)

# check m_EBM
fig, ax1 = plt.subplots(figsize=fig_size)
sampler = Sampler( model, mode='marginal' ,  img_shape = (1,28,28), sample_size =128, from_scrach=from_scrach ,max_len=8192)
batch = next(iter(train_loader))

x, y= batch

inp_imgs = (0.25*torch.randn_like(x)).clamp_(min=-1.0, max=1.0).to(device)
inp_labs = None

# inp_imgs = x.to(device)

plt.imshow(inp_imgs[1,0,:,:].cpu().detach(),cmap='gray')
plt.show()

fig, ax1 = plt.subplots(figsize=fig_size)
generated , generated_labels= sampler.generate_samples(model, inp_imgs=inp_imgs,inp_labs=inp_labs ,steps=1000, step_size=1, return_img_per_step=False)

plt.imshow(generated[1,0,:,:].cpu().detach(),cmap='gray')
plt.tight_layout()

plt.savefig('generatedsample', dpi = 300)


# check j_EBM

fig, ax1 = plt.subplots(figsize=fig_size)
sampler = Sampler( model, mode='joint' ,  img_shape = (1,28,28), sample_size =128, from_scrach=from_scrach ,max_len=8192)
batch= next(iter(train_loader))

x, y= batch

inp_imgs = (0.25*torch.randn_like(x)).clamp_(min=-1.0, max=1.0).to(device)
inp_labs = (torch.multinomial(torch.tensor(labels).float(), y.shape[0], replacement=True) ).to(device)

# inp_imgs = x.to(device)
inp_labs = y.to(device)
plt.tight_layout()
plt.imshow(inp_imgs[1,0,:,:].cpu().detach(),cmap='gray')
plt.show()

fig, ax1 = plt.subplots(figsize=fig_size)
generated , generated_labels= sampler.generate_samples(model, inp_imgs=inp_imgs,inp_labs=inp_labs ,steps=1000, step_size=5, return_img_per_step=False)

plt.imshow(generated[1,0,:,:].cpu().detach(),cmap='gray')
print(generated_labels[1])

plt.savefig('jointsample', dpi = 300)
