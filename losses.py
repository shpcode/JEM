import torch
import torch.nn as nn

class compute_loss:
    """
    This class provides three options for optimization objectives:
        1) Minimizing the negative log-likelihood of the marginal model >> Learn a generative Energy-Based Model (EBM) P(X)
        2) Minimizing the negative log-likelihood of the joint model >> Learn a generative EBM P(X, Y)
        3) Minimizing the cross-entropy function >> Learn a classifier P(Y|X)

    Note: Learning (minimizing) the second objective and unlearning (maximizing) the first objective simultaneously
          leads to a fourth possibility, which is learning a generative classifier model.
    """

    def __init__(self, model, sampler_j, sampler_m, steps, noise):
        self.model = model
        self.sampler_j = sampler_j
        self.sampler_m = sampler_m
        self.crosEn = nn.CrossEntropyLoss()
        self.steps = steps

    def marginal(self, batch):
        real_imgs, _ = batch
        real_imgs = real_imgs.to(device)
        small_noise = torch.randn_like(real_imgs) * noise
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        fake_imgs, _ = self.sampler_m.sample_new_exmps(steps=self.steps, step_size=10)

        real_out = self.model(real_imgs)
        fake_out = self.model(fake_imgs)

        reg_loss = (real_out ** 2).mean() + (fake_out ** 2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()

        return cdiv_loss, reg_loss, fake_out, fake_imgs

    def joint(self, batch):
        real_imgs, real_labels = batch
        real_imgs = real_imgs.to(device)
        real_labels = real_labels.to(device)
        small_noise = torch.randn_like(real_imgs) * noise
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        fake_imgs, fake_labels = self.sampler_j.sample_new_exmps(steps=self.steps, step_size=10)

        real_out = self.model.logits(real_imgs).gather(1, real_labels.view(-1, 1))
        fake_out = self.model.logits(fake_imgs).gather(1, fake_labels.view(-1, 1))

        reg_loss = (real_out ** 2).mean() + (fake_out ** 2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()

        return cdiv_loss, reg_loss, fake_out, fake_imgs, fake_labels

    def classifier(self, x, y):
        y_ = self.model.logits(x).squeeze()
        cross_loss = self.crosEn(y_, y)
        return cross_loss








class Trainer:

    '''
    The method 'learn' conducts an optimization process over the selected loss function by specifying the 'mode' with the following options:

    - 'gcl': Generative Classifier, minimizing joint negative likelihood while maximizing marginal negative likelihood.
    - 'dcl': Discriminative Classifier, minimizing cross-entropy.
    - 'm_ebm': Generative Energy-Based Model (EBM), minimizing the negative likelihood of the marginal distribution.
    - 'j_ebm': Joint EBM, minimizing the negative likelihood of the joint distribution.
    '''
    def __init__(self, mode):
        self.ll = []
        self.val = []
        self.thermo= []

        self.W = 0
        self.Q = 0

        self.j_W = 0
        self.j_Q = 0


        self.first_step = True
        self.mode = mode

        self.delta_E_j = 0
        self.delta_E_m = 0


    def learn(self,model, gamma):

        step = 0
        while step<100:

          if step % int(len(train_set)/batch_size)==0:
            train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True,  num_workers=2, pin_memory=True)
            train_loader = iter(train_loader)

          batch = next(train_loader)
          x,y =batch ; x= x.to(device) ; y = y.to(device)
          step+=1
          optimizer.zero_grad()

          m_loss, m_reg, log_f0m , m_samples            = losses.marginal(batch )
          j_loss, j_reg, log_f0j , j_samples , j_labels = losses.joint(batch)

          is_nan = torch.stack([torch.isnan(p).any() for p in model.parameters()]).any().item()


          self.E_0   = -log_f0m.detach().cpu().numpy().astype(np.float64)
          self.j_E_0 = -log_f0j.detach().cpu().numpy().astype(np.float64)

          if self.first_step == False:
            self.Q +=  self.E_0.mean() - self.E_1.mean()
            self.j_Q += self.j_E_0.mean() - self.j_E_1.mean()
          else:
            print('no heat')
            self.first_step = False


          "###############" # defined model type
          if self.mode=='gcl':
            reg        =   m_reg + j_reg
            objective  =  j_loss - gamma * m_loss

          elif self.mode=='dcl':
            reg = torch.zeros(1).to(device)
            objective = losses.classifier(x,y)

          elif self.mode=='m_ebm':
            objective = m_loss
            reg       = m_reg

          elif self.mode=='j_ebm':
            objective = j_loss
            reg       = j_reg
          else:
            print('undifined training mode')

          "###############"

          loss       =  objective + 0.1 * reg
          loss.backward()
          optimizer.step()


          self.E_1   =  -model(m_samples).detach().cpu().numpy().astype(np.float64)
          self.j_E_1 =  -model.logits(j_samples).gather(1, j_labels.view(-1,1)).detach().cpu().numpy().astype(np.float64)

          self.W   += (self.E_1.mean()   - self.E_0.mean())
          self.j_W += (self.j_E_1.mean() - self.j_E_0.mean())

          m_E = - log_f0m.detach().cpu().numpy().mean()
          j_E = - log_f0j.detach().cpu().numpy().mean()

          x_,y_ = test_sample[0].to(device), test_sample[1].to(device)
          cl_test  = losses.classifier(x_,y_)
          # cl_train = losses.classifier(x ,y ).detach().cpu().numpy()

          m_en = -model(x_).mean().detach().cpu().numpy()
          j_en =- model.logits(x_).gather(1,y_.view(-1,1)).mean().detach().cpu().numpy()


          # online AIS
          self.delta_E_m += self.E_1 - self.E_0    # stochastics work
          c = np.median(self.delta_E_m)
          F_m = - np.log( np.exp( - self.delta_E_m + c ).mean() ) + c


          self.delta_E_j += self.j_E_1 - self.j_E_0    # stochastics work
          c = np.median(self.delta_E_j)
          F_j = - np.log( np.exp( -self.delta_E_j + c ).mean() ) + c


          self.ll.append([cl_test.item()  , objective.item(), m_reg.item() ,  j_reg.item() ])
          self.thermo.append([self.W, self.Q, self.j_W , self.j_Q, m_E, j_E, j_en, F_j, m_en , F_m])

        self.val.append([validate(model).miss_match(test_loader) , validate(model).cross_ent(test_loader), validate(model).miss_match(train_loader)])

        return model




class validate():

  '''this class prvovide standard validation tools '''

  def __init__(self, model):
    self.model = model

  def miss_match(self, loader):
      err = 0
      N = 0
      data = iter(loader)
      for x, y in data:
          x = x.to(device)
          y = y.to(device)
          y_ = self.model.logits(x)
          y_ = torch.argmax(torch.softmax(y_, dim=1), dim=1)
          err += ((y - y_) != 0).sum()
          N += x.shape[0]
      return (err/N * 100).detach().cpu().numpy()


  def cross_ent(self, loader):
    i=0 ; error=0
    crosEn = nn.CrossEntropyLoss(reduction = 'sum' )
    data = iter(loader)
    for batch in data:
      i=i+1
      x,y = batch
      x, y = x.to(device) , y.to(device)
      y_ = (self.model.logits(x))
      error = error + crosEn ( y_, y).detach().cpu().numpy()
    return error/len(loader.dataset)

